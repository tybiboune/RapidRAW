use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, GenericImageView, GrayImage, ImageFormat};
use serde::Serialize;
use serde_json::{Value, json};

use crate::ai::connector;
use crate::ai::models::{
    AiMaskModel, AiModels, AiState, ManualModel, get_or_init_ai_models, get_or_init_manual_model,
};
use crate::ai::processing::{
    AiDepthMaskParameters, AiFaceMaskParameters, AiForegroundMaskParameters, AiSkyMaskParameters,
    AiSubjectMaskParameters, AiTextMaskParameters, CachedDepthMap,
    ESAM3_FACE_ALTERNATIVE_BOX_SCORE, ESAM3_FACE_SCORE_THRESHOLD, ESAM3_SKY_REFINE_EPS,
    ESAM3_SKY_SCORE_THRESHOLD, ESAM3_TEXT_REFINE_EPS, ESAM3_TEXT_SCORE_THRESHOLD, Esam3Features,
    ImageEmbeddings, generate_image_embeddings, run_birefnet_model, run_depth_anything_model,
    run_esam3_boxes, run_esam3_decoder, run_esam3_image_encoder, run_esam3_zoomed_decoder,
    run_sam_decoder,
};
use crate::app::settings::load_settings;
use crate::app::state::{AiTaskGuard, AppState, LoadedImage};
use crate::effects::maps::EffectMapKind;
use crate::masks::face::{
    FaceAnalysis, FaceMethod, FaceParsing, FacePart, analyze_faces, compose_face_mask, parse_faces,
};
use crate::render::cache_utils::GEOMETRY_KEYS;
use crate::render::geometry::{get_geometry_params_from_json, is_geometry_identity};
use crate::render::preview::get_cached_full_warped_image;

const NO_FOREGROUND_FOUND: &str = "No foreground found";
const NO_SKY_FOUND: &str = "No sky found";
const NO_MATCH_FOUND: &str = "No match found for";
pub(crate) const NO_FACES_FOUND: &str = "No faces found";
const NO_RESULT_ERRORS: [&str; 4] = [
    NO_FOREGROUND_FOUND,
    NO_SKY_FOUND,
    NO_MATCH_FOUND,
    NO_FACES_FOUND,
];

fn is_no_result_error(error: &str) -> bool {
    NO_RESULT_ERRORS
        .iter()
        .any(|prefix| error.starts_with(prefix))
}

#[tauri::command]
pub fn cancel_ai_task(task_id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let tasks = state.active_ai_tasks.lock().unwrap();
    if let Some(token) = tasks.get(&task_id) {
        token.cancel();
    }
    Ok(())
}

fn encode_to_base64_png(image: &GrayImage) -> Result<String, String> {
    let mut buf = Cursor::new(Vec::new());
    image
        .write_to(&mut buf, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    let base64_str = general_purpose::STANDARD.encode(buf.get_ref());
    Ok(format!("data:image/png;base64,{}", base64_str))
}

#[derive(Clone, Copy)]
pub struct MaskTransform {
    pub rotation: f32,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub orientation_steps: u8,
}

impl MaskTransform {
    pub fn from_adjustments(adjustments: &serde_json::Value) -> Self {
        Self {
            rotation: adjustments["rotation"].as_f64().unwrap_or(0.0) as f32,
            flip_horizontal: adjustments["flipHorizontal"].as_bool().unwrap_or(false),
            flip_vertical: adjustments["flipVertical"].as_bool().unwrap_or(false),
            orientation_steps: adjustments["orientationSteps"].as_u64().unwrap_or(0) as u8,
        }
    }

    pub fn box_space(&self, (width, height): (u32, u32)) -> (u32, u32) {
        if self.orientation_steps % 2 == 1 {
            (height, width)
        } else {
            (width, height)
        }
    }
}

pub fn to_json<T: Serialize>(params: T) -> Result<Value, String> {
    serde_json::to_value(params).map_err(|e| e.to_string())
}

pub fn or_empty_mask(
    result: Result<Value, String>,
    transform: MaskTransform,
    ai_source: Option<String>,
) -> Result<(Value, Option<String>), String> {
    match result {
        Err(error) if is_no_result_error(&error) => {
            let mut params = json!({
                "maskDataBase64": encode_to_base64_png(&GrayImage::new(1, 1))?,
                "rotation": transform.rotation,
                "flipHorizontal": transform.flip_horizontal,
                "flipVertical": transform.flip_vertical,
                "orientationSteps": transform.orientation_steps,
            });
            if let Some(ai_source) = ai_source {
                params["aiSource"] = json!(ai_source);
            }
            Ok((params, Some(error)))
        }
        result => result.map(|params| (params, None)),
    }
}

fn mask_command_result(
    result: Result<Value, String>,
    transform: MaskTransform,
    ai_source: Option<String>,
) -> Result<Value, String> {
    let (mut params, no_result) = or_empty_mask(result, transform, ai_source)?;
    if let Some(error) = no_result {
        params["noResult"] = json!(error);
    }
    Ok(params)
}

pub fn ai_cache_key(path: &str, adjustments: &serde_json::Value) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(path.as_bytes());
    let mut geo_hasher = DefaultHasher::new();
    for key in GEOMETRY_KEYS {
        if let Some(val) = adjustments.get(key) {
            key.hash(&mut geo_hasher);
            val.to_string().hash(&mut geo_hasher);
        }
    }
    hasher.update(&geo_hasher.finish().to_le_bytes());
    hasher.finalize().to_hex().to_string()
}

/// The key the skin tools keep their face analysis under. The mask panel finds faces in the picture
/// after geometry (crop, rotation, perspective), the skin tools in the picture before it, so the two
/// can only share one analysis, and with it the face detection already done, while geometry leaves the
/// picture as it is. Otherwise the skin tools get a key of their own, so neither is handed landmarks
/// that sit in the other's space.
fn skin_cache_key(path: &str, adjustments: &serde_json::Value) -> String {
    let key = ai_cache_key(path, adjustments);
    let geometry = get_geometry_params_from_json(path, adjustments);
    if is_geometry_identity(&geometry) {
        key
    } else {
        format!("{key}-source")
    }
}

fn loaded_source_id(state: &tauri::State<'_, AppState>) -> Option<String> {
    let loaded = state.original_image.lock().unwrap().clone()?;
    crate::ai::jobs::source_id_for(&loaded.path)
}

fn check_cancelled(guard: &Option<AiTaskGuard>) -> Result<(), String> {
    if guard.as_ref().is_some_and(|g| g.token.is_cancelled()) {
        return Err("Task cancelled".to_string());
    }
    Ok(())
}

pub fn generate_foreground_mask(
    models: &AiModels,
    warped_image: &DynamicImage,
    transform: MaskTransform,
    ai_source: Option<String>,
) -> Result<AiForegroundMaskParameters, String> {
    let mask = run_birefnet_model(
        warped_image,
        &*models.birefnet().map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if mask.iter().all(|&v| v < 128) {
        return Err(NO_FOREGROUND_FOUND.to_string());
    }

    Ok(AiForegroundMaskParameters {
        mask_data_base64: Some(encode_to_base64_png(&mask)?),
        rotation: Some(transform.rotation),
        flip_horizontal: Some(transform.flip_horizontal),
        flip_vertical: Some(transform.flip_vertical),
        orientation_steps: Some(transform.orientation_steps),
        ai_source,
    })
}

fn cached_esam3_features(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    cache_key: &str,
    warped_image: &DynamicImage,
) -> Result<Arc<Esam3Features>, String> {
    if let Some(cached) = ai_state
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|s| s.esam3_features.clone())
        .filter(|cached| cached.path_hash == cache_key)
    {
        return Ok(cached);
    }

    let mut features = run_esam3_image_encoder(
        warped_image,
        &*models.esam3_image_encoder().map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    features.path_hash = cache_key.to_string();
    let features = Arc::new(features);

    if let Some(state) = ai_state.lock().unwrap().as_mut() {
        state.esam3_features = Some(features.clone());
    }
    Ok(features)
}

pub fn generate_sky_mask(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    cache_key: &str,
    warped_image: &DynamicImage,
    transform: MaskTransform,
    ai_source: Option<String>,
) -> Result<AiSkyMaskParameters, String> {
    let features = cached_esam3_features(ai_state, models, cache_key, warped_image)?;
    let prompt = models.esam3_sky_prompt().map_err(|e| e.to_string())?;
    let mask = run_esam3_decoder(
        &*models.esam3_decoder().map_err(|e| e.to_string())?,
        &features,
        &prompt,
        warped_image,
        ESAM3_SKY_SCORE_THRESHOLD,
        ESAM3_SKY_REFINE_EPS,
    )
    .map_err(|e| e.to_string())?
    .ok_or_else(|| NO_SKY_FOUND.to_string())?;

    Ok(AiSkyMaskParameters {
        mask_data_base64: Some(encode_to_base64_png(&mask)?),
        rotation: Some(transform.rotation),
        flip_horizontal: Some(transform.flip_horizontal),
        flip_vertical: Some(transform.flip_vertical),
        orientation_steps: Some(transform.orientation_steps),
        ai_source,
    })
}

pub fn generate_text_mask(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    cache_key: &str,
    warped_image: &DynamicImage,
    prompt: &str,
    transform: MaskTransform,
    ai_source: Option<String>,
) -> Result<AiTextMaskParameters, String> {
    let features = cached_esam3_features(ai_state, models, cache_key, warped_image)?;
    let text_prompt = models
        .esam3_text_prompt(prompt)
        .map_err(|e| e.to_string())?;
    let mask = run_esam3_zoomed_decoder(
        &*models.esam3_image_encoder().map_err(|e| e.to_string())?,
        &*models.esam3_decoder().map_err(|e| e.to_string())?,
        &features,
        &text_prompt,
        warped_image,
        ESAM3_TEXT_SCORE_THRESHOLD,
        ESAM3_TEXT_REFINE_EPS,
    )
    .map_err(|e| e.to_string())?
    .ok_or_else(|| format!("{} \"{}\"", NO_MATCH_FOUND, prompt.trim()))?;

    Ok(AiTextMaskParameters {
        prompt: prompt.to_string(),
        mask_data_base64: Some(encode_to_base64_png(&mask)?),
        rotation: Some(transform.rotation),
        flip_horizontal: Some(transform.flip_horizontal),
        flip_vertical: Some(transform.flip_vertical),
        orientation_steps: Some(transform.orientation_steps),
        ai_source,
    })
}

pub(crate) fn cached_face_analysis(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    cache_key: &str,
    warped_image: &DynamicImage,
    copies: Option<&DetectionCopies>,
    transform: MaskTransform,
) -> Result<Arc<FaceAnalysis>, String> {
    if let Some(cached) = ai_state
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|s| s.face_analysis.clone())
        .filter(|cached| cached.path_hash == cache_key && !cached.faces.is_empty())
    {
        return Ok(cached);
    }

    // Only a real result is worth remembering. A failure, or an analysis with no face in it, is
    // indistinguishable from a model that was set up badly, so it is never cached and it throws the
    // sessions away: the next request re-verifies the model files and builds new sessions instead of
    // reusing the same ones for the rest of the run.
    //
    // A failure also gets one immediate second try on the fresh sessions. That is what makes a GPU
    // that dies mid-inference recoverable in the same click: the failure switches the AI models to
    // the CPU, and the rebuilt sessions are CPU ones.
    let started = std::time::Instant::now();
    log::info!("Face analysis started");
    let mut result = compute_face_analysis(ai_state, models, cache_key, warped_image, transform);
    if let Err(error) = &result {
        log::warn!("Face analysis failed, rebuilding the face models and retrying: {error}");
        reset_face_models(ai_state, models);
        result = compute_face_analysis(ai_state, models, cache_key, warped_image, transform);
    }
    let analysis = match result {
        Ok(analysis) => Arc::new(
            keep_the_fullest_faces(models, cache_key, warped_image, analysis, copies, transform).0,
        ),
        Err(error) => {
            log::error!("Face analysis failed after a retry: {error}");
            reset_face_models(ai_state, models);
            return Err(error);
        }
    };
    log::info!(
        "Face analysis found {} face(s) in {:?}",
        analysis.faces.len(),
        started.elapsed()
    );

    if analysis.faces.is_empty() {
        reset_face_models(ai_state, models);
    } else if let Some(state) = ai_state.lock().unwrap().as_mut() {
        state.face_analysis = Some(analysis.clone());
    }
    Ok(analysis)
}

pub(crate) fn compute_face_analysis(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    cache_key: &str,
    warped_image: &DynamicImage,
    transform: MaskTransform,
) -> Result<FaceAnalysis, String> {
    let features = cached_esam3_features(ai_state, models, cache_key, warped_image)?;
    faces_from_features(models, &features, cache_key, warped_image, transform)
}

/// The faces on a picture from its encoded features: the boxes the detector puts round them, then the
/// landmarks fitted in each.
fn faces_from_features(
    models: &AiModels,
    features: &Esam3Features,
    cache_key: &str,
    warped_image: &DynamicImage,
    transform: MaskTransform,
) -> Result<FaceAnalysis, String> {
    let detections = run_esam3_boxes(
        &*models.esam3_decoder().map_err(|e| e.to_string())?,
        features,
        &*models.esam3_face_prompt().map_err(|e| e.to_string())?,
        ESAM3_FACE_ALTERNATIVE_BOX_SCORE,
    )
    .map_err(|e| e.to_string())?;
    analyze_faces(
        warped_image,
        &detections,
        ESAM3_FACE_SCORE_THRESHOLD,
        &*models.face_landmarks().map_err(|e| e.to_string())?,
        &*models.face_segmenter().map_err(|e| e.to_string())?,
        -(transform.orientation_steps as f32) * std::f32::consts::FRAC_PI_2,
        cache_key.to_string(),
    )
    .map_err(|e| e.to_string())
}

/// The copies of the loaded photo the faces are looked for in as well as in the photo itself: the app's
/// own Auto, as its button sets it, and the same Auto with Shadows and Blacks at +100, rendered as the app
/// renders. Depending on the light, the faces are found best, or most fully, on the photo, on the one or
/// on the other copy, and whichever shows the most face skin is kept (`keep_the_fullest_faces`). The
/// copies are only for finding the faces and what is looked for on them: they are never shown or
/// exported.
pub(crate) struct DetectionCopies<'a> {
    context: crate::render::gpu::context::GpuContext,
    state: &'a AppState,
    loaded: &'a LoadedImage,
    adjustments: &'a Value,
    render: crate::render::options::RenderOptions,
    /// whether the copies are put in the photo's cropped and turned frame, as the masks' picture is, or
    /// left in the photo's own frame, as the skin tools' picture is
    warped: bool,
}

impl<'a> DetectionCopies<'a> {
    pub(crate) fn new(
        context: crate::render::gpu::context::GpuContext,
        state: &'a AppState,
        loaded: &'a LoadedImage,
        adjustments: &'a Value,
        render: crate::render::options::RenderOptions,
        warped: bool,
    ) -> Self {
        Self {
            context,
            state,
            loaded,
            adjustments,
            render,
            warped,
        }
    }

    /// The copies of the loaded photo in the app. None when the GPU can't be had: the faces are then
    /// looked for in the photo alone.
    pub(crate) fn for_app(
        state: &'a AppState,
        app_handle: &tauri::AppHandle,
        loaded: &'a LoadedImage,
        adjustments: &'a Value,
        warped: bool,
    ) -> Option<Self> {
        let context = crate::render::gpu::context::get_or_init_gpu_context(state, app_handle)
            .map_err(|e| log::warn!("No copies of the photo to find the faces in: {e}"))
            .ok()?;
        let render =
            crate::render::options::resolve_render_options_from_handle(app_handle, loaded.is_raw);
        Some(Self::new(
            context,
            state,
            loaded,
            adjustments,
            render,
            warped,
        ))
    }

    /// The copies, each with what it is called in the log. One that fails to render is left out.
    pub(crate) fn make(&self) -> Vec<(&'static str, DynamicImage)> {
        let base = crate::adjustments::negative_conversion::apply_negative_conversion(
            std::borrow::Cow::Borrowed(self.loaded.image.as_ref()),
            self.adjustments,
        );
        let is_raw = crate::adjustments::negative_conversion::effective_is_raw(
            self.loaded.is_raw,
            self.adjustments,
        );
        let exposure = crate::render::adjustment_utils::effective_baseline_exposure(
            &json!({}),
            self.loaded.as_shot,
            self.render.baseline,
        );
        let auto = crate::adjustments::auto_adjust::auto_results_to_json(
            &crate::adjustments::auto_adjust::perform_auto_analysis(&base, exposure),
            crate::adjustments::auto_adjust::AutoAdjustMode::Normal,
        );
        let mut lifted = auto.clone();
        lifted["shadows"] = json!(100.0);
        lifted["blacks"] = json!(100.0);
        // both copies are rendered from the same picture, which the GPU then keeps between the two
        let mut hasher = DefaultHasher::new();
        "face detection copy".hash(&mut hasher);
        self.loaded.path.hash(&mut hasher);
        self.adjustments.to_string().hash(&mut hasher);
        let id = hasher.finish();
        [
            ("Auto", auto),
            ("Auto with Shadows and Blacks at +100", lifted),
        ]
        .into_iter()
        .filter_map(|(name, settings)| {
            let copy = self
                .render_copy(&base, id, is_raw, &settings)
                .map_err(|e| log::warn!("The {name} copy to find the faces in failed: {e}"))
                .ok()?;
            let copy = DynamicImage::ImageRgb8(copy.to_rgb8());
            let copy = if self.warped {
                crate::render::geometry::apply_geometry_warp(
                    copy,
                    &self.loaded.path,
                    self.adjustments,
                )
                .into_owned()
            } else {
                copy
            };
            Some((name, copy))
        })
        .collect()
    }

    fn render_copy(
        &self,
        base: &DynamicImage,
        id: u64,
        is_raw: bool,
        settings: &Value,
    ) -> Result<DynamicImage, String> {
        let mut adjustments = crate::render::uniforms::get_all_adjustments_from_json(
            settings,
            is_raw,
            self.loaded.as_shot,
            self.render,
        );
        adjustments.global.show_clipping = 0;
        let size = base.dimensions();
        crate::render::gpu::output::process_and_get_dynamic_image(
            &self.context,
            self.state,
            &crate::render::source::RenderSource::new(
                base,
                id,
                settings,
                crate::render::source::SourceTransform::default(),
            ),
            crate::render::gpu::processor::RenderRequest {
                adjustments,
                mask_bitmaps: &[],
                mask_cache_key: None,
                lut: None,
                roi: None,
                hsl_probe: false,
                motion_blur: crate::effects::motion_blur::MotionBlurs::new(
                    &self.loaded.path,
                    settings,
                    size,
                    (0.0, 0.0),
                    size.0.max(size.1) as f32,
                ),
            },
            "face_detection_copy",
        )
    }
}

/// How much face skin the landmarks of an analysis cover on its picture.
pub(crate) fn face_skin_area(picture: &DynamicImage, analysis: &FaceAnalysis) -> u64 {
    if analysis.faces.is_empty() {
        return 0;
    }
    compose_face_mask(
        picture,
        analysis,
        None,
        &[FacePart::FaceSkin],
        FaceMethod::Landmarks,
    )
    .pixels()
    .map(|p| p[0] as u64)
    .sum()
}

/// Looks for the faces on the copies of the photo too, and keeps whichever of the photo and its copies
/// shows the most face skin: a face in hard light, or half in shadow, is often found more fully, or only,
/// on a copy. On a tie the photo itself is kept.
pub(crate) fn keep_the_fullest_faces(
    models: &AiModels,
    cache_key: &str,
    picture: &DynamicImage,
    found: FaceAnalysis,
    copies: Option<&DetectionCopies>,
    transform: MaskTransform,
) -> (FaceAnalysis, &'static str) {
    let Some(copies) = copies else {
        return (found, "the photo as it is");
    };
    let started = std::time::Instant::now();
    let mut best_area = face_skin_area(picture, &found);
    log::info!(
        "Faces on the photo as it is: {} face(s), {best_area} of face skin",
        found.faces.len()
    );
    let mut best = found;
    let mut best_name = "the photo as it is";
    for (name, copy) in copies.make() {
        // a copy is only of use in the picture's own frame and size
        if copy.dimensions() != picture.dimensions() {
            log::warn!(
                "The {name} copy is {:?}, the picture {:?}: not looked in",
                copy.dimensions(),
                picture.dimensions()
            );
            continue;
        }
        let features = models
            .esam3_image_encoder()
            .map_err(|e| e.to_string())
            .and_then(|encoder| {
                run_esam3_image_encoder(&copy, &encoder).map_err(|e| e.to_string())
            });
        let analysis = features.and_then(|features| {
            faces_from_features(models, &features, cache_key, &copy, transform)
        });
        match analysis {
            Ok(mut analysis) => {
                let area = face_skin_area(&copy, &analysis);
                log::info!(
                    "Faces on the {name} copy: {} face(s), {area} of face skin",
                    analysis.faces.len()
                );
                if area > best_area {
                    best_area = area;
                    best_name = name;
                    analysis.found_on = Some(Arc::new(copy));
                    best = analysis;
                }
            }
            Err(e) => log::warn!("Looking for the faces on the {name} copy failed: {e}"),
        }
    }
    log::info!(
        "Kept the faces found on {best_name} (the copies took {:?})",
        started.elapsed()
    );
    (best, best_name)
}

fn reset_face_models(ai_state: &Mutex<Option<AiState>>, models: &AiModels) {
    models.invalidate_face_models();
    if let Some(state) = ai_state.lock().unwrap().as_mut() {
        state.forget_face_results();
    }
}

/// What the face parser made of the faces of an analysis, run once for it and kept with it. The parser's
/// files have to be in place first (`AiMaskModel::FaceParsing`).
pub(crate) fn cached_face_parsing(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    analysis: &FaceAnalysis,
    image: &DynamicImage,
) -> Result<Arc<FaceParsing>, String> {
    if let Some(cached) = ai_state
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|s| s.face_parsing.clone())
        .filter(|cached| cached.path_hash == analysis.path_hash)
    {
        return Ok(cached);
    }
    let session = models.face_parser().map_err(|e| e.to_string())?;
    let parsing = Arc::new(parse_faces(image, analysis, &session).map_err(|e| e.to_string())?);
    if let Some(state) = ai_state.lock().unwrap().as_mut() {
        state.face_parsing = Some(parsing.clone());
    }
    Ok(parsing)
}

/// Whether a face mask with these parameters needs the face parser.
pub fn face_mask_needs_parser(parts: &[String], method: Option<&str>) -> bool {
    let parts: Vec<FacePart> = parts.iter().filter_map(|p| FacePart::from_id(p)).collect();
    FaceMethod::from_id(method).needs_parsing(&parts)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_face_mask(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    cache_key: &str,
    warped_image: &DynamicImage,
    parts: &[String],
    method: Option<String>,
    copies: Option<&DetectionCopies>,
    transform: MaskTransform,
    ai_source: Option<String>,
) -> Result<AiFaceMaskParameters, String> {
    let analysis =
        cached_face_analysis(ai_state, models, cache_key, warped_image, copies, transform)?;
    if analysis.faces.is_empty() {
        return Err(NO_FACES_FOUND.to_string());
    }
    // the parts are found on the picture the faces were found on
    let warped_image = analysis.picture(warped_image);
    let face_parts: Vec<FacePart> = parts.iter().filter_map(|p| FacePart::from_id(p)).collect();
    let face_method = FaceMethod::from_id(method.as_deref());
    let parsing = if face_method.needs_parsing(&face_parts) {
        Some(cached_face_parsing(
            ai_state,
            models,
            &analysis,
            warped_image,
        )?)
    } else {
        None
    };
    let mask = compose_face_mask(
        warped_image,
        &analysis,
        parsing.as_deref(),
        &face_parts,
        face_method,
    );

    Ok(AiFaceMaskParameters {
        parts: parts.to_vec(),
        method,
        mask_data_base64: Some(encode_to_base64_png(&mask)?),
        rotation: Some(transform.rotation),
        flip_horizontal: Some(transform.flip_horizontal),
        flip_vertical: Some(transform.flip_vertical),
        orientation_steps: Some(transform.orientation_steps),
        ai_source,
    })
}

fn cached_depth_estimate(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    cache_key: &str,
    warped_image: &DynamicImage,
) -> Result<CachedDepthMap, String> {
    if let Some(cached) = ai_state
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|s| s.depth_map.clone())
        .filter(|cached| cached.path_hash == cache_key)
    {
        return Ok(cached);
    }

    let depth_image = run_depth_anything_model(
        warped_image,
        &*models.depth_anything().map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let depth = CachedDepthMap {
        path_hash: cache_key.to_string(),
        depth_image,
        original_size: warped_image.dimensions(),
    };
    if let Some(state) = ai_state.lock().unwrap().as_mut() {
        state.depth_map = Some(depth.clone());
    }
    Ok(depth)
}

pub fn generate_depth_mask(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    cache_key: &str,
    warped_image: &DynamicImage,
    params: AiDepthMaskParameters,
    transform: MaskTransform,
    ai_source: Option<String>,
) -> Result<AiDepthMaskParameters, String> {
    let depth = cached_depth_estimate(ai_state, models, cache_key, warped_image)?;
    let raw_depth_fullres = image::imageops::resize(
        &depth.depth_image,
        depth.original_size.0,
        depth.original_size.1,
        image::imageops::FilterType::Triangle,
    );

    Ok(AiDepthMaskParameters {
        mask_data_base64: Some(encode_to_base64_png(&raw_depth_fullres)?),
        rotation: Some(transform.rotation),
        flip_horizontal: Some(transform.flip_horizontal),
        flip_vertical: Some(transform.flip_vertical),
        orientation_steps: Some(transform.orientation_steps),
        ai_source,
        ..params
    })
}

fn cached_image_embeddings(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    cache_key: &str,
    warped_image: &DynamicImage,
) -> Result<ImageEmbeddings, String> {
    if let Some(cached) = ai_state
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|s| s.embeddings.clone())
        .filter(|cached| cached.path_hash == cache_key)
    {
        return Ok(cached);
    }

    let mut embeddings = generate_image_embeddings(
        warped_image,
        &*models.sam_encoder().map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    embeddings.path_hash = cache_key.to_string();

    if let Some(state) = ai_state.lock().unwrap().as_mut() {
        state.embeddings = Some(embeddings.clone());
    }
    Ok(embeddings)
}

fn unrotate_box(
    start_point: (f64, f64),
    end_point: (f64, f64),
    (img_w, img_h): (u32, u32),
    transform: MaskTransform,
) -> ((f64, f64), (f64, f64)) {
    let (coarse_rotated_w, coarse_rotated_h) = {
        let (w, h) = transform.box_space((img_w, img_h));
        (w as f64, h as f64)
    };
    let center = (coarse_rotated_w / 2.0, coarse_rotated_h / 2.0);

    let angle_rad = (transform.rotation as f64).to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let unrotate = |p: (f64, f64)| {
        let px = p.0 - center.0;
        let py = p.1 - center.1;
        (
            px * cos_a + py * sin_a + center.0,
            -px * sin_a + py * cos_a + center.1,
        )
    };

    let unflip = |p: (f64, f64)| {
        let mut new_px = p.0;
        let mut new_py = p.1;
        if transform.flip_horizontal {
            new_px = coarse_rotated_w - p.0;
        }
        if transform.flip_vertical {
            new_py = coarse_rotated_h - p.1;
        }
        (new_px, new_py)
    };

    let un_coarse_rotate = |p: (f64, f64)| -> (f64, f64) {
        match transform.orientation_steps {
            0 => p,
            1 => (p.1, img_h as f64 - p.0),
            2 => (img_w as f64 - p.0, img_h as f64 - p.1),
            3 => (img_w as f64 - p.1, p.0),
            _ => p,
        }
    };

    let corners = [
        start_point,
        (start_point.0, end_point.1),
        end_point,
        (end_point.0, start_point.1),
    ]
    .map(|p| un_coarse_rotate(unflip(unrotate(p))));

    let min_x = corners.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let min_y = corners.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_x = corners
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_y = corners
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    ((min_x, min_y), (max_x, max_y))
}

#[allow(clippy::too_many_arguments)]
pub fn generate_subject_mask(
    ai_state: &Mutex<Option<AiState>>,
    models: &AiModels,
    cache_key: &str,
    warped_image: &DynamicImage,
    start_point: (f64, f64),
    end_point: (f64, f64),
    skip_refinement: bool,
    transform: MaskTransform,
    ai_source: Option<String>,
) -> Result<AiSubjectMaskParameters, String> {
    let embeddings = cached_image_embeddings(ai_state, models, cache_key, warped_image)?;
    let (unrotated_start, unrotated_end) =
        unrotate_box(start_point, end_point, embeddings.original_size, transform);

    let mask_bitmap = run_sam_decoder(
        &*models.sam_decoder().map_err(|e| e.to_string())?,
        &embeddings,
        unrotated_start,
        unrotated_end,
        (!skip_refinement).then_some(warped_image),
    )
    .map_err(|e| e.to_string())?;

    let (box_space_width, box_space_height) = transform.box_space(embeddings.original_size);

    Ok(AiSubjectMaskParameters {
        start_x: start_point.0,
        start_y: start_point.1,
        end_x: end_point.0,
        end_y: end_point.1,
        mask_data_base64: Some(encode_to_base64_png(&mask_bitmap)?),
        rotation: Some(transform.rotation),
        flip_horizontal: Some(transform.flip_horizontal),
        flip_vertical: Some(transform.flip_vertical),
        orientation_steps: Some(transform.orientation_steps),
        box_space_width: Some(box_space_width),
        box_space_height: Some(box_space_height),
        ai_source,
    })
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn generate_ai_foreground_mask(
    js_adjustments: serde_json::Value,
    rotation: f32,
    flip_horizontal: bool,
    flip_vertical: bool,
    orientation_steps: u8,
    task_id: Option<String>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Value, String> {
    let guard = task_id.map(|id| AiTaskGuard::new(&state.active_ai_tasks, id));
    check_cancelled(&guard)?;

    let models = get_or_init_ai_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
        AiMaskModel::Foreground,
    )
    .await
    .map_err(|e| e.to_string())?;
    check_cancelled(&guard)?;

    let warped_image = get_cached_full_warped_image(&state, &js_adjustments)?;
    check_cancelled(&guard)?;

    let transform = MaskTransform {
        rotation,
        flip_horizontal,
        flip_vertical,
        orientation_steps,
    };
    let ai_source = loaded_source_id(&state);
    let params = mask_command_result(
        generate_foreground_mask(&models, &warped_image, transform, ai_source.clone())
            .and_then(to_json),
        transform,
        ai_source,
    )?;
    check_cancelled(&guard)?;
    Ok(params)
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn generate_ai_sky_mask(
    js_adjustments: serde_json::Value,
    path: String,
    rotation: f32,
    flip_horizontal: bool,
    flip_vertical: bool,
    orientation_steps: u8,
    task_id: Option<String>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Value, String> {
    let guard = task_id.map(|id| AiTaskGuard::new(&state.active_ai_tasks, id));
    check_cancelled(&guard)?;

    let models = get_or_init_ai_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
        AiMaskModel::Sky,
    )
    .await
    .map_err(|e| e.to_string())?;
    check_cancelled(&guard)?;

    let warped_image = get_cached_full_warped_image(&state, &js_adjustments)?;
    check_cancelled(&guard)?;

    let transform = MaskTransform {
        rotation,
        flip_horizontal,
        flip_vertical,
        orientation_steps,
    };
    let ai_source = loaded_source_id(&state);
    let params = mask_command_result(
        generate_sky_mask(
            &state.ai_state,
            &models,
            &ai_cache_key(&path, &js_adjustments),
            &warped_image,
            transform,
            ai_source.clone(),
        )
        .and_then(to_json),
        transform,
        ai_source,
    )?;
    check_cancelled(&guard)?;
    Ok(params)
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn generate_ai_text_mask(
    js_adjustments: serde_json::Value,
    path: String,
    prompt: String,
    rotation: f32,
    flip_horizontal: bool,
    flip_vertical: bool,
    orientation_steps: u8,
    task_id: Option<String>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Value, String> {
    let guard = task_id.map(|id| AiTaskGuard::new(&state.active_ai_tasks, id));
    check_cancelled(&guard)?;

    let models = get_or_init_ai_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
        AiMaskModel::Text,
    )
    .await
    .map_err(|e| e.to_string())?;
    check_cancelled(&guard)?;

    let warped_image = get_cached_full_warped_image(&state, &js_adjustments)?;
    check_cancelled(&guard)?;

    let transform = MaskTransform {
        rotation,
        flip_horizontal,
        flip_vertical,
        orientation_steps,
    };
    let ai_source = loaded_source_id(&state);
    let params = mask_command_result(
        generate_text_mask(
            &state.ai_state,
            &models,
            &ai_cache_key(&path, &js_adjustments),
            &warped_image,
            &prompt,
            transform,
            ai_source.clone(),
        )
        .and_then(to_json),
        transform,
        ai_source,
    )?;
    check_cancelled(&guard)?;
    Ok(params)
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn generate_ai_face_mask(
    js_adjustments: serde_json::Value,
    path: String,
    parts: Vec<String>,
    method: Option<String>,
    rotation: f32,
    flip_horizontal: bool,
    flip_vertical: bool,
    orientation_steps: u8,
    task_id: Option<String>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Value, String> {
    let guard = task_id.map(|id| AiTaskGuard::new(&state.active_ai_tasks, id));
    check_cancelled(&guard)?;

    let models = get_or_init_ai_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
        AiMaskModel::Face,
    )
    .await
    .map_err(|e| e.to_string())?;
    if face_mask_needs_parser(&parts, method.as_deref()) {
        get_or_init_ai_models(
            &app_handle,
            &state.ai_state,
            &state.ai_init_lock,
            AiMaskModel::FaceParsing,
        )
        .await
        .map_err(|e| e.to_string())?;
    }
    check_cancelled(&guard)?;

    let warped_image = get_cached_full_warped_image(&state, &js_adjustments)?;
    check_cancelled(&guard)?;

    let transform = MaskTransform {
        rotation,
        flip_horizontal,
        flip_vertical,
        orientation_steps,
    };
    let ai_source = loaded_source_id(&state);
    let loaded_image = get_loaded_image(&state)?;
    let copies =
        DetectionCopies::for_app(&state, &app_handle, &loaded_image, &js_adjustments, true);
    let params = mask_command_result(
        generate_face_mask(
            &state.ai_state,
            &models,
            &ai_cache_key(&path, &js_adjustments),
            &warped_image,
            &parts,
            method,
            copies.as_ref(),
            transform,
            ai_source.clone(),
        )
        .and_then(to_json),
        transform,
        ai_source,
    )?;
    check_cancelled(&guard)?;
    Ok(params)
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn generate_ai_depth_mask(
    js_adjustments: serde_json::Value,
    path: String,
    min_depth: f32,
    max_depth: f32,
    min_fade: f32,
    max_fade: f32,
    feather: f32,
    rotation: f32,
    flip_horizontal: bool,
    flip_vertical: bool,
    orientation_steps: u8,
    task_id: Option<String>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<AiDepthMaskParameters, String> {
    let guard = task_id.map(|id| AiTaskGuard::new(&state.active_ai_tasks, id));
    check_cancelled(&guard)?;

    let models = get_or_init_ai_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
        AiMaskModel::Depth,
    )
    .await
    .map_err(|e| e.to_string())?;
    check_cancelled(&guard)?;

    let warped_image = get_cached_full_warped_image(&state, &js_adjustments)?;
    check_cancelled(&guard)?;

    let transform = MaskTransform {
        rotation,
        flip_horizontal,
        flip_vertical,
        orientation_steps,
    };
    let params = AiDepthMaskParameters {
        min_depth,
        max_depth,
        min_fade,
        max_fade,
        feather,
        ..Default::default()
    };
    let params = generate_depth_mask(
        &state.ai_state,
        &models,
        &ai_cache_key(&path, &js_adjustments),
        &warped_image,
        params,
        transform,
        loaded_source_id(&state),
    )?;
    check_cancelled(&guard)?;
    Ok(params)
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn generate_ai_subject_mask(
    js_adjustments: serde_json::Value,
    path: String,
    start_point: (f64, f64),
    end_point: (f64, f64),
    rotation: f32,
    flip_horizontal: bool,
    flip_vertical: bool,
    orientation_steps: u8,
    skip_refinement: Option<bool>,
    task_id: Option<String>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<AiSubjectMaskParameters, String> {
    let guard = task_id.map(|id| AiTaskGuard::new(&state.active_ai_tasks, id));
    check_cancelled(&guard)?;

    let models = get_or_init_ai_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
        AiMaskModel::Subject,
    )
    .await
    .map_err(|e| e.to_string())?;
    check_cancelled(&guard)?;

    let warped_image = get_cached_full_warped_image(&state, &js_adjustments)?;
    check_cancelled(&guard)?;

    let transform = MaskTransform {
        rotation,
        flip_horizontal,
        flip_vertical,
        orientation_steps,
    };
    let params = generate_subject_mask(
        &state.ai_state,
        &models,
        &ai_cache_key(&path, &js_adjustments),
        &warped_image,
        start_point,
        end_point,
        skip_refinement.unwrap_or(false),
        transform,
        loaded_source_id(&state),
    )?;
    check_cancelled(&guard)?;
    Ok(params)
}

#[tauri::command]
pub async fn precompute_ai_subject_mask(
    js_adjustments: serde_json::Value,
    path: String,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let models = get_or_init_ai_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
        AiMaskModel::Subject,
    )
    .await
    .map_err(|e| e.to_string())?;

    let cache_key = ai_cache_key(&path, &js_adjustments);
    let is_cached = state
        .ai_state
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|s| s.embeddings.as_ref())
        .is_some_and(|cached| cached.path_hash == cache_key);
    if is_cached {
        return Ok(());
    }

    let warped_image = get_cached_full_warped_image(&state, &js_adjustments)?;
    cached_image_embeddings(&state.ai_state, &models, &cache_key, &warped_image)?;
    Ok(())
}

pub async fn generate_effect_map(
    app_handle: &tauri::AppHandle,
    kind: EffectMapKind,
    source_image: &DynamicImage,
    source_id: &str,
) -> Result<String, String> {
    let state = tauri::Manager::state::<AppState>(app_handle);
    let map = match kind {
        EffectMapKind::Depth => {
            let models = get_or_init_ai_models(
                app_handle,
                &state.ai_state,
                &state.ai_init_lock,
                AiMaskModel::Depth,
            )
            .await
            .map_err(|e| e.to_string())?;
            let session = models.depth_anything().map_err(|e| e.to_string())?;
            DynamicImage::ImageLuma8(
                run_depth_anything_model(source_image, &session).map_err(|e| e.to_string())?,
            )
        }
        EffectMapKind::Normal => {
            let session = crate::ai::models::get_or_init_normal_model(
                app_handle,
                &state.ai_state,
                &state.ai_init_lock,
            )
            .await
            .map_err(|e| e.to_string())?;
            DynamicImage::ImageRgba8(
                crate::ai::processing::run_normal_model(source_image, &session)
                    .map_err(|e| e.to_string())?,
            )
        }
    };

    crate::effects::maps::store_source_map(kind, source_id, &map)
}

async fn generate_loaded_effect_map(
    kind: EffectMapKind,
    js_adjustments: serde_json::Value,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let loaded_image = get_loaded_image(&state)?;
    let source_id = crate::ai::jobs::source_id_for(&loaded_image.path)
        .ok_or("Unable to identify the loaded image")?;
    let source_image = crate::masks::generation::build_converted_source_image(
        &loaded_image.image,
        loaded_image.is_raw,
        &js_adjustments,
    );
    generate_effect_map(&app_handle, kind, &source_image, &source_id).await
}

#[tauri::command]
pub async fn generate_full_image_depth_map(
    js_adjustments: serde_json::Value,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    generate_loaded_effect_map(EffectMapKind::Depth, js_adjustments, state, app_handle).await
}

/// The face skin of the loaded picture, as one map over the whole frame, for the skin tools to work
/// in. It is the same face-part masking the mask panel offers, asked for one part and kept as a map
/// rather than as a mask, so the skin sliders need no mask of their own to be drawn first.
#[tauri::command]
pub async fn generate_full_image_face_map(
    js_adjustments: serde_json::Value,
    path: String,
    method: Option<String>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<FaceMaps, String> {
    let models = get_or_init_ai_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
        AiMaskModel::Face,
    )
    .await
    .map_err(|e| e.to_string())?;
    // the skin the tool works on can be found by the face parser as well as, or instead of, the landmarks
    let face_method = FaceMethod::from_id(method.as_deref());
    if face_method.needs_parsing(&[FacePart::FaceSkin]) {
        get_or_init_ai_models(
            &app_handle,
            &state.ai_state,
            &state.ai_init_lock,
            AiMaskModel::FaceParsing,
        )
        .await
        .map_err(|e| e.to_string())?;
    }

    let loaded_image = get_loaded_image(&state)?;
    // the picture as the effects see it, before any geometry: the map is put back through the warp
    // when it is used, the same way the depth map is
    let source_image = crate::masks::generation::build_converted_source_image(
        &loaded_image.image,
        loaded_image.is_raw,
        &js_adjustments,
    );
    let copies =
        DetectionCopies::for_app(&state, &app_handle, &loaded_image, &js_adjustments, false);
    let analysis = cached_face_analysis(
        &state.ai_state,
        &models,
        &skin_cache_key(&path, &js_adjustments),
        &source_image,
        copies.as_ref(),
        MaskTransform::from_adjustments(&js_adjustments),
    )?;
    if analysis.faces.is_empty() {
        return Err(NO_FACES_FOUND.to_string());
    }
    let picture = analysis.picture(&source_image);
    let parsing = if face_method.needs_parsing(&[FacePart::FaceSkin]) {
        Some(cached_face_parsing(
            &state.ai_state,
            &models,
            &analysis,
            picture,
        )?)
    } else {
        None
    };
    let mask = compose_face_mask(
        picture,
        &analysis,
        parsing.as_deref(),
        &[FacePart::FaceSkin],
        face_method,
    );
    let regions = crate::masks::face::compose_skin_regions(
        source_image.width(),
        source_image.height(),
        &analysis,
    );
    let source_space = |map: &image::GrayImage| -> Result<String, String> {
        let encoded = encode_to_base64_png(map)?;
        Ok(format!(
            "{}{}",
            crate::effects::maps::SOURCE_SPACE_MAP_PREFIX,
            encoded.trim_start_matches("data:image/png;base64,")
        ))
    };
    Ok(FaceMaps {
        face_map: source_space(&mask)?,
        lines_map: source_space(&regions.lines)?,
        under_eye_map: source_space(&regions.under_eye)?,
    })
}

/// The maps the skin tools work in, all in the picture's own space: the face skin, where a face tends
/// to fold, and the band under each eye.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceMaps {
    face_map: String,
    lines_map: String,
    under_eye_map: String,
}

/// Where the lines are on the faces of the loaded picture, found by the FFHQ-Wrinkle model rather than
/// by the curvature of the brightness, for the skin tool's lines to be read from instead. It works on the
/// same faces the face map was made from, and like the other skin maps it is in the picture's own space
/// and put back through the warp when it is used.
#[tauri::command]
pub async fn generate_full_image_wrinkle_map(
    js_adjustments: serde_json::Value,
    path: String,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    skin_model_map(
        js_adjustments,
        path,
        &state,
        &app_handle,
        ManualModel::Wrinkles,
    )
    .await
}

/// Where the blemishes are on the faces of the loaded picture - spots, moles, milia, acne marks - found
/// by ABPN's blemish detector, for the skin tool's spots to be read from instead of from the shape of the
/// brightness alone. Made and used the same way as the wrinkle map.
#[tauri::command]
pub async fn generate_full_image_blemish_map(
    js_adjustments: serde_json::Value,
    path: String,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    skin_model_map(
        js_adjustments,
        path,
        &state,
        &app_handle,
        ManualModel::Blemishes,
    )
    .await
}

/// The two maps of the skin tool's learned retouch (ABPN, see masks::face::compose_ai_retouch) for the faces of
/// the loaded picture: the blend layer that evens the skin and the change that fills its spots, both as maps
/// in the picture's own space, kept with the edit so the retouch is laid on without any model.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiSkinRetouchMaps {
    blend_map: String,
    spots_map: String,
}

#[tauri::command]
pub async fn generate_ai_skin_retouch(
    js_adjustments: serde_json::Value,
    path: String,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<AiSkinRetouchMaps, String> {
    // the three models are checked for first: without them there is no point finding the faces
    let mut sessions = Vec::new();
    for model in [
        ManualModel::Blemishes,
        ManualModel::RetouchBlend,
        ManualModel::RetouchInpaint,
    ] {
        sessions.push(
            get_or_init_manual_model(&app_handle, &state.ai_state, &state.ai_init_lock, model)
                .await
                .map_err(|e| e.to_string())?,
        );
    }
    let models = get_or_init_ai_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
        AiMaskModel::Face,
    )
    .await
    .map_err(|e| e.to_string())?;
    let loaded_image = get_loaded_image(&state)?;
    let source_image = crate::masks::generation::build_converted_source_image(
        &loaded_image.image,
        loaded_image.is_raw,
        &js_adjustments,
    )
    .into_owned();
    let copies =
        DetectionCopies::for_app(&state, &app_handle, &loaded_image, &js_adjustments, false);
    let analysis = cached_face_analysis(
        &state.ai_state,
        &models,
        &skin_cache_key(&path, &js_adjustments),
        &source_image,
        copies.as_ref(),
        MaskTransform::from_adjustments(&js_adjustments),
    )?;
    if analysis.faces.is_empty() {
        return Err(NO_FACES_FOUND.to_string());
    }
    let maps = tauri::async_runtime::spawn_blocking(move || {
        // the blemishes are looked for where the faces were found, the skin is retouched on the photo
        let seen = analysis.found_on.clone();
        crate::masks::face::compose_ai_retouch(
            &source_image,
            seen.as_deref().unwrap_or(&source_image),
            &analysis,
            &sessions[0],
            &sessions[1],
            &sessions[2],
            crate::effects::skin::DETECT_LONG_SIDE,
        )
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    let encode = |map: &image::RgbImage| -> Result<String, String> {
        let mut buf = Cursor::new(Vec::new());
        map.write_to(&mut buf, ImageFormat::Png)
            .map_err(|e| e.to_string())?;
        Ok(format!(
            "{}{}",
            crate::effects::maps::SOURCE_SPACE_MAP_PREFIX,
            general_purpose::STANDARD.encode(buf.get_ref())
        ))
    };
    Ok(AiSkinRetouchMaps {
        blend_map: encode(&maps.blend)?,
        spots_map: encode(&maps.spots)?,
    })
}

// A skin map from one of the face models put in by hand, over the faces of the loaded picture, as a map
// in the picture's own space.
async fn skin_model_map(
    js_adjustments: serde_json::Value,
    path: String,
    state: &AppState,
    app_handle: &tauri::AppHandle,
    model: ManualModel,
) -> Result<String, String> {
    // the model is checked for first: without it there is no point finding the faces
    let session = get_or_init_manual_model(app_handle, &state.ai_state, &state.ai_init_lock, model)
        .await
        .map_err(|e| e.to_string())?;
    let models = get_or_init_ai_models(
        app_handle,
        &state.ai_state,
        &state.ai_init_lock,
        AiMaskModel::Face,
    )
    .await
    .map_err(|e| e.to_string())?;

    let loaded_image = get_loaded_image(state)?;
    // owned, because it goes on to the blocking task below, which may outlive the loaded picture
    let source_image = crate::masks::generation::build_converted_source_image(
        &loaded_image.image,
        loaded_image.is_raw,
        &js_adjustments,
    )
    .into_owned();
    let copies = DetectionCopies::for_app(state, app_handle, &loaded_image, &js_adjustments, false);
    let analysis = cached_face_analysis(
        &state.ai_state,
        &models,
        &skin_cache_key(&path, &js_adjustments),
        &source_image,
        copies.as_ref(),
        MaskTransform::from_adjustments(&js_adjustments),
    )?;
    if analysis.faces.is_empty() {
        return Err(NO_FACES_FOUND.to_string());
    }
    let picture = analysis
        .found_on
        .clone()
        .unwrap_or_else(|| Arc::new(source_image));
    let map = tauri::async_runtime::spawn_blocking(move || {
        let compose = match model {
            ManualModel::Wrinkles => crate::masks::face::compose_wrinkle_map,
            ManualModel::Blemishes => crate::masks::face::compose_blemish_map,
            ManualModel::DeblurMotion
            | ManualModel::DeblurDefocus
            | ManualModel::RetouchBlend
            | ManualModel::RetouchInpaint => {
                return Err(anyhow::anyhow!("{model:?} does not make a skin map"));
            }
        };
        compose(
            &picture,
            &analysis,
            &session,
            crate::effects::skin::DETECT_LONG_SIDE,
        )
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    let encoded = encode_to_base64_png(&map)?;
    Ok(format!(
        "{}{}",
        crate::effects::maps::SOURCE_SPACE_MAP_PREFIX,
        encoded.trim_start_matches("data:image/png;base64,")
    ))
}

#[tauri::command]
pub async fn generate_relight_normal_map(
    js_adjustments: serde_json::Value,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    generate_loaded_effect_map(EffectMapKind::Normal, js_adjustments, state, app_handle).await
}

#[tauri::command]
pub async fn get_depth_preview(
    js_adjustments: serde_json::Value,
    map_key: Option<String>,
    map_data: Option<String>,
    max_size: u32,
    state: tauri::State<'_, AppState>,
) -> Result<tauri::ipc::Response, String> {
    let loaded_image = get_loaded_image(&state)?;
    tauri::async_runtime::spawn_blocking(move || {
        crate::effects::maps::build_depth_preview(
            &loaded_image.image,
            loaded_image.is_raw,
            &loaded_image.path,
            &js_adjustments,
            map_key.as_deref(),
            map_data.as_deref(),
            max_size.clamp(16, 512),
        )
    })
    .await
    .map_err(|e| e.to_string())?
    .map(tauri::ipc::Response::new)
    .ok_or_else(|| "Depth map unavailable".to_string())
}

fn get_loaded_image(state: &AppState) -> Result<LoadedImage, String> {
    state
        .original_image
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .ok_or_else(|| "No original image loaded".to_string())
}

#[tauri::command]
pub async fn check_ai_connector_status(app_handle: tauri::AppHandle) {
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let is_connected = if let Some(address) = settings.ai_connector_address {
        connector::check_status(&address).await.unwrap_or(false)
    } else {
        false
    };
    use tauri::Emitter;
    let _ = app_handle.emit(
        "ai-connector-status-update",
        serde_json::json!({ "connected": is_connected }),
    );
}

#[tauri::command]
pub async fn test_ai_connector_connection(address: String) -> Result<(), String> {
    match connector::check_status(&address).await {
        Ok(true) => Ok(()),
        Ok(false) => Err("Server reachable but returned bad health status".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

#[cfg(test)]
mod detection_probe {
    use super::*;
    use std::sync::OnceLock;

    fn gpu() -> crate::render::gpu::context::GpuContext {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no wgpu adapter");
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .expect("device");
        crate::render::gpu::context::GpuContext {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            limits,
            display: Arc::new(Mutex::new(None)),
            viewer_display: Arc::new(Mutex::new(None)),
            pipelines: Arc::new(OnceLock::new()),
            compute: Arc::new(OnceLock::new()),
        }
    }

    /// For each raw in RRAW_PICTURES (paths split by `;`), finds the faces as the skin tools do: on the
    /// photo as it is and on its two copies, keeping the one with the most face skin. Reports each pass's
    /// faces and face skin, and the pass kept. RRAW_MODELS is the models folder; RRAW_OUT, if set, is a
    /// folder to save the kept picture with its face skin tinted in.
    #[test]
    #[ignore]
    fn face_detection_keeps_the_fullest_faces() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
        let models = AiModels::new(std::path::PathBuf::from(var("RRAW_MODELS")), false);
        let context = gpu();
        let _ = crate::render::gpu::context::TEST_GPU.set(context.clone());
        let state = AppState::new();
        let settings = crate::app::settings::AppSettings::default();
        let render = crate::render::options::resolve_render_options(&settings, true);
        let adjustments = json!({});
        let _ = fern::Dispatch::new()
            .level(log::LevelFilter::Info)
            .format(|out, message, _| {
                out.finish(format_args!(
                    "LOG {:?} {message}",
                    std::time::SystemTime::now()
                ))
            })
            .chain(std::io::stderr())
            .apply();
        let tint = |picture: &DynamicImage, analysis: &FaceAnalysis| {
            let mask = compose_face_mask(
                picture,
                analysis,
                None,
                &[FacePart::FaceSkin],
                FaceMethod::Landmarks,
            );
            let mut shown = picture.to_rgb8();
            for (x, y, p) in shown.enumerate_pixels_mut() {
                let m = mask.get_pixel(x, y)[0] as f32 / 255.0 * 0.55;
                let tint = [255.0, 40.0, 140.0];
                for c in 0..3 {
                    p[c] = (p[c] as f32 * (1.0 - m) + tint[c] * m) as u8;
                }
            }
            shown
        };
        let dng_cache = std::path::PathBuf::from(var("LOCALAPPDATA"))
            .join("io.github.CyberTimon.RapidRAW")
            .join("dng_converted");
        for path in var("RRAW_PICTURES").split(';') {
            // an HE NEF is read from the DNG the app converted it to, as the app reads it
            let dng = crate::decode::dng_converter::cache_key(std::path::Path::new(path))
                .map(|key| dng_cache.join(format!("{key}.dng")))
                .ok()
                .filter(|dng| dng.exists())
                .or_else(|| {
                    // or from one converted the same way beforehand, named after the raw
                    let stem = std::path::Path::new(path).file_stem()?;
                    let dir = std::env::var("RRAW_DNG").ok()?;
                    let dng = std::path::Path::new(&dir).join(stem).with_extension("dng");
                    dng.exists().then_some(dng)
                });
            let source = dng
                .as_ref()
                .map_or(path.to_string(), |dng| dng.to_string_lossy().to_string());
            let bytes = std::fs::read(&source).expect("raw");
            let full = crate::decode::image_loader::load_base_image_from_bytes(
                &bytes, &source, false, &settings, None,
            )
            .expect("decoded");
            let base = full;
            let loaded = LoadedImage {
                path: source.clone(),
                image: Arc::new(base),
                is_raw: true,
                as_shot: crate::decode::as_shot::as_shot(&source),
            };
            let name = std::path::Path::new(path)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            eprintln!("PROBE {name} read from {source}");
            let seen = crate::masks::generation::build_converted_source_image(
                &loaded.image,
                true,
                &adjustments,
            )
            .into_owned();
            let transform = MaskTransform::from_adjustments(&adjustments);
            let copies = DetectionCopies::new(
                context.clone(),
                &state,
                &loaded,
                &adjustments,
                render,
                false,
            );
            let pixels = (seen.width() * seen.height()) as f64 * 255.0;
            let out = std::env::var("RRAW_OUT").ok();
            // each pass on its own, to see them all
            let mut passes = vec![("seen", seen.clone())];
            for (copy_name, copy) in copies.make() {
                passes.push((
                    if copy_name == "Auto" {
                        "auto"
                    } else {
                        "lifted"
                    },
                    copy,
                ));
            }
            for (key, picture) in &passes {
                let encoder = models.esam3_image_encoder().expect("encoder");
                let features = run_esam3_image_encoder(picture, &encoder).expect("features");
                let analysis = faces_from_features(&models, &features, &name, picture, transform)
                    .expect("analysis");
                eprintln!(
                    "PROBE {name} {key}: {} face(s), face skin {:.2}%",
                    analysis.faces.len(),
                    face_skin_area(picture, &analysis) as f64 / pixels * 100.0
                );
                if let Some(dir) = &out {
                    picture
                        .to_rgb8()
                        .save(format!("{dir}/{name}_{key}.jpg"))
                        .unwrap();
                    tint(picture, &analysis)
                        .save(format!("{dir}/{name}_{key}_mask.jpg"))
                        .unwrap();
                }
            }
            // and the app's own choice
            let started = std::time::Instant::now();
            let found = compute_face_analysis(&Mutex::new(None), &models, &name, &seen, transform)
                .expect("analysis");
            let (kept, kept_name) =
                keep_the_fullest_faces(&models, &name, &seen, found, Some(&copies), transform);
            eprintln!(
                "PROBE {name} kept: {kept_name}, {} face(s), three passes took {:?}",
                kept.faces.len(),
                started.elapsed()
            );
        }
    }

    /// For each raw in RRAW_PICTURES, read at full size as the app reads it, and for each of the three
    /// versions: every box the detector puts round a face down to a score of 0.1, whether the app keeps
    /// it (score over ESAM3_FACE_SCORE_THRESHOLD, not a duplicate) and how well the face points fit in
    /// it. RRAW_OUT, if set, gets each version with its boxes drawn: green a face, orange a box kept but
    /// whose face points fit too poorly, blue a box under the detector's bar, grey a duplicate.
    #[test]
    #[ignore]
    fn why_a_face_is_missed() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
        let models = AiModels::new(std::path::PathBuf::from(var("RRAW_MODELS")), false);
        let context = gpu();
        let _ = crate::render::gpu::context::TEST_GPU.set(context.clone());
        let state = AppState::new();
        let settings = crate::app::settings::AppSettings::default();
        let render = crate::render::options::resolve_render_options(&settings, true);
        let adjustments = json!({});
        let dng_cache = std::path::PathBuf::from(var("LOCALAPPDATA"))
            .join("io.github.CyberTimon.RapidRAW")
            .join("dng_converted");
        let encoder = models.esam3_image_encoder().expect("encoder");
        let decoder = models.esam3_decoder().expect("decoder");
        let prompt = models.esam3_face_prompt().expect("prompt");
        let landmarks = models.face_landmarks().expect("landmarks");
        for path in var("RRAW_PICTURES").split(';') {
            let name = std::path::Path::new(path)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let dng = crate::decode::dng_converter::cache_key(std::path::Path::new(path))
                .map(|key| dng_cache.join(format!("{key}.dng")))
                .ok()
                .filter(|dng| dng.exists())
                .or_else(|| {
                    let dng = std::path::Path::new(&var("RRAW_DNG")).join(format!("{name}.dng"));
                    dng.exists().then_some(dng)
                })
                .expect("a DNG");
            let source = dng.to_string_lossy().to_string();
            let bytes = std::fs::read(&dng).expect("dng");
            let base = crate::decode::image_loader::load_base_image_from_bytes(
                &bytes, &source, false, &settings, None,
            )
            .expect("decoded");
            let loaded = LoadedImage {
                path: source.clone(),
                image: Arc::new(base),
                is_raw: true,
                as_shot: crate::decode::as_shot::as_shot(&source),
            };
            let seen = crate::masks::generation::build_converted_source_image(
                &loaded.image,
                true,
                &adjustments,
            )
            .into_owned();
            let copies = DetectionCopies::new(
                context.clone(),
                &state,
                &loaded,
                &adjustments,
                render,
                false,
            );
            let mut passes = vec![("seen", seen)];
            for (copy_name, copy) in copies.make() {
                passes.push((
                    if copy_name == "Auto" {
                        "auto"
                    } else {
                        "lifted"
                    },
                    copy,
                ));
            }
            for (key, picture) in &passes {
                let (w, h) = picture.dimensions();
                let features = run_esam3_image_encoder(picture, &encoder).expect("features");
                let boxes = run_esam3_boxes(&decoder, &features, &prompt, 0.1).expect("boxes");
                let kept: Vec<(f32, [f32; 4])> = boxes
                    .iter()
                    .copied()
                    .filter(|(score, _)| *score > ESAM3_FACE_SCORE_THRESHOLD)
                    .collect();
                let report = crate::masks::face::box_report(picture, &boxes, &landmarks);
                let kept_report = crate::masks::face::box_report(picture, &kept, &landmarks);
                let mut drawn = picture.to_rgb8();
                for (i, (b, score, dropped, presence)) in report.iter().enumerate() {
                    let in_app = kept_report
                        .iter()
                        .find(|(k, ..)| k == b)
                        .map(|(_, _, dropped, _)| dropped.is_none());
                    let verdict = match in_app {
                        None => "under the detector bar",
                        Some(false) => "dropped as a duplicate",
                        Some(true) if *presence >= 6.0 => "FACE",
                        Some(true) => "face points fit too poorly",
                    };
                    eprintln!(
                        "PROBE {name} {key} box {i}: score {score:.2}, {:.0}x{:.0} px at {:.0},{:.0} ({:.1}% of the width), points {presence:.1} -> {verdict}{}",
                        b[2] - b[0],
                        b[3] - b[1],
                        b[0],
                        b[1],
                        (b[2] - b[0]) / w as f32 * 100.0,
                        dropped.map_or(String::new(), |d| format!(" (overlaps box {d})"))
                    );
                    let colour = match verdict {
                        "FACE" => image::Rgb([40u8, 220, 90]),
                        "face points fit too poorly" => image::Rgb([255, 150, 0]),
                        "dropped as a duplicate" => image::Rgb([160, 160, 160]),
                        _ => image::Rgb([60, 140, 255]),
                    };
                    let t = (w.max(h) / 400).max(2) as i32;
                    for d in 0..t {
                        let x0 = b[0] as i32 - d;
                        let y0 = b[1] as i32 - d;
                        let rw = (b[2] - b[0]) as u32 + 2 * d as u32;
                        let rh = (b[3] - b[1]) as u32 + 2 * d as u32;
                        imageproc::drawing::draw_hollow_rect_mut(
                            &mut drawn,
                            imageproc::rect::Rect::at(x0, y0).of_size(rw.max(1), rh.max(1)),
                            colour,
                        );
                    }
                }
                if report.is_empty() {
                    eprintln!("PROBE {name} {key}: no box at all, even down to a score of 0.1");
                }
                // the face points fitted in the strongest box, drawn on a close crop of it
                if let (Some((b, ..)), Ok(dir)) = (report.first(), std::env::var("RRAW_OUT")) {
                    let (points, score) =
                        crate::masks::face::points_in_box(picture, *b, &landmarks);
                    let pad = (b[2] - b[0]).max(b[3] - b[1]) * 0.35;
                    let x0 = (b[0] - pad).max(0.0) as u32;
                    let y0 = (b[1] - pad).max(0.0) as u32;
                    let x1 = ((b[2] + pad) as u32).min(w);
                    let y1 = ((b[3] + pad) as u32).min(h);
                    let mut crop =
                        image::imageops::crop_imm(&picture.to_rgb8(), x0, y0, x1 - x0, y1 - y0)
                            .to_image();
                    let r = ((x1 - x0) / 250).max(3) as i32;
                    for p in &points {
                        imageproc::drawing::draw_filled_circle_mut(
                            &mut crop,
                            (p[0] as i32 - x0 as i32, p[1] as i32 - y0 as i32),
                            r,
                            image::Rgb([0u8, 255, 120]),
                        );
                    }
                    eprintln!(
                        "PROBE {name} {key}: points fitted in the strongest box score {score:.1}"
                    );
                    crop.save(format!("{dir}/{name}_{key}_points.jpg")).unwrap();
                }
                if let Ok(dir) = std::env::var("RRAW_OUT") {
                    let small = image::imageops::resize(
                        &drawn,
                        w * 1400 / w.max(h),
                        h * 1400 / w.max(h),
                        image::imageops::FilterType::Triangle,
                    );
                    small.save(format!("{dir}/{name}_{key}_boxes.jpg")).unwrap();
                }
            }
        }
    }
}
