# Suggestions to work on later

Ideas and references collected for RapiderRaw. Nothing here is started; this is the list to pick from.
(Issue links are upstream RapidRAW feature requests; titles and states as of 15 Nov 2026.)

| Issue | Request | Where it lands | Touches the caches? |
|---|---|---|---|
| [#2083](https://github.com/CyberTimon/RapidRAW/issues/2083) | New demosaic algorithm for RapidRAW-Dnglab | the decoder, below everything | **yes, the deepest of all** |
| [#2059](https://github.com/CyberTimon/RapidRAW/issues/2059) | Visualise affected areas for colour and tonal adjustments | an overlay on the display | **no, and it must not** |
| [#1992](https://github.com/CyberTimon/RapidRAW/issues/1992) | Method for slider fine control | the front end alone | no |
| [#1983](https://github.com/CyberTimon/RapidRAW/issues/1983) | Add focus peaking | an overlay on the display | **no, and it must not** |
| [#1962](https://github.com/CyberTimon/RapidRAW/issues/1962) | Integration with an AI sharpening tool | a tiled AI job | yes, like denoise |
| [#1898](https://github.com/CyberTimon/RapidRAW/issues/1898) | Update AI models and introduce a small AI hub | the model registry | **yes, and it is the subtle one** |
| [#1539](https://github.com/CyberTimon/RapidRAW/issues/1539) | More noise reduction and sharpening options | the main processor, plus mask algebra | yes, the ordinary way |
| [#1129](https://github.com/CyberTimon/RapidRAW/issues/1129) | Film emulation specific features | the main processor | yes, the ordinary way |
| — | [Dust spot detector](https://huggingface.co/fujiba/klarsky-dust-spot-detector-experimental) | a map plus a staged effect | yes, like skin |

---

## Read this first: how caching and previews work here

This is the part of the codebase that punishes a careless addition, and it punishes it quietly - the
feature looks implemented, the slider moves, and the picture does not change. It has already happened
once on this fork: the skin effect was written, compiled and wired in, and did nothing, because its
settings were not in `calculate_effects_hash`. The rendered effects stage was simply reused. Fixed in
`db1f66d0`; the lesson belongs here so the next one costs nothing.

### The hash chain

All of it lives in `src-tauri/src/render/cache_utils.rs`.

| hash | what it covers | what it governs |
|---|---|---|
| `calculate_patch_hash` | AI patches (clone/heal), AI denoise, negative conversion; plus orientation and geometry when any patch relies on a generated mask | the patched source, thumbnails, presets |
| `calculate_geometry_hash` | the patch hash, plus every key in `GEOMETRY_KEYS` | `ai_cache_key`, so every AI mask and cached embedding is per path **and** per geometry; `get_cached_full_warped_image` |
| `calculate_effects_hash` | per staged effect: its enabled flag, then its own keys - `lensBlur`, `relight`, `fog`, `skin` | `Source::key()` in `render/source.rs`, the key of the whole staged-effects result |
| `calculate_transform_hash` | orientation, rotation, flips, lens blur enabled, patches, denoise, negative conversion, skin | the export size estimate, `render/commands.rs`, the preview worker |
| `calculate_full_job_hash` | the path and the entire adjustments JSON | the preview worker's job identity |

### Two places to add a feature, with different duties

**The main processor** - `render/gpu/processor.rs`, the pass that applies the global adjustments. Glow,
halation, flare, clarity, sharpening, noise reduction and all the colour work run here. Anything added
here is covered by the full adjustments JSON through the preview worker, so a new uniform field and a
new key need **no hash change at all**. This is the cheap side of the fence.

**The staged effects** - `render/source.rs`, inside `if transform.effects`, each a `prepare`/`encode`
pair under `src-tauri/src/effects/`. Relight, fog, lens blur and skin live here, as compute passes
between the main processor and the frame. These are cached by `Source::key()`, and `Source::key()` knows
only what `calculate_effects_hash` tells it. **Forget your keys and your effect will never re-render.**

### The caches themselves

Single-slot `(u64, Arc<...>)`, cleared by `clear_preview_stage_caches`: `patched_cache`,
`cleanup_source_cache`, `working_cache`.

Per preview target, cleared by `clear_preview_slot`: `gpu_image_cache_for`, `cached_preview_for`.

Whole-image, cleared by `clear_image_caches`: `decoded_image_cache` (budgeted against system RAM),
`full_warped_cache` (one slot, path plus geometry).

Session, cleared by `clear_session_caches`: `patch_cache`, `mask_cache`, `uncropped_source_cache`.

AI, in `ai_state`, each a single slot matched on `ai_cache_key`: `face_analysis`, the EfficientSAM3
features, the SAM embeddings. Effect maps live in `effects/maps.rs`, capacity four, keyed by
`effect_map_key(data_url, adjustments)`.

**A single-slot cache that stores a failure keeps returning it.** `cached_face_analysis` stored an empty
analysis exactly as it stored a real one, which turned a transient model problem into one that looked
permanent and could not be retried out of. Whatever you add: never cache an empty result, or mark it so
a retry recomputes.

### The checklist for anything new

1. Which stage does it belong to - the main processor, or a staged effect?
2. A staged effect: add every key to `calculate_effects_hash`, **and** to `calculate_transform_hash` if
   it changes the output size or the export estimate.
3. Does it count as an edit? `metadata/sidecar.rs`, `is_image_edited`.
4. Does it produce a map or a model result worth keeping? Key it on content, not identity - the renderer
   hands over a new picture every frame. Bound its capacity, and never let it hold a failure.
5. Is it an overlay rather than an edit? Then it must touch **none** of the hashes.
6. Front end: the key in `Effect`, the field on `Adjustments`, the default in `defaultAdjustments.json`,
   the line in the loader, the copy/paste group and the effects key list.

---

## #2083 - A better demosaic algorithm

**What is asked.** Strong colour artefacts with the present PPG demosaic. There is a stale dnglab PR
with DDFAPD, and a working if unoptimised implementation at `christian-photo/dnglab`. RapidRAW keeps its
own dnglab fork, so it can take it directly rather than wait upstream.

**Where it lands.** Below everything: `src-tauri/src/decode/`, in the rawler fork. Not the shaders, not
the adjustments.

**How.** Port `ddfapd.rs` into the fork's `rawler/src/imgop/sensor/bayer/`, then expose the choice. Two
other open requests ([#1530](https://github.com/CyberTimon/RapidRAW/issues/1530),
[#684](https://github.com/CyberTimon/RapidRAW/issues/684)) ask for a selectable demosaic, so build it as
a setting from the start rather than swapping the default and being asked for the old one back.

**The cache note, and it is the serious one.** The demosaic result *is* the decoded image, at the bottom
of every cache above it. A setting that changes it must invalidate `decoded_image_cache`, and through it
`full_warped_cache`, every preview stage slot, and every AI cache keyed by `ai_cache_key` - the
embeddings and the face analysis were computed from the old pixels. Two honest options: make it a
restart-level application setting and call `clear_image_caches` plus `clear_session_caches` when it
changes, or put the demosaic choice into `calculate_patch_hash`, the root of the chain, which carries it
into the geometry hash and the AI cache key for free. The second is more work and the right answer if it
is ever to be per-image.

---

## #2059 - Visualise which areas an adjustment affects

**What is asked.** Show what a Colour Mixer band, a tonal range or a colour-grading zone actually
targets, and how strongly: affected colours kept while the rest desaturates, or a greyscale map of
influence. Explicitly, it should work **even at a setting of zero** - showing potential influence rather
than the difference currently being made.

**Where it lands.** The display path, `render/gpu/display.rs` and `shaders/display.wgsl`, plus a toggle
in the front end. Not the processor.

**How.** The influence of each of these adjustments is already computed inside the main shader as a
weight before being applied - the hue distance for a mixer band, the tonal falloff for highlights or
shadows. The work is to write that same weight into a single-channel texture and blend it over the
displayed frame, not to invent new maths. A mode enum picks which weight to show.

**The cache note.** This must stay out of the adjustment hashes entirely. It changes what is drawn, not
what is rendered. If the flag reaches `calculate_full_job_hash` or `calculate_effects_hash`, toggling it
re-renders the whole preview, and the "works at zero" requirement turns into a re-render on every hover.
Keep it in front-end state and the display uniform, and it costs one extra pass over a frame already on
the card. It must also never reach `is_image_edited`, or looking at a photograph would mark it edited.

---

## #1992 - Fine control for sliders

**What is asked.** On a touchpad, small adjustments are painful; shift plus wheel still moves too fast.
Asked for a configurable scroll rate, or a modifier for small steps.

**Where it lands.** `src/components/ui/Slider.tsx` and the shared `ValueDrag` handler. Pure front end.

**How.** The drag already divides by a fixed `S(600)`; make that divisor respond to a modifier - ctrl or
alt for a tenth of the rate - and add a step-size preference in app settings. Do it in the handler the
panels share, so every slider gains it at once.

**The cache note.** None. The value that comes out is an ordinary value; nothing about how it was
reached is stored. The one thing to preserve is that a drag still ends in a single final commit rather
than one per pixel moved, so the preview is not re-rendered at every intermediate value.

---

## #1983 - Focus peaking

**What is asked.** A live overlay marking what is in sharp focus, for judging sharpness and depth of
field without zooming. Toggleable, drawn on top, never touching the exported file.

**Where it lands.** The display path, exactly as #2059. **Build these two together** - they are the same
mechanism with a different weight, and doing them separately means writing the overlay plumbing twice.

**How.** A high-pass of the luminance of the frame already on the card, thresholded, drawn in a chosen
colour. One compute pass, or a branch in the display shader. Sensitivity and colour as settings.

**The cache note.** As #2059, and worth repeating because it is the thing that will go wrong: **no hash,
no sidecar, no adjustment key.** If it is ever tempted to read the *unprocessed* image instead - which
is arguably more useful, since focus is a property of the capture rather than of the edit - take it from
the decoded cache rather than adding a stage, so no new cache slot appears.

---

## #1962 - AI sharpening

**What is asked.** An open AI sharpening tool. The requester knows only GFPGAN, and notes it does faces
only.

**Where it lands.** `src-tauri/src/ai/`, as a tiled job beside denoise and upscale: `ai/jobs.rs`,
`ai/tiling.rs`, a model entry in `ai/models.rs`.

**How.** Denoise is the template in every respect - `hash_ai_denoise`, the tiling, the progress events,
the per-image application. On the model: GFPGAN is face-only and is really face *restoration*, a
different feature that overlaps awkwardly with the skin work on `skin-by-fault`. A general
super-resolution model run at 1x, or the existing `2xBHI_small_span_rapidraw` applied and downscaled, is
closer to what is being asked and needs no new model at all. Worth measuring before downloading
anything.

**The cache note.** Follow `hash_ai_denoise` exactly: it is already in `calculate_patch_hash` and
`calculate_transform_hash`, which is what makes a denoise setting invalidate everything downstream. A
sharpening job applied to the source needs the same; one applied at the end needs much less. Decide
which it is before writing any of it, because the answer decides where it is hashed.

---

## #1898 - Update the AI models, and an AI hub

**What is asked.** Some models are ageing, SAM especially; SAM 3, Depth Anything 3 and EmbeddingGemma 2
are suggested. Plus a hub: browse, download and choose which model serves each feature, instead of the
hardcoded table.

**Where it lands.** `ai/models.rs` - the `ModelAsset` table, `download_and_verify_model`, the
`AiMaskModel` sets - and a settings page.

**How.** The table is already the right shape: name, filename, URL, sha256. A hub makes it data rather
than constants, with the per-feature choice in settings. The verification path should stay exactly as it
is: it is what guarantees a corrupt file is replaced rather than used.

**On the models themselves, we have measurements.** Depth Anything 3 Small was tried against V2 Small
for lens blur and was *worse*: on a close portrait it returns the subject as a near-flat white cut-out
against a blotchy background - a segmentation rather than a depth field - and turns hair into a hard
silhouette. V2 Large was tried too and measured no better than V2 Small: the face modelled over 21.6,
23.8 and 25.1 % of the face-to-background gap for Small, Base and Large, with nose-and-socket structure
worth 2.83, 2.87 and 2.86 %, at 3.5 times the wait and fourteen times the download. "Newer" is not the
argument; measure each swap on real photographs first.

**The cache note, and this is the subtle one.** Every AI result cached anywhere was produced by a
particular model. The face analysis, the EfficientSAM3 features, the SAM embeddings, the depth and
normal maps are all keyed by `ai_cache_key`, which is path plus geometry and knows nothing about which
model made them. Change a model through a hub and stale results are served as if nothing happened.
Whatever identifies the chosen models has to enter `ai_cache_key`, or every AI cache has to be dropped
when the choice changes. The second is simpler and nearly free, since these are single slots.

---

## #1539 - More noise reduction and sharpening options

**What is asked.** Three things, and they are not one feature:

1. Radius, detail and a black-and-white masking visualiser for sharpening, as Lightroom has them.
2. Detail and contrast sliders for noise reduction.
3. Subtracting a subject from a mask, so the mask does not apply there.

**Where each lands.** 1 and 2 are the main processor: new fields on the uniform struct in
`render/uniforms.rs`, new maths in `shaders/shader.wgsl`, new keys in the front end. The sharpening
masking visualiser is the odd one out - it is an overlay, and belongs with #2059 and #1983.

3 is mask algebra, in `src-tauri/src/masks/`. The mask system is bitmap-based, so subtraction is
arithmetic on the bitmap; the work is in the UI and the mask model, which needs a per-component subtract
flag.

**The cache note.** 1 and 2 need no hash change - the main processor is covered by the full adjustments
JSON. 3 touches `mask_cache`, cleared by `clear_session_caches`: a subtract flag changes the composed
mask, so it must be part of that cache's key. The visualiser stays out of every hash, as above.

---

## #1129 - Film emulation

**What is asked.** Halation that behaves like halation: edge-aware, driven by contrast and luminance
difference, biased towards highlights and strong light sources, with control over spread, intensity and
colour - rather than the present uniform treatment. Glow control in the same spirit.

**Where it lands.** The main processor. Glow, halation and flare already run there through
`effects/light.rs` and `shaders/light_effects.wgsl`, dispatched from `render/gpu/processor.rs` - **not**
through the staged effects in `source.rs`. That distinction matters: it is the cheap side of the fence.

**How.** Halation is light scattering back off the film base, so it comes from bright areas and spreads
into what is beside them. Edge-aware means weighting the source of the bloom by local contrast as well
as by brightness: a bright sky with nothing dark against it should halate less than the same brightness
against a dark branch. The existing bloom pass already produces the spread; the change is to what feeds
it. Spread, intensity and colour become three uniform fields instead of one amount.

**The cache note.** The ordinary way: new keys on the existing adjustments, covered by the full
adjustments JSON, no hash edit needed. If the bloom is ever moved into a staged effect for performance,
that changes - and then `calculate_effects_hash` must list every one of its keys.

---

## Dust spot detector

**The reference.** [`fujiba/klarsky-dust-spot-detector-experimental`](https://huggingface.co/fujiba/klarsky-dust-spot-detector-experimental),
AGPL-3.0 on its model page. Candidate for automatic sensor dust removal.

**Why it fits here.** It is the same shape of problem as the Spots map in
[`pixtune-skin-reference.md`](pixtune-skin-reference.md): find small blobs that do not belong, then take
them out without taking the surrounding detail with them. The difference is that dust sits on the sensor
rather than on a face, so it needs no face, no landmarks and no regions - strictly simpler than the skin
work already on `skin-by-fault`.

**How.** The model produces a map; the removal is the existing healing. So: a map produced like the
depth map (`generate_full_image_depth_map` through `EffectMapKind` is the template), then either feed
the detections to `retouch/clone_heal.rs` as patches, or take them out with a frequency-split correction
in a staged effect, the way the skin spots are.

**The cache note.** A map through `effects/maps.rs` is keyed by `effect_map_key(data_url, adjustments)`
and bounded at four entries - that is the pattern to follow. If the removal becomes a staged effect, its
keys go into `calculate_effects_hash`; see the skin miss above. If instead it emits clone/heal patches,
it rides on `hash_ai_patches` into `calculate_patch_hash`, which invalidates far more and is right if
the dust is removed from the source rather than at the end.

Dust is also per *sensor*, not per picture: the same spots land in the same places across a whole shoot.
Detecting once and offering the result across a selection would be worth more than per-image detection,
and that is a library-level feature rather than an editing one.
