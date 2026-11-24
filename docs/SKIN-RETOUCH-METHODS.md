# Skin retouch: three methods, one Strength slider, and the maps in view

Skin > the one-button retouch now works one of three ways, picked with the buttons at the top of its box. All three
share the **Strength** slider (the setting is still `skinRenewal`, so older edits open as they were saved), all three
find the face with the Skin tool's own face map, and all three are worked out once and kept as a change, so moving the
slider never redoes the work.

| Method | What it does | Strong at | Leaves alone |
|---|---|---|---|
| **Renewal** | PiXimperfect's "one-minute retouch": a median base, a new pore grain, the original's finest detail (docs/SKIN-RENEWAL.md) | badly marked skin | the face's shape |
| **Natural (dodge & burn)** | evens the light and shade between the size of a pore and the size of the face's shapes; takes the spots out | already decent skin to make flawless | every pore, the beard, freckles, the face's shape and light |
| **AI (learned)** | Alibaba's ABPN, trained on retouched portraits: fills the spots with skin, then evens the skin with a learned blend layer | a "retoucher's look" in one click | the eyes, lips, hair (it was trained to touch skin only) |

The Skin tool's sliders below (Spots, Redness, Pigment, Tone, Shine, Lines, Under eyes) still work on top of whichever
method is chosen.

## 1. Natural (dodge & burn)

`src-tauri/src/effects/dodge_burn.rs`

By hand, dodge & burn is an hour with two curves layers and a soft brush: each patch a little darker than the skin
around it is lightened, each lighter one darkened, until the light runs evenly. Nothing is blurred. The program does
the same by splitting the skin into bands of size and evening each band against the face's own best skin.

### How it works

1. **The faces.** The skin map is split into its blobs: each face is worked out at its own size, and none twice.
2. **A working copy** of each face's box, about 1000 px wide, in linear light. Everything below is measured in shares
   of the face's width `F`, so resolution and framing do not matter.
3. **Three channels:** the logarithm of the brightness (a correction is then a ratio, like a curve), and the
   logarithms of red / green and blue / green (the colour, kept apart from the brightness).
4. **Bands.** Blurs *over the skin only* (the average of what is skin near each place, so hair, eyes and background
   never leak in and the mask's edge makes no band) at widths from `s_lo` to 8 % of `F`, a √2 apart. Each band is the
   difference of two neighbouring blurs.
   - `s_lo` is above the **pore grain, measured in this photo** (the renewal's measurement: how far the finest detail
     goes on looking like itself, matched against made-up grains). A soft lens or an upscaled file has coarser grain,
     so the bands start higher, and the pores never go into any of them.
5. **How much each band is evened is not a fixed amount.** For each band: how rough it is around each place (`ρ`), and
   how rough the **quietest quarter of this face's skin** is (`τ`). Each place is brought down towards half of `τ`:
   `take = 1 − 0.5 τ / ρ`. A band rough everywhere (a hard light's shading, the face's relief) has a high `τ` and is
   hardly touched; a band quiet in most of the skin and rough in places (blotches, bump shadows, mottling) is evened
   exactly there. This is what makes it adapt to the person and the light: a young face and an old one, soft light and
   hard, get the same "even to this face's best skin".
6. **The coarsest bands** (from 3.5 % to 8 % of `F`) are taken out less and less: they are the face's shape.
7. **Left alone** (the "Left alone" map): hair, a beard and stubble (where the skin has far more fine detail than its
   own pores, so a beard's shadow is not lightened away), the strongest edges (nostrils, lid lines, the lip line),
   the deepest shadows, and, for brightening patches only, the highlights (the face's sheen).
8. **Spots** (B of B.S.T.): in the bands at spot size (0.3 % to 2 % of `F`), a place that stands more than 4 to 6.5
   times the band's usual spread out of it **and is round** (the two curvatures of the same sign and similar; a crease
   or an edge bends one way only). In brightness or in colour (a red spot need not be darker). Spread over the whole
   spot, then taken out whole: what the dodge & burn has left of the spot-sized bands at the strength asked for, so the
   two never take out more than is there. **When spots cover more than 4 to 10 % of the skin they are freckles** and
   are kept.
9. **Strength:** the dodge & burn follows the slider; the spots are out by 40 %; the colour gets 0.6 of it.

Spot detection follows the Skin tool's **Spots** and **All** detection sliders (50 is the usual; higher finds fainter
spots).

## 2. AI (learned)

`src-tauri/src/effects/ai_retouch.rs`, `masks::face::compose_ai_retouch`, `ai::commands::generate_ai_skin_retouch`,
`scripts/export_abpn_retouch_onnx.py`

ABPN (Lei et al., CVPR 2022; ModelScope `damo/cv_unet_skin_retouching_torch`, Apache-2.0), run the way its pipeline
runs it on each face, cut out 1.5 times the face box along the face's own angle:

1. the **blemish detector** (`abpn_blemish_detection.onnx`, already used by Skin > Spots > AI) finds the spots;
2. the **spot filler** (`abpn_retouch_inpaint.onnx`, a gated-convolution U-Net) fills them with skin, 512 px at a time
   with 32 px of overlap, at the face's own resolution (up to 2048 px a side);
3. the **blend network** (`abpn_retouch_blend.onnx`, a U-Net) looks at the filled face at 512 × 512 and makes a blend
   layer `m` for each channel; the skin `x` becomes `(1 − 2m) x² + 2m x`, a soft light, faded out at the cut-out's
   border as the pipeline fades it.

The models run **once**, when the method is chosen, and leave two maps with the edit (`skinAiBlendMap`,
`skinAiSpotsMap`, in the picture's own space). The render only lays them on: the spots are out by 40 % of the slider,
and the slider scales the blend layer around 0.5, which is exactly the pipeline's own `degree` (100 % = ABPN's result).
A raw picture is changed through the same display curve the face models are given (gamma 2.38, contrast 1.28) and
brought back to linear; a pixel that curve clips is left alone. The face's skin map, widened a little, keeps the change
on the face.

The pipeline's last step, whitening, is left out on purpose.

### Models

Three files in the models folder (`%APPDATA%\io.github.CyberTimon.RapidRAW\models`):
`abpn_blemish_detection.onnx` (`scripts/export_blemish_onnx.py`), `abpn_retouch_blend.onnx` and
`abpn_retouch_inpaint.onnx` (`scripts/export_abpn_retouch_onnx.py`, from `pytorch_model.pt` and `joint_20210926.pth`
of the ModelScope repository; weights are read with `weights_only=True`). The export checks each ONNX file against
PyTorch (largest difference 6e-7 and 7e-6). Without them, choosing the method says which file is missing and goes back
to Renewal.

## 3. Show a map

Skin > **Show a map** (above the detection sliders) shows, in the preview only, what a detection or a method finds:
the picture in grey and the map in magenta, the stronger the colour the stronger the detection or the change. It is
never saved with the edit and never exported (it rides along with the preview request only), and it is turned off when
another photo is opened or the Skin tool is switched off.

| Map | Drawn by | What it shows |
|---|---|---|
| Skin | skin.wgsl | the face skin the tool works in |
| Spots | skin.wgsl | the blemishes to heal (from the shape, or from the AI detector when its switch is on) |
| Redness | skin.wgsl | the patches redder than this face |
| Pigment | skin.wgsl | brown / dark patches that are pigment, not shadow |
| Lines | skin.wgsl | the lines (curvature, or the wrinkle model) |
| Under eyes | skin.wgsl | the under-eye band |
| Shine | skin.wgsl | brightness above the face around it |
| Renewal | renewal.rs | where the renewal goes and how far |
| Dodge & burn | dodge_burn.rs | how much each place changes at full strength |
| D&B spots | dodge_burn.rs | the spots it takes out whole |
| Left alone | dodge_burn.rs | hair, stubble, strong edges, deep shadows it does not touch |
| AI retouch | ai_retouch.rs | how much the learned retouch changes each place |

The Skin tool's maps are drawn at a full slider, whatever the slider says, so the **detection sliders can be set
first, map in view**, before anything is applied. The method maps follow the method chosen.

Limit: while AI Upscale or Superfocus is on, the method maps (Renewal, Dodge & burn, AI retouch) are not shown, so that
showing a map does not start an upscale again; the Skin tool's maps are.

## 4. What to test, and how

Build: the draft pull request on the fork compiles it (no release, nothing published). The skin unit tests now run in
CI (Lint > "cargo test (skin retouch)").

1. **Method switch.** Skin on, a portrait. The three buttons; Strength at 0 changes nothing; at 70 each method gives
   its own look. Texture shows for Renewal only.
2. **Dodge & burn, the basics.** Strength 0 → 100 on a face with blotches or redness: blotches and uneven colour go,
   pores stay (zoom to 100 %), the face keeps its volume (nose, cheekbones), no halo at the nostrils, eyes or jaw.
3. **Dodge & burn, the person.** A man with stubble or a beard: the beard is not lightened (map "Left alone" shows it
   white). A face with freckles: they are kept (map "D&B spots" empty, the log says "kept as freckles"). A face with a
   few red spots: gone by Strength 40.
4. **Dodge & burn, the light.** A hard side light: the shading stays; only the blotches go.
5. **Spot detection.** Map "D&B spots" in view, move Skin > Detection > Spots: more or fewer spots appear.
6. **Two faces** in one photo: both retouched, neither twice (no doubled effect where their boxes overlap).
7. **AI (learned).** Choose AI: "Retouching the skin with AI…" for a few seconds, then the slider works at once.
   Spots filled, skin evened. Without the models: a message naming the missing file, back to Renewal.
8. **Raw and JPEG**, each method: no colour shift, no clipped highlights on the forehead.
9. **Maps.** Each map button; Off returns the picture; changing photo turns it off; export a JPEG while a map is shown:
   the export has the retouch, not the map.
10. **Export** with each method at 70: same as the preview.

The log (`Skin dodge & burn worked out in …`) gives, for each face, the pore grain measured, each band's width, its
quiet roughness and the share taken, and how much of the skin the spots cover.
