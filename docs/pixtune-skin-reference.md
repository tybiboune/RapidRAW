# PixTune's skin work, in full

What PixTune does to skin, every part of it, and exactly how. This is the reference for carrying it into
RapidRAW: the branch `skin-by-fault` has the first slice of it, and this document is what the rest is to
be built from.

Source: `D:\Gemini\PixTune\engine\Skin.cs` (the atlas), `SkinDefects.cs` (the maps and the corrections),
`Face.cs` (the Portrait tool and the brush tool), `AppTools.cs` (the sliders).

Everything is measured against the face in front of it rather than against a fixed idea of skin. A
blemish on a pale face in shade and a blemish on a dark face in sun have nothing in common in absolute
terms - only in how far each stands out from what surrounds it. That principle is why there are so many
ratios below and so few constants in units of brightness.

---

## 1. The atlas: where the skin is and what colour it is

`SkinAtlas.Build(src, faces, facesKey)` -> `Atlas`, holding six region maps (`Skin`, `Forehead`,
`Cheeks`, `UnderEye`, `Nasolabial`, `NoseChin`), the face's own colour (`CbMid`, `CrMid`, `CbSpread`,
`CrSpread`), `FaceSize` in pixels, and `FromMesh`.

### 1.1 Anchors, from whichever source works

Per face box, in this order, falling through on failure:

| source | what it gives |
|---|---|
| `FaceMesh.Landmarks` - 468 points | eye centres from the eye rings, mouth corners, nose tip, chin, and a convex hull of all 468 points |
| `FromFive` - the detector's 5 keypoints | eyes, nose, mouth corners; brow line at eye centre minus 0.28 x eye span; chin at mouth + 1.6 x (mouth - nose); a 24-point oval over the box |
| `FromBox` - the box alone | eyes at 0.31 and 0.69 of the width, 0.42 of the height; mouth at 0.75; nose at 0.60; chin at 1.02; brow at 0.33; a 24-point oval |

Face size is `eyeSpan x 2.6` from the mesh, `max(eyeSpan x 2.6, boxWidth)` from the five points, and the
box width otherwise.

The hull's winding direction is **measured, not assumed** (`Winding`), because the mesh hull and the
substitute oval are not built the same way, and `InHull` with the sign the wrong way round puts every
pixel outside the face and returns an empty atlas. `InHull` gives the distance in pixels to the nearest
edge, negative outside.

### 1.2 Regions, before any colour is looked at

`face = SmoothStep(-Size x 0.02, Size x 0.06, inside)` - a little way inside the outline, so the hairline
and the jaw edge are left out. Then, per face, taking the maximum across faces:

- **cut** (never skin, never smoothed or healed): both eyes as blobs `EyeR x 1.5` by `EyeR x 1.1`; both
  brows at `BrowY`, `EyeR x 1.6` by `EyeR x 0.75`; the mouth, `MouthHalf x 1.25` by `MouthHalf x 0.8`;
  the nostrils at `NoseY + Size x 0.02`, `Size x 0.075` by `Size x 0.045`.
- **Forehead** = `face x (1 - SmoothStep(BrowY - Size x 0.06, BrowY, y))`
- **Cheeks** = `face x side x band`, where `side = SmoothStep(Size x 0.08, Size x 0.17, |x - NoseX|)` and
  `band = SmoothStep(eyeMid + EyeR x 0.6, eyeMid + EyeR x 1.4, y) x (1 - SmoothStep(MouthY - MouthHalf x 0.6, MouthY + MouthHalf x 0.3, y))`
- **UnderEye** = `face x` a blob under each eye at `EyeY + EyeR x 1.35`, radii `EyeR x 1.5` by `EyeR x 0.85`
- **Nasolabial** = `face x` a soft band from `(NoseX +- Size x 0.09, NoseY + Size x 0.03)` to
  `(MouthX +- MouthHalf, MouthY + MouthHalf x 0.4)`, half-width `Size x 0.05`
- **NoseChin** = `face x` the nose, a blob at `NoseY - Size x 0.06` of `Size x 0.09` by `Size x 0.13`,
  and the chin, a blob at the midpoint of mouth and chin, `MouthHalf x 0.9` by `|ChinY - MouthY| x 0.45`

Every region is then multiplied by `1 - cut`. `Skin` starts as `face x (1 - cut)`.

### 1.3 The face's own colour

Sampled from **cheeks and forehead only** (`Cheeks + Forehead >= 0.6`), skipping anything below 0.08 or
above 0.97 luminance - deep shadow and blown highlights say nothing about colour. Sampling step is
`max(1, sqrt(n / 20000))`.

`CbMid`, `CrMid` are **medians**; `CbSpread`, `CrSpread` are the median absolute deviation x 1.4826, floored
at 1.5. A median rather than an average because a face carries exactly the red patches the maps are
looking for, and they would drag an average towards themselves - the measure would move with what it is
measuring.

With fewer than 50 samples it falls back to fixed values - Cb 110, Cr 152, spreads 16 and 13 - and clears
`FromMesh`.

### 1.4 Narrowing the skin by colour

```
icb = 1 / max(2, CbSpread x 2.6)
icr = 1 / max(2, CrSpread x 3.4)
Skin *= exp(-(dcb^2 + dcr^2) x 0.5) x SmoothStep(0.05, 0.14, L) x (1 - SmoothStep(0.96, 1, L))
```

Deliberately **more generous across Cr than Cb** (3.4 against 2.6): redness is a move along Cr and a
blemish is still skin, so a tight Cr would cut out the very things the defect maps are meant to find.

Finally every region is blurred by `max(1, FaceSize x 0.006)` so no edge is stepped.

---

## 2. The six defect maps

`SkinDefects.Build(src, faces, facesKey)` -> `Maps`. `fs = max(24, FaceSize)`. Everything below is
weighted by `Skin` and skipped where `Skin <= 0.01`.

First, the local scale that everything else is measured in:

```
fine   = max(1, fs x 0.010)        about the width of a blemish
coarse = max(2, fs x 0.055)        the shading across a face, which is not a blemish
band   = Gauss(L, fine) - Gauss(L, coarse)
spread = Gauss(|band|, max(3, fs x 0.09))      how much this skin varies locally
```

`sp = max(0.004, spread)` is the yardstick in three of the maps below.

| map | what it finds | how |
|---|---|---|
| **Spots** | small blobs darker or lighter than the skin around them | `t = band / (sp x 1.6)`; `v = -t` if dark, `t x 0.55` if light; `Skin x C01(SmoothStep(0.55, 1.8, v))`. **Dark counts nearly twice what light does** - a light blob the same size is usually a highlight, not something anyone wants removed |
| **Redness** | redder than this face's own skin, over an area | `over = (Gauss(Cr, max(2, fs x 0.03)) - CrMid) / max(2, CrSpread)`; `Skin x C01(SmoothStep(0.3, 1.5, over))`. Full at a spread and a half above the middle |
| **Lines** | thin dark valleys: wrinkles, creases, the folds beside the nose | two widths, `max(1, fs x 0.006)` and `max(1.5, fs x 0.013)`; at each, `dark = Gauss(L, sc x 2.2) - Gauss(L, sc x 0.6)`, positive where the middle is darker; `v = C01(SmoothStep(0.45, 1.6, dark / (sp x 1.5)))`; weighted by where a face folds anyway, `where = 0.45 + 0.55 x C01(Nasolabial + UnderEye x 0.8 + Forehead x 0.6)`; the maximum of the two widths |
| **Shine** | light bounced off the skin rather than coming out of it | `sat = (max - min) / max` of RGB; `Skin x SmoothStep(0.62, 0.92, L) x (1 - SmoothStep(0.12, 0.34, sat))`. **Bright and colourless together** - that is what separates it from skin that is simply pale |
| **UnderEye** | darker than the cheek below it, inside the under-eye band | `cheekRef = Gauss(L, max(3, fs x 0.12))`; `deficit = (cheekRef - L) / max(0.02, cheekRef)`; `UnderEyeRegion x C01(SmoothStep(0.015, 0.16, deficit))`. Note this is weighted by the **region**, not by `Skin` |
| **Texture** | the fine detail that is none of the above and has to survive | `lSmall = Gauss(L, max(0.8, fs x 0.004))`; `texture = Gauss(\|L - lSmall\|, max(2, fs x 0.02))`; `claimed = max(Spots, Lines, Shine)`; `Skin x C01(texture / (sp x 1.4)) x (1 - claimed)` |

Texture is the whole reason a result can avoid looking like plastic. The other five say what to take
away; this one says what to put back, and it explicitly stands down wherever another map has claimed
that pixel.

### 2.1 The four working signals

Kept beside the maps because they depend on the photograph and not on any setting, so moving a slider is
one pass over the pixels and nothing is recomputed:

- **Blob** = `band` - the blemish-sized ups and downs of brightness, the spot signal itself
- **Smoothed** = `Gauss(L, max(1.2, fs x 0.010))` - brightness with the thin valleys taken out
- **Around** = `cheekRef` - brightness over a wide area: what shine is above and a shadow below
- **Fine** = `L - lSmall` - the finest detail, which is what goes back at the end

---

## 3. Applying the corrections

`SkinDefects.Apply(src, M, spots, redness, rednessWay, lines, shine, underEye, keepTexture)`.

Per pixel where `Skin > 0.01`, converted to **studio-swing YCbCr** (`Y = 16 + 65.481R + 128.553G +
24.966B`, Cb and Cr about 128), corrected, converted back, then faded by `Skin`:

```
D = lerp(original, corrected, Skin[i])
```

so the edge of a correction is never a visible border.

**Order matters and is deliberate** - each setting works only through its own map, so turning one up
cannot quietly do another's job:

1. **Spots** `yy -= 219 x Blob x Spots x spots`
   Takes the blob away and the skin around it closes over; anything finer than the blob is not in `Blob`
   at all, so the pores stay exactly where they were.
2. **Lines** `yy += 219 x (Smoothed - lNow) x Lines x lines`, where `lNow` is read from the **original**
   brightness, not the running value. Fills the valley part of the way back up to the skin beside it.
3. **Shine** `over = lNow - Around`; if positive, `yy -= 219 x over x Shine x shine x 0.85`. `lNow` here
   is the running value. Moved back towards what the face is doing around it, never past it.
4. **Under eyes** `under = Around - lNow`; if positive, `yy += 219 x under x UnderEye x underEye x 0.8`.
5. **Redness** - colour only, brightness untouched. Correct a flush by darkening it and the face goes
   grey, which is the usual giveaway.
   ```
   k = clamp(Redness / Skin, 0, 1) x redness      the map's skin weight divided back out
   cr -= (cr - toCr) x k
   cb -= (cb - toCb) x k x cbShare
   ```
   The division is not cosmetic: the map carries `Skin` and the whole correction is faded by `Skin`
   again at the end, and counting it twice is what kept a full slider from ever arriving.

   | `rednessWay` | name | target | cbShare | extra |
   |---|---|---|---|---|
   | 0 | This skin | `CrMid` | 0.35 | the patch comes out the colour of the cheek beside it |
   | 1 | Half way | `CrMid` | 0.35 | `k x 0.5` - the flush calmed rather than erased |
   | 2 | Warm | `CrMid + CrSpread` | 0.15 | the skin colour with some warmth kept |
   | 3 | Even | `CrMid - CrSpread x 0.6` | 0.70 | the red taken out and nothing put back, the flattest |

6. **Keep texture** `yy += 219 x Fine x Texture x keepTexture` - the skin's own grain back on top of
   whatever was done to it.

---

## 4. The whole-skin steps, which run after

`Face.Portrait` runs the per-fault work **first**, then these. Taking a blemish out and then smoothing is
not the same as smoothing a blemish away.

These four use a different, older skin mask: `Face.SkinMask`, which is `SkinScore` - a **fixed**
Cb 110 / Cr 152, spreads 16 / 13 - times an ellipse around each face (`FaceRegion`, grow 1.5, soft 0.6),
blurred by `max(1, 2.5 x Scale)`. It is not the fitted atlas. *(Carrying this over is an opportunity to
drop the fixed thresholds and use the atlas throughout.)*

| step | parameter | what it does |
|---|---|---|
| **Skin smoothing** | `smoothing` | `SmoothSkin` = a guided filter per channel, radius `max(2, fs x 0.035)`, eps 0.0028; blended at `min(1, skin x 1.4) x smoothing x 0.92`. This is the old single slider that blurs every bit of skin alike - the per-fault maps exist to make it unnecessary |
| **Skin tone** | `skintone` 0-5 | adds a per-channel offset plus a lift, times the skin mask. 0 none; 1 pale `(+0.02, +0.01, +0.03)` lift `+0.06`; 2 light `(+0.04, +0.015, -0.01)` lift `+0.03`; 3 medium `(+0.05, 0, -0.04)` lift 0; 4 deep `(0, -0.03, -0.06)` lift `-0.05`; 5 rosy `(+0.05, -0.01, +0.01)` lift `+0.01` |
| **Face spotlight** | `spotlight` | multiplies RGB by `1 + spot x 0.32 x m - spot x 0.28 x (1 - m)`, where `m` is `FaceRegion(1.25, 1.2)` or, with no face, a centred ellipse at 0.42 of the height, radius `MinDim x 0.45`. Lifts the face and drops everything else |
| **Eye clarity** | `eyes` | `EyeClarity`: `d = (L - Gauss(L, max(1, fs x 0.012))) x 2.4 + (L - Gauss(L, max(0.7, fs x 0.004))) x 2.0 + 0.04`, added to all three channels, weighted by gaussian blobs at 0.31 and 0.69 of the face width, 0.40 of its height, radius `fw x 0.13` and 0.7 as tall. Local contrast, a touch of sharpening, and a small lift |

---

## 5. The brush tool

`Face.FaceEnhance` - the same operations, painted by hand instead of masked automatically. Face size is
assumed, `MinDim x 0.3`:

- stroke channel **`skin`** -> `SmoothSkin` blended at `C01(m / 10) x 0.9`
- stroke channel **`spot`** -> RGB multiplied by `1 + C01(m / 10) x 0.35`, blurred by `6 x Scale`
- stroke channel **`eyes`** -> `EyeClarity` at amount 0.9, face size `fs x 0.6`

---

## 6. Caching

Both `SkinAtlas.Build` and `SkinDefects.Build` are keyed by a **content print**: the average colour of a
4 x 4 grid, 48 floats, matching within 0.004 per value, plus the picture's size and the faces string. Two
entries each, oldest dropped.

So the atlas and all ten maps are computed once per photograph, and moving any slider is one pass over
the pixels. The print is on content rather than on identity because the renderer hands over a new picture
for every frame.

Both classes have a `Forget()`, and **neither is called anywhere**: `Caches.Clear()` in `Model.cs` clears
the mask, denoise, depth, blur, mesh and pool caches but not these two. In practice it does not bite,
because each holds only two entries and the print key is on content - a different picture simply misses
and evicts. It is still a real gap, and the ported version should not copy it.

---

## 7. Every parameter

In `AppTools.cs`, tool id `portrait`, all sliders 0-100. **Nothing is on until a slider is moved** - a
face is not something to correct by default.

| key | UI default | default in `Face.Portrait` | what it drives |
|---|---|---|---|
| `spotlight` | 0 | 25 | face spotlight |
| `smoothing` | 0 | 25 | the old whole-skin guided-filter smoothing |
| `eyes` | 0 | 25 | eye clarity |
| `spots` | 0 | 0 | the Spots map |
| `redness` | 0 | 0 | the Redness map |
| `rednessway` | 0 | 0 | which of the four targets |
| `wrinkles` | 0 | 0 | the Lines map |
| `shine` | 0 | 0 | the Shine map |
| `undereye` | 0 | 0 | the UnderEye map |
| `keeptexture` | 0 | **60** | the Texture map |
| `skintone` | 0 | 0 | one of six tone presets |
| `faces` | — | — | the detected boxes, `"x,y,w,h;..."` normalised |

The two columns differ because the code defaults predate the decision that nothing is applied
automatically. The UI always writes a value, so the code defaults only reach edits saved before those
keys existed - `keeptexture` is the one where that matters, since an old edit gets 60 and a new one 0.

The per-fault block only runs at all when a face was found **and** at least one of `spots`, `redness`,
`wrinkles`, `shine`, `undereye` is above zero. `keeptexture` alone does nothing.

---

## 8. What is already in RapidRAW, and what is missing

On branch `skin-by-fault` (`src-tauri/src/effects/skin.rs`, `src-tauri/src/shaders/skin.wgsl`):

| | state |
|---|---|
| Spots | ported, same blob-at-its-own-size method |
| Redness | ported, all four targets |
| Shine | ported |
| Keep texture | ported |
| the face's own colour | ported, measured on the card as a 256-bin histogram over the skin pixels, median and the 75th percentile for the spread |
| **Lines** | ported (`sk_valley`, `sk_band`, `sk_lines`): two widths of valley that follow the face's width, read against how much this skin varies from place to place, weighted by where a face folds |
| **Under eyes** | ported (`sk_lines`, applied in `sk_apply`): darker than the cheek below it, inside the under-eye band |
| the regions | ported as far as the maps need: `compose_skin_regions` in `masks/face.rs` builds the fold weight (Nasolabial + UnderEye x 0.8 + Forehead x 0.6) and the UnderEye band from the 478 landmarks, in the face's own upright frame, and the two travel with the edit as `skinLinesMap` and `skinUnderEyeMap`. Cheeks and NoseChin are not built: nothing ported reads them |
| skin smoothing, skin tone, spotlight, eye clarity | not ported, and arguably should not be: RapidRAW can already do all four through its existing adjustments on a face-part mask |
| the brush tool | not ported; RapidRAW's own brush masks cover it |

The regions were the piece of work left when this was written, and they are now done, built from the
478-point mesh in the face's own upright frame rather than from the anchor-and-oval scheme described
above - which existed in PixTune only because it had to cope with having no mesh at all. Two of the five
have no reader: Cheeks and NoseChin serve in PixTune only to fit the skin colour and to steer the line
weight, and the ported version arrives at both another way.

So every fault PixTune corrects is now corrected here. What is left of this document is reference
rather than plan: the numbers to compare against when a ported map behaves differently from the
original, and the reasoning behind the choices that look arbitrary until you know what they were
measured against.
