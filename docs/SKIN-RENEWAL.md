# Skin renewal: the "1-minute skin retouch" as one button

This document studies the PiXimperfect tutorial *Flawless 1-Min Skin Retouch Action in Photoshop!*
(youtube.com/watch?v=0q_GxbuEr_M, 15 min 48 s, chapters "How it Works" to "Fully Automatic Retouch Actions"), says what
it does and where it leans on the retoucher's eye, and describes how RapiderRaw does it by itself.

## 1. What the tutorial does

The technique builds a new skin out of three ingredients and puts it on where the retoucher paints.

| Step (tutorial) | What it makes | Values said in the video |
|---|---|---|
| Two copies of the picture, the lower one called "color" | the layer that carries the skin's colour and tone | |
| **Filter > Noise > Median** on "color", as a smart filter | an even base: every blemish, blotch and fold narrower than the radius is taken out, while edges stay "contained" | "about 120 works best for this one" (a picture of 4000 x 4000, 8-bit, the face about 2650 px wide: 4.5 % of it). Not so high that the nose blends, not so low that edges remain inside the skin |
| A new layer, **Overlay**, filled with 50 % grey | a layer that is invisible until something is drawn on it | |
| **Add Noise** (uniform, monochromatic) on that layer | grain: "the skin texture is similar to noise or grain" | "about 30 %": measured on his Gaussian Blur preview, the noise layer deviates about 13 levels of 255 from grey before the blur and about 5.5 after it (0.022); what shows in his finished skin is about 3.6 levels (0.014) |
| **Gaussian Blur** on that noise | pores, not pixel noise | 6 to 8 is "way too much"; "about 1.6, or even 2"; kept at **1.8** (on the 2650 px face: 0.068 % of its width) |
| The top copy, called "OG texture", **High Pass** | the original's finest detail only | the radius is "the exact number you chose for the Gaussian blur": **1.8** ("stop where you begin to see halos") |
| OG texture set to **Linear Light** | the detail is added to the new skin, doubled | opacity later lowered to **72 to 74 %** |
| The three layers grouped, **Blend If** on the group, Alt-split sliders | the effect is taken off the highlights and the shadows of the layer under it | "keep it about this way", by eye |
| A **black layer mask**, painted with white, soft round brush, flow up to 100 % | the effect appears only where it is painted | "paint over the areas where you want to apply it"; "it does not do very well around the edges"; low flow to take it off again |
| Group opacity, texture opacity | strength | "you can always decrease the opacity to make it more natural" |
| The action is recorded; values differ by picture | one shortcut | "the resolution you shoot in is different, in some photos the face is a large part of the image and in some a small head, so the values are going to be different all over the place" |
| The paid version asks, at play time, for the median radius, the blur, the high-pass radius, and how many highlights and shadows to reveal | | |

In short: **final skin = even base + new fine texture + the original's finest detail, kept off highlights and shadows, and
painted on where needed.**

## 2. What the retoucher decides by eye

Reading the transcript for every sentence of the form "stop at the point where", "by eye", "paint where":

1. **How big the median radius is** (120 here): "everything evened out, but not so much that the nose blends".
2. **How strong and how fine the texture is** (30 %, blur 1.8, "looks natural").
3. **The high-pass radius** (= the blur) and its opacity (about 72 %): "stop where you begin to see halos".
4. **Where the highlights and shadows end**, for Blend If.
5. **Where to paint**: the areas "with irregularities"; and how hard (flow).
6. **What to avoid**: edges, the neck's outline, where the effect "does not do very well".
7. **How much in the end** (group opacity), "to make it more natural".
8. **Which values suit this picture**, because resolution and face size change them all.

## 3. How each decision is made by the program

Everything below is worked out from the picture and the face the program already finds; nothing is a fixed pixel value, so
a small face in a large picture and a large one are treated alike (every size is a share of the face's width, `F`).

| Decision | How it is made automatically |
|---|---|
| **Median radius** | Five radii are tried (2 %, 3 %, 4.5 %, 7 % and 10 % of `F`). For each, two numbers are measured on the skin: *unevenness* (what is left of the skin's middle-sized variation) and *structure kept* (how much of the skin's real edges, from the strongest 15 % of gradients, survive). The radius chosen is the smallest that has taken out most of what the biggest takes out, among those that keep at least 60 % of the structure. This is "evened out, but the nose not blended" as a rule. In the portrait used to build it, 3 % of `F` (35 px on a 1167 px face). |
| **The median itself** | Photoshop's Median, done the same way: on each of red, green and blue, every pixel takes the middle value of the pixels within the radius (a round neighbourhood, the edge pixels repeated past the picture's border). It runs on the full-size picture with a sliding histogram (Huang's method), so a large radius costs a few times a small one rather than its square. The five candidate radii are compared on a copy whose face is at most about 1200 px wide; the chosen one is run at full size. |
| **The face's volume** | The median also clips the face's own light: the top of a cheekbone, the shade beside the nose, the wings of the nose, the eyelids, the lips' edges come out flatter. What the median took from the brightness is given back, spread over **a fifth of the median's radius** and counted only over the skin, so contours come back while blemishes, narrower still, stay evened. Half a radius, tried first, left noses and faces seen lying down looking flat. |
| **Pore grain** (the blur of 1.8) | The tutorial's, scaled to the face: **1.8 px on a 2650 px face** (0.068 % of `F`), between 0.5 and 4 px. Measuring the grain of the face's own cleanest skin, tried first, made it 30–40 % finer than his on most portraits. |
| **Texture strength** (the 30 %) | The noise layer deviates **0.014** from grey in the gamma-coded copy (about 3.6 levels of 255), the strength his finished skin shows, laid on in **Overlay**, as in the video. The *Texture* slider (50 = as in the tutorial) moves it. Copying the strength of the face's cleanest skin, tried first, made it 3 to 5 times too faint on soft pictures, which then looked plastic; his exact setting (0.025) looked like grey sand on light, smooth skin. |
| **High-pass radius and blend** | The tutorial's rule, exactly: the High Pass radius *is* the grain's blur. The High Pass is taken on each channel and laid on in **Linear Light at 72 %**, which adds the original's fine detail at about 1.44 times. One addition to the video: the detail is **soft-clipped** at 1.5 times its usual size in clean skin, so that a bump's bright tip, a blemish's rim or a hair is not brought back as "texture". At 2.5 times, the bumps of a nose and the pebbles of a shiny cheek came back as they were. |
| **Highlights and shadows (Blend If)** | Shadows only, taken from the skin's own tones: the effect fades out between the 0.2nd and 4th percentile of the skin's brightness, with a smooth ramp (the Alt-split of the sliders). The brightness judged is that of the *evened-out* skin, not of each pixel: a dark blemish on mid-tone skin is a blemish to take out, while a shadow is dark all around it and stays. (Judged per pixel, the darkest blemishes were protected as "shadows": a test caught it.) The highlights are **not** left alone: a shine on bumpy skin is where the bumps show most, and kept out (from the 93rd percentile up, tried first) it stood out as a rough island on evened skin. |
| **Where to paint** | **The whole skin**, as the tutorial paints it. A *need map* (how far each place is from its own evened-out self, judged against the rest of the same face, with a floor of 20 %) was tried first: on skin that is rough all over most of the face got the floor, so after the amount only 20–30 % of the renewal was applied and irregular skin was hardly touched. The need map is still worked out, only to find the cleanest skin the usual detail is measured in; contours wider than about 1 % of `F` do not count as unevenness there. |
| **What to avoid** | An **edge guard**: gradients of the lightly blurred skin, scaled to its 99th percentile (only the strongest edges: nostrils, lip line, eye outline, a beard's edge), take the effect away near those edges, and the skin mask's own border is feathered. The percentile is high on purpose: a blemish has strong edges of its own, and it is the thing to take out, not to keep. This replaces "it does not do well around the edges". A **line guard** keeps lashes and stray hairs: places where the light curves strongly across and hardly along, looked for at 0.15 % of `F` or the grain's size, whichever is bigger. The rim of a round bump fools that test (across the rim the light curves, along it hardly), so places that curve alike both ways, and their rims, are taken out of it: before, every bump on a nose was kept as a "line". |
| **A face too big to work out at full size** | A face box over about 16 million pixels is worked out on a copy reduced by a whole factor, and the change is brought back to full size. The full size's own detail finer than the copy is then soft-clipped the same way (1.5 times its usual size in the skin): left as it was, the bumps finer than the copy came back untouched. |
| **How much in the end** | One slider, **Renew** (the group opacity), from 0 to 100; *Auto renew* sets it to 70. Because the base, the texture and the mask are measured separately, it only blends: nothing is worked out again when it moves. |
| **Which values suit this picture** | All sizes are shares of the face's width, and the skin's brightness is scaled to mid-grey before anything is done, so the same code serves any resolution, exposure and face size. |

### One button

*Skin > Auto renew* (or the **Renew** slider) is the whole tool. The skin switch finds the face; the face's skin mask is the
program's version of the mask the retoucher paints; the measurements above are the retoucher's eye. The sliders *Renew*
and *Texture* are only for taste.

## 4. Where it sits in the program

* `src-tauri/src/effects/renewal.rs`: the measurements and the renewed layer, kept as a *change* to add to the picture
  (one for the face's box), so the amount can move without any of it being done again.
* It runs on the developed picture before the GPU pipeline, after any AI denoise and negative conversion and before
  the brushed repairs, the upscaler and Superfocus; its inputs are part of the preview and export cache keys.
* The face skin mask is the one the Skin tool already makes (`skinFaceMap`).

## 5. Limits

* One face's skin mask is renewed; groups are treated as one box (the same measures for all).
* The texture is monochromatic grain of the tutorial's size and strength scaled to the face: the video uses no
  ready-made skin texture, only Add Noise blurred to pore size. A texture taken from the face's own cleanest patches
  would be a closer match and is a possible next step.
* On very bumpy skin the renewal now smooths clearly more than before (pore detail about 20–30 % below the original
  on the roughest review portraits); *Texture* and *Renew* are there to bring some back.
* Picture noise coarser than the grain (a dark, high-ISO shot) is evened on the skin like any texture, so the skin can
  look cleaner than the noisy background around it: denoise first.
* It is for retouching skin in the sense of the tutorial, not dodging and burning: the shape and shading of the face are
  kept.
