# Skin renewal: the "1-minute skin retouch" as one button

This document studies the PiXimperfect tutorial *Flawless 1-Min Skin Retouch Action in Photoshop!*
(youtube.com/watch?v=0q_GxbuEr_M, 15 min 48 s, chapters "How it Works" to "Fully Automatic Retouch Actions"), says what
it does and where it leans on the retoucher's eye, and describes how RapiderRaw does it by itself.

## 1. What the tutorial does

The technique builds a new skin out of three ingredients and puts it on where the retoucher paints.

| Step (tutorial) | What it makes | Values said in the video |
|---|---|---|
| Two copies of the picture, the lower one called "color" | the layer that carries the skin's colour and tone | |
| **Filter > Noise > Median** on "color", as a smart filter | an even base: every blemish, blotch and fold narrower than the radius is taken out, while edges stay "contained" | "about 120 works best for this one" (a picture of about 5000 x 4000, the face large in it). Not so high that the nose blends, not so low that edges remain inside the skin |
| A new layer, **Overlay**, filled with 50 % grey | a layer that is invisible until something is drawn on it | |
| **Add Noise** (uniform, monochromatic) on that layer | grain: "the skin texture is similar to noise or grain" | "about 30 %" |
| **Gaussian Blur** on that noise | pores, not pixel noise | 6 to 8 is "way too much"; "about 1.6, or even 2"; kept at **1.8** |
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
| **Pore grain** (the blur of 1.8) | The grain size is *measured*: the width at which the autocorrelation of the cleanest skin's finest detail falls to half is the grain of that face. The blur is 0.85 times that width, between 0.8 and 4 px. |
| **Texture strength** (the 30 %) | The noise layer's strength is the measured root-mean-square of that finest detail in the cleanest skin, so the new grain is as rough as the face's own good skin. It is laid on in **Overlay**, as in the video. The *Texture* slider (50 = as measured) moves it. |
| **High-pass radius and blend** | The tutorial's rule, exactly: the High Pass radius *is* the grain's blur. The High Pass is taken on each channel and laid on in **Linear Light at 72 %**, which adds the original's fine detail at about 1.44 times. One addition to the video: the detail is **soft-clipped** at 2.5 times its usual size in clean skin, so that a hair's or a blemish's rim is not brought back as "texture". |
| **Highlights and shadows (Blend If)** | Taken from the skin's own tones: the effect fades out between the 93rd and 99.5th percentile of the skin's brightness, and between the 0.2nd and 4th at the dark end, with smooth ramps (the Alt-split of the sliders). The brightness judged is that of the *evened-out* skin, not of each pixel: a dark blemish on mid-tone skin is a blemish to take out, while a shadow is dark all around it and stays. (Judged per pixel, the darkest blemishes were protected as "shadows": a test caught it.) |
| **Where to paint** | A **need map**: for each place, how far it is from its own evened-out self, three ways (brightness 45 %, colour 40 %, pore roughness 15 %), each in units of what is usual for this skin, then mapped between its 25th and 92nd percentile. Clean skin gets a floor of 40 % so that it is evened a little, blotchy skin gets all of it. This replaces "paint where there are irregularities" and the flow. |
| **What to avoid** | An **edge guard**: gradients of the lightly blurred skin, scaled to its 99th percentile (only the strongest edges: nostrils, lip line, eye outline, a beard's edge), take the effect away near those edges, and the skin mask's own border is feathered. The percentile is high on purpose: a blemish has strong edges of its own, and it is the thing to take out, not to keep. This replaces "it does not do well around the edges". |
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
* The texture is monochromatic grain with a measured size and strength, as the tutorial's is: the video uses no
  ready-made skin texture, only Add Noise blurred to pore size. A texture taken from the face's own cleanest patches
  would be a closer match and is a possible next step.
* It is for retouching skin in the sense of the tutorial, not dodging and burning: the shape and shading of the face are
  kept.
