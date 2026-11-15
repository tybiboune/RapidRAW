// Skin by kind of fault.
//
// One smoothing slider blurs every bit of skin alike, which is why skin retouched that way looks like
// plastic: the blemish and the pores go together. Here each kind of fault gets its own map, and the
// correction is only allowed where its map says so. The skin's own grain goes back over whatever was
// done to it, so taking a blemish out does not cost the pores.
//
// Every map is read against this face's own colour, measured on the face itself rather than assumed,
// which is what lets the redness slider send a flush to the colour of the cheek beside it. It is taken
// as a middle and a spread rather than an average, because a face carries exactly the patches this is
// meant to find and they would drag an average towards themselves - the measure would move with what
// it is measuring.
//
// Brightness and colour are kept apart throughout. Redness comes out along the colour axes alone and
// leaves the brightness exactly as it was: correct a flush by darkening it and the face goes grey,
// which is the usual giveaway.

// A pair of numbers sits on an eight-byte boundary here, so the order is the layout and the Rust copy
// of this struct is written to come out the same.
struct Params {
    grid: Grid,
    spots: f32,
    redness: f32,
    shine: f32,
    keep_texture: f32,
    low_size: vec2<u32>,
    face_size: vec2<u32>,
    region_size: vec2<u32>,
    redness_way: u32,
    lines: f32,
    under_eye: f32,
    step_a: u32,
    step_b: u32,
    step_c: u32,
}

@group(0) @binding(7) var<uniform> params: Params;

const STATS_THREADS: u32 = 256u;
const BINS: u32 = 256u;
const SKIN_HERE: f32 = 0.6;

// the colour axes run from about -128 to 128, so one bin is one level
var<workgroup> cb_bins: array<atomic<u32>, 256>;
var<workgroup> cr_bins: array<atomic<u32>, 256>;
var<workgroup> seen: atomic<u32>;

const YCC_Y: vec3<f32> = vec3<f32>(65.481, 128.553, 24.966);
const YCC_CB: vec3<f32> = vec3<f32>(-37.797, -74.203, 112.0);
const YCC_CR: vec3<f32> = vec3<f32>(112.0, -93.786, -18.214);

fn to_ycc(rgb: vec3<f32>) -> vec3<f32> {
    let c = max(rgb, vec3<f32>(0.0));
    return vec3<f32>(16.0 + dot(YCC_Y, c), dot(YCC_CB, c), dot(YCC_CR, c));
}

fn to_rgb(y_level: f32, cb: f32, cr: f32) -> vec3<f32> {
    return vec3<f32>(
        y_level + 0.00625893 * cr,
        y_level - 0.00153632 * cb - 0.00318811 * cr,
        y_level + 0.00791071 * cb,
    );
}

fn bin_of(value: f32) -> u32 {
    return u32(clamp(value + 128.0, 0.0, 255.0));
}

// what the face's own colour came to: the middle of each axis, then how far it usually sits from it
fn skin_middle() -> vec4<f32> {
    return vec4<f32>(
        bitcast<f32>(atomicLoad(&scratch[0])),
        bitcast<f32>(atomicLoad(&scratch[1])),
        bitcast<f32>(atomicLoad(&scratch[2])),
        bitcast<f32>(atomicLoad(&scratch[3])),
    );
}

// t0: the picture, small. t1: how much of each pixel is face skin.
// Out: nothing; the four numbers this face's skin comes to, left in scratch for the passes after.
@compute @workgroup_size(256, 1, 1)
fn sk_stats(@builtin(local_invocation_index) li: u32) {
    if (li < BINS) {
        atomicStore(&cb_bins[li], 0u);
        atomicStore(&cr_bins[li], 0u);
    }
    if (li == 0u) { atomicStore(&seen, 0u); }
    workgroupBarrier();

    let size = params.low_size;
    let count = size.x * size.y;
    var mine = 0u;
    for (var i = li; i < count; i += STATS_THREADS) {
        let xy = vec2<u32>(i % size.x, i / size.x);
        let uv = (vec2<f32>(xy) + 0.5) / vec2<f32>(size);
        let skin = tap_sample(t1, tap_uv(uv, params.face_size)).x;
        if (skin < SKIN_HERE) { continue; }
        let ycc = to_ycc(textureLoad(t0, xy, 0).rgb);
        atomicAdd(&cb_bins[bin_of(ycc.y)], 1u);
        atomicAdd(&cr_bins[bin_of(ycc.z)], 1u);
        mine += 1u;
    }
    atomicAdd(&seen, mine);
    workgroupBarrier();
    if (li != 0u) { return; }

    let total = atomicLoad(&seen);
    // too little skin to measure anything on: the numbers a mid tone comes to, so nothing goes wild
    if (total < 64u) {
        atomicStore(&scratch[0], bitcast<u32>(-8.0));
        atomicStore(&scratch[1], bitcast<u32>(20.0));
        atomicStore(&scratch[2], bitcast<u32>(4.0));
        atomicStore(&scratch[3], bitcast<u32>(5.0));
        return;
    }
    let half_way = total / 2u;
    let three_quarters = total * 3u / 4u;
    var cb_run = 0u;
    var cr_run = 0u;
    var cb_mid = 0.0;
    var cr_mid = 0.0;
    var cb_high = 0.0;
    var cr_high = 0.0;
    var cb_done = 0u;
    var cr_done = 0u;
    for (var b = 0u; b < BINS; b += 1u) {
        let value = f32(b) - 128.0;
        cb_run += atomicLoad(&cb_bins[b]);
        cr_run += atomicLoad(&cr_bins[b]);
        if (cb_done == 0u && cb_run >= half_way) { cb_mid = value; cb_done = 1u; }
        if (cr_done == 0u && cr_run >= half_way) { cr_mid = value; cr_done = 1u; }
        if (cb_done == 1u && cb_run >= three_quarters) { cb_high = value; cb_done = 2u; }
        if (cr_done == 1u && cr_run >= three_quarters) { cr_high = value; cr_done = 2u; }
    }
    atomicStore(&scratch[0], bitcast<u32>(cb_mid));
    atomicStore(&scratch[1], bitcast<u32>(cr_mid));
    atomicStore(&scratch[2], bitcast<u32>(max(cb_high - cb_mid, 1.0)));
    atomicStore(&scratch[3], bitcast<u32>(max(cr_high - cr_mid, 2.0)));
}

// t0: the picture. t1: how much of each pixel is face skin, 0 to 1.
// Out: brightness, the two colour axes and the skin weight. The weight travels with them, so a blur of
// (value times weight) over a blur of weight is the average of the skin alone and nothing outside the
// face is averaged into the face's own numbers.
@compute @workgroup_size(8, 8, 1)
fn sk_split(@builtin(global_invocation_id) id: vec3<u32>) {
    if (outside(id)) { return; }
    let p = vec2<i32>(id.xy);
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(params.grid.dst);
    let skin = clamp(tap_sample(t1, tap_uv(uv, params.face_size)).x, 0.0, 1.0);
    let ycc = to_ycc(textureLoad(t0, p, 0).rgb);
    textureStore(out_f32, p, vec4<f32>((ycc.x - 16.0) / 219.0 * skin, ycc.y * skin, ycc.z * skin, skin));
}

// t0: the split picture. t1: it blurred at the size of a blemish. t2: blurred wider, which is the lie
// of the face. Out: the maps - spots, redness, shine - and the skin weight carried through.
@compute @workgroup_size(8, 8, 1)
fn sk_maps(@builtin(global_invocation_id) id: vec3<u32>) {
    if (outside(id)) { return; }
    let p = vec2<i32>(id.xy);
    let here = textureLoad(t0, p, 0);
    let skin = here.w;
    if (skin <= 0.01) {
        textureStore(out_f32, p, vec4<f32>(0.0, 0.0, 0.0, skin));
        return;
    }
    let fine = textureLoad(t1, p, 0);
    let wide = textureLoad(t2, p, 0);
    let middle = skin_middle();
    // every blurred copy carries the weight, so dividing by it gives the average over skin only
    let l_here = here.x / max(skin, 1e-4);
    let l_fine = fine.x / max(fine.w, 1e-4);
    let l_wide = wide.x / max(wide.w, 1e-4);
    let cr_fine = fine.z / max(fine.w, 1e-4);

    // A blemish is a blob of brightness at its own size: what the skin does at the size of a spot
    // against what it does over the area around it. Anything finer than the blob - the pores - is not
    // in that difference at all, which is what lets the spots go while they stay.
    let blob = l_fine - l_wide;
    let spots = skin * smooth_step(0.004, 0.030, max(-blob, 0.0) + max(blob, 0.0) * 0.55);

    // How far along the red axis a patch sits from this face's own middle, over an area wide enough to
    // be a flush or a patch of irritation rather than one noisy pixel. A patch a spread and a half
    // above the middle is as red as this face gets, so that is where the map reads full.
    let over = (cr_fine - middle.y) / max(middle.w, 2.0);
    let redness = skin * smooth_step(0.3, 1.5, over);

    // Shine is brightness above what the face is doing around it, never below.
    let shine = skin * smooth_step(0.05, 0.22, l_here - l_wide);

    textureStore(out_f32, p, vec4<f32>(spots, redness, shine, skin));
}

// How much this skin varies at the size of a blemish, point by point, to be blurred into how much it
// varies in each neighbourhood. That is the yardstick a valley is read against: a crease on skin that
// is smooth around it is a crease, and the same depth on skin that is rough all over is only the skin.
//
// t0: the split picture blurred at the size of a blemish. t1: blurred wider. t2: the split picture.
@compute @workgroup_size(8, 8, 1)
fn sk_band(@builtin(global_invocation_id) id: vec3<u32>) {
    if (outside(id)) { return; }
    let p = vec2<i32>(id.xy);
    let skin = textureLoad(t2, p, 0).w;
    let fine = textureLoad(t0, p, 0);
    let wide = textureLoad(t1, p, 0);
    let l_fine = fine.x / max(fine.w, 1e-4);
    let l_wide = wide.x / max(wide.w, 1e-4);
    textureStore(out_f32, p, vec4<f32>(abs(l_fine - l_wide) * skin, 0.0, 0.0, skin));
}

// The brightness of one of the blurred copies of the picture at a point, over skin only.
fn ridge_level(which: u32, pos: vec2<f32>) -> f32 {
    let uv = (pos + 0.5) / vec2<f32>(params.grid.dst);
    let tap = tap_uv(uv, params.grid.dst);
    var c = tap_sample(t0, tap);
    if (which == 1u) { c = tap_sample(t1, tap); }
    if (which == 2u) { c = tap_sample(t2, tap); }
    return c.x / max(c.w, 1e-4);
}

// How the brightness bends at a point, read at a step of h pixels: xx, yy and xy.
fn bend(t: texture_2d<f32>, p: vec2<i32>, h: i32) -> vec3<f32> {
    let hi = vec2<i32>(params.grid.dst) - vec2<i32>(1, 1);
    let at = vec2<i32>(h, 0);
    let down = vec2<i32>(0, h);
    let c = level_at(t, clamp(p, vec2<i32>(0, 0), hi));
    let l = level_at(t, clamp(p - at, vec2<i32>(0, 0), hi));
    let r = level_at(t, clamp(p + at, vec2<i32>(0, 0), hi));
    let u = level_at(t, clamp(p - down, vec2<i32>(0, 0), hi));
    let d = level_at(t, clamp(p + down, vec2<i32>(0, 0), hi));
    let ur = level_at(t, clamp(p + at - down, vec2<i32>(0, 0), hi));
    let dl = level_at(t, clamp(p - at + down, vec2<i32>(0, 0), hi));
    let ul = level_at(t, clamp(p - at - down, vec2<i32>(0, 0), hi));
    let dr = level_at(t, clamp(p + at + down, vec2<i32>(0, 0), hi));
    let h2 = f32(h * h);
    return vec3<f32>((l + r - 2.0 * c) / h2, (u + d - 2.0 * c) / h2, (dr - ur - dl + ul) / (4.0 * h2));
}

fn level_at(t: texture_2d<f32>, p: vec2<i32>) -> f32 {
    let c = textureLoad(t, p, 0);
    return c.x / max(c.w, 1e-4);
}

// What one width says about a point: how strongly a line runs through it and which way is across it.
// Returns (strength, across.x, across.y, depth).
//
// Across a dark line the brightness bends sharply upward, along it hardly at all: of the two curvatures
// the larger is positive and large and the other is near nothing. A pore bends both ways, an edge bends
// up on one side and down on the other, and noise bends every which way at a small size, so the ratio of
// the two curvatures is what tells a line from the rest. The depth is in units of brightness, so it can
// be read against how rough this skin is.
fn ridge(t: texture_2d<f32>, p: vec2<i32>, h: i32, yardstick: f32) -> vec4<f32> {
    let m = bend(t, p, h);
    let mean = 0.5 * (m.x + m.y);
    let half_gap = 0.5 * (m.x - m.y);
    let root = sqrt(half_gap * half_gap + m.z * m.z);
    let strong = mean + root;
    let weak = mean - root;
    if (strong <= 0.0) { return vec4<f32>(0.0, 1.0, 0.0, 0.0); }
    // The direction across the line is the one the strong curvature belongs to. Either of two forms of it
    // gives that, and each is empty in the case the other handles best (a line running exactly along an
    // axis), so the longer of the two is the one to trust.
    let by_x = vec2<f32>(m.z, strong - m.x);
    let by_y = vec2<f32>(strong - m.y, m.z);
    var across = vec2<f32>(1.0, 0.0);
    if (max(dot(by_x, by_x), dot(by_y, by_y)) > 1e-18) {
        across = normalize(select(by_y, by_x, dot(by_x, by_x) >= dot(by_y, by_y)));
    } else if (m.x < m.y) {
        across = vec2<f32>(0.0, 1.0);
    }
    let depth = strong * f32(h * h) * 2.0;
    let single = 1.0 - smooth_step(0.35, 0.95, abs(weak) / strong);
    let clear = smooth_step(0.45, 1.6, depth / (yardstick * 1.5));
    return vec4<f32>(clear * single, across, depth);
}

// t0, t1, t2: the split picture blurred to the three widths lines are looked for at. t3: the yardstick,
// blurred. t4: the regions of the face, the folds in the first channel and the band under each eye in the
// second. t5: the split picture blurred wide, the lie of the face.
// Out: how sure this is a line, how much of it is under-eye shadow, the brightness of the skin on either
// side of the line (what it is lifted towards), and the skin weight.
@compute @workgroup_size(8, 8, 1)
fn sk_lines(@builtin(global_invocation_id) id: vec3<u32>) {
    if (outside(id)) { return; }
    let p = vec2<i32>(id.xy);
    let skin = textureLoad(t0, p, 0).w;
    if (skin <= 0.05) {
        textureStore(out_f32, p, vec4<f32>(0.0, 0.0, 0.0, skin));
        return;
    }
    let spread = textureLoad(t3, p, 0);
    let yardstick = max(spread.x / max(spread.w, 1e-4), 0.004);
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(params.grid.dst);
    let region = tap_sample(t4, tap_uv(uv, params.region_size));

    // the three widths, and the one that finds the strongest line here
    let a = ridge(t0, p, i32(params.step_a), yardstick);
    let b = ridge(t1, p, i32(params.step_b), yardstick);
    let c = ridge(t2, p, i32(params.step_c), yardstick);
    var best = a;
    var which = 0u;
    var reach = f32(params.step_a);
    if (b.x > best.x) { best = b; which = 1u; reach = f32(params.step_b); }
    if (c.x > best.x) { best = c; which = 2u; reach = f32(params.step_c); }

    // The valley is filled from the skin on either side of it, found by going across the line: its own
    // brightness is replaced by the average of the two sides. Blurring it instead would smear it into the
    // skin along its length, which is what makes a healed line look like a smudge.
    let pos = vec2<f32>(id.xy);
    let across = best.yz;
    let side = 2.2 * reach;
    let target_level = 0.5 * (ridge_level(which, pos + across * side) + ridge_level(which, pos - across * side));

    // a line counts for most where a face folds anyway - beside the nose, under the eyes, across the
    // forehead - and still counts for something everywhere else on the skin
    let folds = 0.45 + 0.55 * clamp(region.x, 0.0, 1.0);
    let line = skin * best.x * folds;

    // under the eye is darker than the cheek below it, inside the band where that happens
    let l_wide = textureLoad(t5, p, 0).x / max(textureLoad(t5, p, 0).w, 1e-4);
    let deficit = (l_wide - ridge_level(0u, pos)) / max(l_wide, 0.02);
    let under_eye = clamp(region.y, 0.0, 1.0) * smooth_step(0.015, 0.16, deficit);

    textureStore(out_f32, p, vec4<f32>(line, under_eye, target_level, skin));
}

// t0: the picture. t1: the maps. t2: the split picture blurred at the size of a blemish, which is the
// skin with its grain taken off. t3: blurred wider, the lie of the face. t4: the lines and the
// under-eye band, which is only read when one of those two is asked for.
// Out: the picture with each fault corrected by its own slider, and nothing else touched.
@compute @workgroup_size(8, 8, 1)
fn sk_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    if (outside(id)) { return; }
    let p = vec2<i32>(id.xy);
    let texel = textureLoad(t0, p, 0);
    let maps = textureLoad(t1, p, 0);
    let skin = maps.w;
    if (skin <= 0.01) {
        textureStore(out_f16, p, texel);
        return;
    }
    var ycc = to_ycc(texel.rgb);
    let l0 = (ycc.x - 16.0) / 219.0;
    let fine = textureLoad(t2, p, 0);
    let wide = textureLoad(t3, p, 0);
    let l_fine = fine.x / max(fine.w, 1e-4);
    let l_wide = wide.x / max(wide.w, 1e-4);
    // the pores, the grain, the stubble: everything finer than a blemish, kept aside to go back on top
    var grain = l0 - l_fine;
    var level = l0;

    // a blemish is a blob of brightness at its own size: take that blob away and the skin around it
    // closes over, while anything finer than the blob stays where it was
    if (params.spots > 0.0) {
        level -= (l_fine - l_wide) * maps.x * params.spots;
    }
    // A line is a valley: lift it part of the way up to the skin on either side of it. How far was worked
    // out against the brightness the picture came in with, so the spots just taken out are not counted
    // twice.
    let extra = textureLoad(t4, p, 0);
    if (params.lines > 0.0) {
        // up towards the skin on either side, as far as the slider and how sure this is a line allow
        let lift = max(extra.z - l0, 0.0) * extra.x * params.lines;
        level += lift;
        // what was lifted out of the valley is no longer grain either, or putting the grain back on top
        // would put the line back with it
        grain += lift;
    }
    // shine is brightness above what the face is doing around it, moved back towards it, never past it
    if (params.shine > 0.0) {
        let over = level - l_wide;
        if (over > 0.0) { level -= over * maps.z * params.shine * 0.85; }
    }
    // the shadow under an eye is brightness below what the cheek is doing, brought back up towards it
    if (params.under_eye > 0.0) {
        let under = l_wide - level;
        if (under > 0.0) { level += under * extra.y * params.under_eye * 0.8; }
    }
    // and the skin's own grain goes back on top of whatever was done to it
    if (params.keep_texture > 0.0) {
        level += grain * params.keep_texture * skin;
    }

    // Redness comes out along the colour axes alone, with the brightness left exactly as it is. Where
    // it goes is a choice, because there is no single right answer - a flush taken out completely can
    // look waxy, left in it can look sunburnt - so the owner picks one of four:
    //   0  this skin: the patch comes out the colour of the cheek beside it
    //   1  half way: the flush calmed rather than erased
    //   2  warm: the skin colour with some warmth kept, for faces that look dead without it
    //   3  even: the red taken out and nothing put back, the flattest of the four
    var cb = ycc.y;
    var cr = ycc.z;
    if (params.redness > 0.0) {
        let middle = skin_middle();
        var to_cb = middle.x;
        var to_cr = middle.y;
        var cb_share = 0.35;
        var reach = 1.0;
        switch params.redness_way {
            case 1u: { reach = 0.5; }
            case 2u: { to_cr = middle.y + middle.w; cb_share = 0.15; }
            case 3u: { to_cr = middle.y - middle.w * 0.6; cb_share = 0.7; }
            default: {}
        }
        // The map already carries the skin weight and the whole correction is faded by that same
        // weight at the end: counting it twice is what keeps a full slider from ever arriving.
        let k = clamp(maps.y / max(skin, 0.05), 0.0, 1.0) * params.redness * reach;
        cr -= (cr - to_cr) * k;
        cb -= (cb - to_cb) * k * cb_share;
    }

    let fixed_rgb = to_rgb(level, cb, cr);
    // only as far as the mask says, so the edge of a correction is never a visible border
    textureStore(out_f16, p, vec4<f32>(mix(texel.rgb, fixed_rgb, skin), texel.a));
}
