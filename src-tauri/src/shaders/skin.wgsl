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

struct Params {
    grid: Grid,
    spots: f32,
    redness: f32,
    shine: f32,
    keep_texture: f32,
    redness_way: u32,
    low_size: vec2<u32>,
    face_size: vec2<u32>,
    _pad0: u32,
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

// t0: the picture. t1: the maps. t2: the split picture blurred at the size of a blemish, which is the
// skin with its grain taken off. t3: blurred wider, the lie of the face.
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
    let grain = l0 - l_fine;
    var level = l0;

    // a blemish is a blob of brightness at its own size: take that blob away and the skin around it
    // closes over, while anything finer than the blob stays where it was
    if (params.spots > 0.0) {
        level -= (l_fine - l_wide) * maps.x * params.spots;
    }
    // shine is brightness above what the face is doing around it, moved back towards it, never past it
    if (params.shine > 0.0) {
        let over = level - l_wide;
        if (over > 0.0) { level -= over * maps.z * params.shine * 0.85; }
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
