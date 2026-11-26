struct Transform {
    rect: vec4<f32>,
    clip: vec4<f32>,
    window: vec2<f32>,
    image_size: vec2<f32>,
    texture_size: vec2<f32>,
    pixelated: f32,
    hdr_white: f32,
    bg_primary: vec4<f32>,
    bg_secondary: vec4<f32>,
    original_size: vec2<f32>,
    split_origin: vec2<f32>,
    split_normal: vec2<f32>,
    split_offset: vec2<f32>,
    lut: vec4<f32>,
    frame_curve: vec4<f32>,
    frame_to_srgb: mat3x3<f32>,
    linear_max: f32,
    peak: vec4<f32>,
    peak_color: vec4<f32>,
    // the focus map: x is 1 while it is the shown photo's, y its number of levels; and the matrix rows
    // taking the picture's uv to the map's
    peak_map: vec4<f32>,
    peak_u: vec4<f32>,
    peak_v: vec4<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var original_tex: texture_2d<f32>;
@group(0) @binding(4) var color_lut: texture_3d<f32>;
// how blurred each edge of the photo is (see render/focus_map.rs): the blur in 32nds of a pixel, 255 for
// no edge, with a level per halving, each the sharpest of the 2x2 below it
@group(0) @binding(5) var focus_map: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0)
    );
    let pos = uvs[id];

    let uv_x = transform.clip.x + pos.x * transform.clip.z;
    let uv_y = transform.clip.y + pos.y * transform.clip.w;

    let half_pixel_x = 0.5;
    let half_pixel_y = 0.5;
    let outset_x = (pos.x * 2.0 - 1.0) * half_pixel_x;
    let outset_y = (pos.y * 2.0 - 1.0) * half_pixel_y;

    let screen_x = uv_x + outset_x;
    let screen_y = uv_y + outset_y;

    let ndc_x = (screen_x / transform.window.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_y / transform.window.y) * 2.0;

    var out: VertexOutput;
    out.pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    out.uv = vec2<f32>(
        (uv_x - transform.rect.x) / transform.rect.z,
        (uv_y - transform.rect.y) / transform.rect.w
    );

    return out;
}

fn sample_layer(layer: texture_2d<f32>, uv: vec2<f32>, image_size: vec2<f32>, texture_size: vec2<f32>) -> vec4<f32> {
    let adjusted_uv = uv * (image_size / texture_size);
    let half_texel = vec2<f32>(0.5, 0.5) / texture_size;

    let min_uv = half_texel;
    let max_uv = (image_size / texture_size) - half_texel;

    if (transform.pixelated > 0.5) {
        let texel_coords = floor(adjusted_uv * texture_size);
        let nearest_uv = (texel_coords + vec2<f32>(0.5, 0.5)) / texture_size;

        let clamped_nearest = clamp(nearest_uv, min_uv, max_uv);
        return textureSampleLevel(layer, samp, clamped_nearest, 0.0);
    } else {
        let clamped_uv = clamp(adjusted_uv, min_uv, max_uv);
        return textureSampleLevel(layer, samp, clamped_uv, 0.0);
    }
}

fn peak_luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// What is in focus here, drawn over the picture and changing nothing about it: no adjustment, no
// hash, nothing that reaches the export.
//
// The brightness of the 2x2 texels around a texel corner (`corner` in texels of the picture): the
// linear sampler averages them in one read. Always linear, also when the picture is shown pixelated.
fn peak_box(corner: vec2<f32>) -> f32 {
    let size = transform.image_size;
    let tex_size = transform.texture_size;
    let uv = clamp(corner / tex_size, vec2<f32>(0.5) / tex_size, (size - vec2<f32>(0.5)) / tex_size);
    return peak_luma(textureSampleLevel(tex, samp, uv, 0.0).rgb);
}

// The 4x4 texels around a texel corner: four 2x2 reads.
fn peak_box4(corner: vec2<f32>) -> f32 {
    return 0.25 * (peak_box(corner + vec2<f32>(-1.0, -1.0)) + peak_box(corner + vec2<f32>(1.0, -1.0))
        + peak_box(corner + vec2<f32>(-1.0, 1.0)) + peak_box(corner + vec2<f32>(1.0, 1.0)));
}

// How much the brightness turns across `corner`, read on samples `reach` texels apart, but no more
// than it leans: an edge both leans and turns, noise turns about twice as much as it leans, at random,
// and a soft edge leans without turning.
fn peak_turn(here: f32, right: f32, left: f32, below: f32, above: f32) -> f32 {
    let turn = abs(right + left - 2.0 * here) + abs(below + above - 2.0 * here);
    let lean = 0.5 * (abs(right - left) + abs(below - above));
    return min(turn, lean);
}

// How much visible detail the picture on screen has at a point.
//
// Detail is read as the second difference of the brightness across each axis, not the first. A
// gradient - a soft shoulder of light on a cheek, the sky shading off - has a large first difference
// everywhere and is not sharp at all; only detail that turns, rather than merely leans, raises the
// second. That is what separates an edge in focus from a bright slope out of it.
//
// Noise turns too, from one texel to the next, so read on single texels it looks as sharp as an edge
// (at high ISO, sharper: a smooth sky lit up all over). Three things keep it out:
// - the differences are taken on 2x2 averages two texels apart, which halves the noise and keeps an
//   edge in focus, which spans a texel or two, nearly whole;
// - the turn counts only as far as the brightness also leans across it (`peak_turn`);
// - the same is read again at twice the scale, on 4x4 averages four texels apart, and the finer
//   reading counts only as far as the coarser one backs it. An edge in focus is there at both scales
//   with the same strength; noise mostly averages out of the coarser one, which is what leaves the
//   lone specks in foliage and skin.
// On synthetic pictures with noise of 0.02 (in display brightness, 0 to 1), single texels read the
// noise at up to 0.37, far above an edge in focus of 0.1 contrast (0.03); this reads the noise at up
// to 0.027 and that edge at 0.045.
fn peak_detail(uv: vec2<f32>) -> f32 {
    let corner = floor(uv * transform.image_size + vec2<f32>(0.5));
    let fine = peak_turn(
        peak_box(corner),
        peak_box(corner + vec2<f32>(2.0, 0.0)),
        peak_box(corner - vec2<f32>(2.0, 0.0)),
        peak_box(corner + vec2<f32>(0.0, 2.0)),
        peak_box(corner - vec2<f32>(0.0, 2.0))
    );
    let coarse = peak_turn(
        peak_box4(corner),
        peak_box4(corner + vec2<f32>(4.0, 0.0)),
        peak_box4(corner - vec2<f32>(4.0, 0.0)),
        peak_box4(corner + vec2<f32>(0.0, 4.0)),
        peak_box4(corner - vec2<f32>(0.0, 4.0))
    );
    // a coarser reading of the same edge is about half as strong, so it backs the finer one at twice
    return min(fine, 2.0 * coarse);
}

// Peaking from the focus map, crossed with the screen. The map says how blurred an edge is, whatever its
// contrast, but taken alone it also passes the odd noise spike whose blur reads small, and marks every
// texel of a finely textured area in focus; the screen says whether there is visible detail at all, but
// takes contrast for focus. A texel is marked as far as both agree: in focus by the map (up to a blur of
// 0.6 to 2.4 pixels of the full photo, by the slider, evenly in ratio), and detail on screen that stands
// clear of noise (a fixed bar, so the slider means one thing only).
fn map_peak(blur: f32, detail: f32) -> f32 {
    if (blur < 0.0) {
        return 0.0;
    }
    let limit = 0.6 * pow(4.0, clamp(transform.peak.y, 0.0, 1.0));
    let in_focus = 1.0 - smoothstep(limit, limit * 1.5, blur);
    let visible = smoothstep(0.012, 0.03, detail);
    return in_focus * visible;
}

// Where the picture's uv falls on the focus map, in texels of its finest level.
fn map_texel(uv: vec2<f32>) -> vec2<f32> {
    let p = vec3<f32>(uv, 1.0);
    let map_uv = vec2<f32>(dot(transform.peak_u.xyz, p), dot(transform.peak_v.xyz, p));
    return map_uv * vec2<f32>(textureDimensions(focus_map, 0));
}

// The blur of the sharpest edge under a screen pixel, in pixels of the full photo, or -1 where there is no
// edge. Read from the level of the map whose texels are about a screen pixel wide: each holds the sharpest
// of the finer texels under it, so no edge is lost however far the picture is zoomed out.
fn map_blur(texel: vec2<f32>, footprint: f32) -> f32 {
    let base = vec2<f32>(textureDimensions(focus_map, 0));
    if (texel.x < 0.0 || texel.y < 0.0 || texel.x >= base.x || texel.y >= base.y) {
        return -1.0;
    }
    let levels = i32(transform.peak_map.y);
    let level = clamp(i32(ceil(log2(max(footprint, 1.0)))), 0, levels - 1);
    let dims = vec2<i32>(textureDimensions(focus_map, level));
    // a level's last texel also holds the odd row or column left over below it
    let at = min(vec2<i32>(floor(texel / exp2(f32(level)))), dims - vec2<i32>(1));
    let code = textureLoad(focus_map, at, level).r * 255.0;
    if (code > 254.5) {
        return -1.0;
    }
    return code / 32.0;
}

fn frame_to_linear(encoded: vec3<f32>) -> vec3<f32> {
    let curve = transform.frame_curve;
    let value = max(encoded, vec3<f32>(0.0));
    let toe = value / max(curve.z, 0.000001);
    let power = pow((value + curve.y) / (1.0 + curve.y), vec3<f32>(1.0 / curve.x));
    return select(power, toe, value <= vec3<f32>(curve.w * curve.z));
}

fn present(color: vec4<f32>) -> vec4<f32> {
    if (transform.hdr_white > 0.0) {
        let linear = transform.frame_to_srgb * frame_to_linear(color.rgb) * transform.hdr_white;
        return vec4<f32>(min(linear, vec3<f32>(transform.linear_max)), color.a);
    }
    if (transform.lut.x > 0.0) {
        let shaped = pow(frame_to_linear(min(color.rgb, vec3<f32>(1.0))), vec3<f32>(transform.lut.z));
        let position = shaped * transform.lut.x + transform.lut.y;
        return vec4<f32>(textureSampleLevel(color_lut, samp, position, 0.0).rgb, color.a);
    }
    return color;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // how many texels of the focus map one screen pixel spans: read first, while every pixel of the quad
    // still runs, as a derivative must be
    let map_at = map_texel(in.uv);
    let footprint = max(length(dpdx(map_at)), length(dpdy(map_at)));

    let is_original = dot(in.pos.xy - transform.split_origin, transform.split_normal) < 0.0;
    let uv = select(in.uv, in.uv + transform.split_offset / transform.rect.zw, is_original);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return present(transform.bg_secondary);
    }

    if (is_original) {
        return present(sample_layer(original_tex, uv, transform.original_size, transform.original_size));
    }
    let colour = sample_layer(tex, uv, transform.image_size, transform.texture_size);
    // Nothing is marked until the photo's focus map is there and fits the picture shown: marks read some
    // other way first and then replaced would be two different answers in a second.
    if (transform.peak.x <= 0.0 || transform.peak_map.x <= 0.5) {
        return present(colour);
    }
    // over the edited side only: the point of it is to judge the picture being worked on
    let strength = map_peak(map_blur(map_at, footprint), peak_detail(uv));
    let marked = mix(colour.rgb, transform.peak_color.rgb, strength * transform.peak.x);
    return present(vec4<f32>(marked, colour.a));
}
