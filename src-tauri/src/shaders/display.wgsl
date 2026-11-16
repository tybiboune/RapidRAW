struct Transform {
    rect: vec4<f32>,
    clip: vec4<f32>,
    window: vec2<f32>,
    image_size: vec2<f32>,
    texture_size: vec2<f32>,
    pixelated: f32,
    _pad: f32,
    bg_primary: vec4<f32>,
    bg_secondary: vec4<f32>,
    original_size: vec2<f32>,
    split_origin: vec2<f32>,
    split_normal: vec2<f32>,
    split_offset: vec2<f32>,
    peak: vec4<f32>,
    peak_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var original_tex: texture_2d<f32>;

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

// Sharpness is read as the second difference of the brightness across each axis, not the first. A
// gradient - a soft shoulder of light on a cheek, the sky shading off - has a large first difference
// everywhere and is not sharp at all; only detail that turns, rather than merely leans, raises the
// second. That is what separates an edge in focus from a bright slope out of it.
//
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
fn peak_strength(uv: vec2<f32>) -> f32 {
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
    let detail = min(fine, 2.0 * coarse);
    // how little detail counts as sharp: the sensitivity runs from taking only the plainest edges
    // (0.12) to taking almost anything (0.008), evenly in ratio, so that each step of the slider does
    // as much as the last
    let bar = 0.12 * pow(0.008 / 0.12, clamp(transform.peak.y, 0.0, 1.0));
    return smoothstep(bar, bar * 2.5, detail);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let is_original = dot(in.pos.xy - transform.split_origin, transform.split_normal) < 0.0;
    let uv = select(in.uv, in.uv + transform.split_offset / transform.rect.zw, is_original);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return transform.bg_secondary;
    }

    if (is_original) {
        return sample_layer(original_tex, uv, transform.original_size, transform.original_size);
    }
    let colour = sample_layer(tex, uv, transform.image_size, transform.texture_size);
    if (transform.peak.x <= 0.0) {
        return colour;
    }
    // over the edited side only: the point of it is to judge the picture being worked on
    let marked = mix(colour.rgb, transform.peak_color.rgb, peak_strength(uv) * transform.peak.x);
    return vec4<f32>(marked, colour.a);
}
