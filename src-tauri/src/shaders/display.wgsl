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
// Sharpness is read as the second difference of the brightness across each axis, not the first. A
// gradient - a soft shoulder of light on a cheek, the sky shading off - has a large first difference
// everywhere and is not sharp at all; only detail that turns, rather than merely leans, raises the
// second. That is what separates an edge in focus from a bright slope out of it.
fn peak_strength(uv: vec2<f32>) -> f32 {
    // a step of one texel of the picture, whatever the zoom: the uv runs over the image, and the
    // sample converts to the texture's own scale
    let step = 1.0 / max(transform.image_size, vec2<f32>(1.0, 1.0));
    let size = transform.image_size;
    let tex_size = transform.texture_size;
    let here = peak_luma(sample_layer(tex, uv, size, tex_size).rgb);
    let right = peak_luma(sample_layer(tex, uv + vec2<f32>(step.x, 0.0), size, tex_size).rgb);
    let left = peak_luma(sample_layer(tex, uv - vec2<f32>(step.x, 0.0), size, tex_size).rgb);
    let below = peak_luma(sample_layer(tex, uv + vec2<f32>(0.0, step.y), size, tex_size).rgb);
    let above = peak_luma(sample_layer(tex, uv - vec2<f32>(0.0, step.y), size, tex_size).rgb);
    let detail = abs(right + left - 2.0 * here) + abs(below + above - 2.0 * here);
    // how little detail counts as sharp: the sensitivity runs from taking only the plainest edges to
    // taking almost anything
    let bar = mix(0.14, 0.004, clamp(transform.peak.y, 0.0, 1.0));
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
