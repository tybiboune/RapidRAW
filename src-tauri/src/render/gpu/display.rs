use crate::app::state::{AppState, PreviewTarget};
use crate::render::gpu::context::current_gpu_context;
use std::num::NonZero;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::Manager;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DisplayTransform {
    pub rect: [f32; 4],
    pub clip: [f32; 4],
    pub window: [f32; 2],
    pub image_size: [f32; 2],
    pub texture_size: [f32; 2],
    pub pixelated: f32,
    pub _pad: f32,
    pub bg_primary: [f32; 4],
    pub bg_secondary: [f32; 4],
    pub original_size: [f32; 2],
    pub split_origin: [f32; 2],
    pub split_normal: [f32; 2],
    pub split_offset: [f32; 2],
    // focus peaking: how strongly it is drawn, how little detail it takes to count, and in what colour.
    // Both are vec4 and sit on the end, where the sixteen-byte alignment is already met.
    pub peak: [f32; 4],
    pub peak_color: [f32; 4],
    // the focus map: x is 1 while it belongs to the picture shown, y its number of levels; and the rows of
    // the matrix taking the picture's uv to the map's (see `Frame::uv_to_source`)
    pub peak_map: [f32; 4],
    pub peak_u: [f32; 4],
    pub peak_v: [f32; 4],
}

/// A photo's focus map on the GPU (see `render::focus_map`), with the photo it belongs to.
pub struct FocusMapTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    content: u64,
    levels: u32,
    uv_to_map: [[f32; 4]; 2],
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DisplayLayer {
    Edited,
    Original,
}

impl Default for DisplayTransform {
    fn default() -> Self {
        Self {
            rect: [0.0, 0.0, 100.0, 100.0],
            clip: [0.0, 0.0, 10000.0, 10000.0],
            window: [1280.0, 720.0],
            image_size: [100.0, 100.0],
            texture_size: [100.0, 100.0],
            pixelated: 0.0,
            _pad: 0.0,
            bg_primary: [24.0 / 255.0, 24.0 / 255.0, 24.0 / 255.0, 1.0],
            bg_secondary: [35.0 / 255.0, 35.0 / 255.0, 35.0 / 255.0, 1.0],
            original_size: [100.0, 100.0],
            split_origin: [0.0, 0.0],
            split_normal: [0.0, 0.0],
            split_offset: [0.0, 0.0],
            peak: [0.0, 0.5, 0.0, 0.0],
            peak_color: [1.0, 0.16, 0.36, 1.0],
            peak_map: [0.0; 4],
            peak_u: [0.0; 4],
            peak_v: [0.0; 4],
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
pub struct DisplayView {
    pub zoom: f32,
    pub center: [f32; 2],
}

impl Default for DisplayView {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            center: [0.5, 0.5],
        }
    }
}

pub struct DisplayFrame {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    capacity: [u32; 2],
}

#[derive(Clone, Copy)]
pub struct DisplayOutput {
    pub target: PreviewTarget,
    pub layer: DisplayLayer,
    pub to_viewer: bool,
    pub content: u64,
}

pub fn display_content_key(path: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}

pub fn align_texture_dimension(value: u32) -> u32 {
    (value + 255) & !255
}

pub struct WgpuDisplay {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    pub transform_buffer: wgpu::Buffer,
    pub latest_transform: DisplayTransform,
    pub current_bind_group: Option<wgpu::BindGroup>,
    pub edited_view: Option<wgpu::TextureView>,
    pub original_texture: Option<(wgpu::Texture, wgpu::TextureView)>,
    pub focus_map: Option<FocusMapTexture>,
    // bound in the focus map's place while there is none; never read then
    pub blank_map: (wgpu::Texture, wgpu::TextureView),
    pub standalone: bool,
    pub view: DisplayView,
    pub frame: Option<DisplayFrame>,
    pub content_key: Option<u64>,
    pub rect_key: Option<u64>,
    pub pending_rect: Option<(u64, [f32; 4])>,
    #[cfg(target_os = "linux")]
    pub wayland: Option<crate::platform::wayland_subsurface::WaylandSubsurface>,
}

impl WgpuDisplay {
    pub fn original_texture(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> &wgpu::Texture {
        if self
            .original_texture
            .as_ref()
            .is_none_or(|(t, _)| t.width() != width || t.height() != height)
        {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Original Display Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.original_texture = Some((texture, view));
        }
        &self.original_texture.as_ref().unwrap().0
    }

    pub fn rebind(&mut self, device: &wgpu::Device) {
        let Some(edited_view) = &self.edited_view else {
            return;
        };
        let original_view = self
            .original_texture
            .as_ref()
            .map_or(edited_view, |(_, view)| view);
        let map_view = self
            .focus_map
            .as_ref()
            .map_or(&self.blank_map.1, |map| &map.view);
        self.current_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(edited_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(original_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(map_view),
                },
            ],
            label: None,
        }));
    }

    /// Puts a photo's focus map on the GPU for focus peaking to draw from, with the matrix from the shown
    /// picture's uv to the map's. Levels too large for the GPU are left out; the map's uv stays the same.
    pub fn set_focus_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        content: u64,
        map: &crate::render::focus_map::FocusMap,
        uv_to_map: [[f64; 3]; 2],
    ) {
        let max = device.limits().max_texture_dimension_2d;
        let levels: Vec<_> = map
            .levels
            .iter()
            .skip_while(|level| level.width > max || level.height > max)
            .collect();
        let Some(first) = levels.first() else {
            return;
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Focus Map"),
            size: wgpu::Extent3d {
                width: first.width,
                height: first.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (mip, level) in levels.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &level.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(level.width),
                    rows_per_image: Some(level.height),
                },
                wgpu::Extent3d {
                    width: level.width,
                    height: level.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let row = |r: [f64; 3]| [r[0] as f32, r[1] as f32, r[2] as f32, 0.0];
        self.focus_map = Some(FocusMapTexture {
            view: texture.create_view(&Default::default()),
            _texture: texture,
            content,
            levels: levels.len() as u32,
            uv_to_map: [row(uv_to_map[0]), row(uv_to_map[1])],
        });
        self.rebind(device);
        self.flush(device, queue);
    }

    pub fn show_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        image_size: [f32; 2],
        texture_size: [f32; 2],
        content: Option<u64>,
    ) {
        self.latest_transform.image_size = image_size;
        self.latest_transform.texture_size = texture_size;
        self.edited_view = Some(view.clone());
        if let Some(key) = content {
            self.content_key = Some(key);
            if let Some((pending_key, rect)) = self.pending_rect
                && pending_key == key
            {
                self.latest_transform.rect = rect;
                self.rect_key = Some(key);
                self.pending_rect = None;
            }
        }
        self.place_image();
        self.rebind(device);
        self.flush(device, queue);
    }

    pub fn show_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Texture,
        size: [u32; 2],
        max_dimension: u32,
    ) {
        let area = |dims: [u32; 2]| dims[0] as u64 * dims[1] as u64;
        let reusable = self.frame.as_ref().is_some_and(|frame| {
            frame.capacity[0] >= size[0]
                && frame.capacity[1] >= size[1]
                && area(frame.capacity) <= 4 * area(size)
        });
        if !reusable {
            let capacity = size.map(|dim| align_texture_dimension(dim).min(max_dimension));
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Viewer Frame Texture"),
                size: wgpu::Extent3d {
                    width: capacity[0],
                    height: capacity[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.frame = Some(DisplayFrame {
                texture,
                view,
                capacity,
            });
        }
        let Some(frame) = &self.frame else {
            return;
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Viewer Frame Copy"),
        });
        encoder.copy_texture_to_texture(
            source.as_image_copy(),
            frame.texture.as_image_copy(),
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let view = frame.view.clone();
        let capacity = frame.capacity;
        self.show_texture(
            device,
            queue,
            &view,
            size.map(|dim| dim as f32),
            capacity.map(|dim| dim as f32),
            None,
        );
    }

    pub fn set_view(&mut self, view: DisplayView) {
        self.view = view;
        self.place_image();
    }

    pub fn set_background(&mut self, color: [f32; 4]) {
        self.latest_transform.bg_primary = color;
        self.latest_transform.bg_secondary = color;
    }

    pub fn resize(&mut self, device: &wgpu::Device, size: tauri::PhysicalSize<u32>) {
        let width = size.width.max(1);
        let height = size.height.max(1);
        if width != self.config.width || height != self.config.height {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(device, &self.config);
        }
        self.place_image();
    }

    pub fn place_editor_rect(&mut self, content: Option<u64>, rect: [f32; 4]) -> bool {
        match content {
            Some(key) if self.content_key != Some(key) => {
                self.pending_rect = Some((key, rect));
                false
            }
            _ => {
                self.latest_transform.rect = rect;
                self.rect_key = self.content_key;
                self.pending_rect = None;
                true
            }
        }
    }

    pub fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        // the focus map is drawn from only while it is the shown photo's
        match &self.focus_map {
            Some(map) if self.content_key == Some(map.content) => {
                self.latest_transform.peak_map = [1.0, map.levels as f32, 0.0, 0.0];
                self.latest_transform.peak_u = map.uv_to_map[0];
                self.latest_transform.peak_v = map.uv_to_map[1];
            }
            _ => self.latest_transform.peak_map = [0.0; 4],
        }
        queue.write_buffer(
            &self.transform_buffer,
            0,
            bytemuck::bytes_of(&self.latest_transform),
        );
        self.render(device, queue)
    }

    fn place_image(&mut self) {
        if !self.standalone {
            return;
        }
        let surface_w = self.config.width as f32;
        let surface_h = self.config.height as f32;
        let [image_w, image_h] = self.latest_transform.image_size;
        let scale = if image_w > 0.0 && image_h > 0.0 {
            (surface_w / image_w).min(surface_h / image_h) * self.view.zoom
        } else {
            0.0
        };
        let placed_w = image_w * scale;
        let placed_h = image_h * scale;
        self.latest_transform.rect = [
            (surface_w / 2.0 - self.view.center[0] * placed_w).round(),
            (surface_h / 2.0 - self.view.center[1] * placed_h).round(),
            placed_w,
            placed_h,
        ];
        self.latest_transform.clip = [0.0, 0.0, surface_w, surface_h];
        self.latest_transform.window = [surface_w, surface_h];
    }

    fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        if self.current_bind_group.is_none() && !self.standalone {
            return false;
        }
        if !self.standalone && self.rect_key != self.content_key {
            return false;
        }
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(device, &self.config);
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(tex)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
                    _ => return false,
                }
            }
            _ => return false,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.latest_transform.bg_primary[0] as f64,
                            g: self.latest_transform.bg_primary[1] as f64,
                            b: self.latest_transform.bg_primary[2] as f64,
                            a: self.latest_transform.bg_primary[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: NonZero::new(0),
            });
            if let Some(bind_group) = &self.current_bind_group {
                let clip_x1 = self.latest_transform.clip[0].max(0.0);
                let clip_y1 = self.latest_transform.clip[1].max(0.0);
                let clip_x2 =
                    (self.latest_transform.clip[0] + self.latest_transform.clip[2]).max(0.0);
                let clip_y2 =
                    (self.latest_transform.clip[1] + self.latest_transform.clip[3]).max(0.0);

                let final_clip_x = clip_x1.floor() as u32;
                let final_clip_y = clip_y1.floor() as u32;
                let final_clip_w = (clip_x2.ceil() as u32).saturating_sub(final_clip_x);
                let final_clip_h = (clip_y2.ceil() as u32).saturating_sub(final_clip_y);

                let max_x = self.config.width;
                let max_y = self.config.height;

                if final_clip_x < max_x && final_clip_y < max_y {
                    let clamped_width = final_clip_w.min(max_x - final_clip_x);
                    let clamped_height = final_clip_h.min(max_y - final_clip_y);

                    if clamped_width > 0 && clamped_height > 0 {
                        rpass.set_scissor_rect(
                            final_clip_x,
                            final_clip_y,
                            clamped_width,
                            clamped_height,
                        );

                        rpass.set_pipeline(&self.pipeline);
                        rpass.set_bind_group(0, bind_group, &[]);
                        rpass.draw(0..4, 0..1);
                    }
                }
            }
        }
        queue.submit(Some(encoder.finish()));
        output.present();
        true
    }
}

#[cfg(not(target_os = "android"))]
pub fn build_wgpu_display(
    surface: wgpu::Surface<'static>,
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    size: tauri::PhysicalSize<u32>,
    standalone: bool,
    #[cfg_attr(not(target_os = "linux"), allow(unused_variables))] platform: PlatformSurface,
) -> Result<WgpuDisplay, String> {
    let swapchain_caps = surface.get_capabilities(adapter);
    let usable = |f: &wgpu::TextureFormat| device.features().contains(f.required_features());
    #[cfg(target_os = "linux")]
    let preferred = [
        wgpu::TextureFormat::Rgb10a2Unorm,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ]
    .into_iter()
    .find(|f| swapchain_caps.formats.contains(f) && usable(f));
    #[cfg(not(target_os = "linux"))]
    let preferred: Option<wgpu::TextureFormat> = None;
    let swapchain_format = preferred
        .or_else(|| {
            swapchain_caps
                .formats
                .iter()
                .copied()
                .find(|f| !f.is_srgb() && usable(f))
        })
        .or_else(|| swapchain_caps.formats.iter().copied().find(usable))
        .ok_or("Surface is not supported by the GPU adapter")?;

    let alpha_mode = if cfg!(target_os = "windows")
        && swapchain_caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::Opaque)
    {
        wgpu::CompositeAlphaMode::Opaque
    } else if swapchain_caps
        .alpha_modes
        .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
    {
        wgpu::CompositeAlphaMode::PreMultiplied
    } else if swapchain_caps
        .alpha_modes
        .contains(&wgpu::CompositeAlphaMode::PostMultiplied)
    {
        wgpu::CompositeAlphaMode::PostMultiplied
    } else {
        swapchain_caps.alpha_modes[0]
    };

    let present_mode = if (standalone || cfg!(target_os = "linux"))
        && swapchain_caps
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
    {
        wgpu::PresentMode::Mailbox
    } else {
        wgpu::PresentMode::Fifo
    };

    let config = wgpu::SurfaceConfiguration {
        width: size.width.max(1),
        height: size.height.max(1),
        format: swapchain_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        present_mode,
        alpha_mode,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(device, &config);
    log::info!(
        "WGPU display surface: {:?}, {:?}",
        swapchain_format,
        present_mode
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Display Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/display.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Display BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                count: None,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                count: None,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                count: None,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                count: None,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            },
        ],
    });

    let blank_map = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("No Focus Map"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let blank_map_view = blank_map.create_view(&Default::default());

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Display Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Display Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: swapchain_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: NonZero::new(0),
        cache: None,
    });

    let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Transform Buffer"),
        size: std::mem::size_of::<DisplayTransform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Display Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    Ok(WgpuDisplay {
        surface,
        config,
        pipeline,
        bind_group_layout,
        transform_buffer,
        latest_transform: DisplayTransform::default(),
        sampler,
        current_bind_group: None,
        edited_view: None,
        original_texture: None,
        focus_map: None,
        blank_map: (blank_map, blank_map_view),
        standalone,
        view: DisplayView::default(),
        frame: None,
        content_key: None,
        rect_key: None,
        pending_rect: None,
        #[cfg(target_os = "linux")]
        wayland: platform,
    })
}

#[cfg(target_os = "linux")]
pub type PlatformSurface = Option<crate::platform::wayland_subsurface::WaylandSubsurface>;

#[cfg(not(target_os = "linux"))]
pub type PlatformSurface = ();

#[cfg(target_os = "linux")]
pub fn surface_for_window(
    instance: &wgpu::Instance,
    window: &tauri::WebviewWindow,
) -> Result<(wgpu::Surface<'static>, PlatformSurface), String> {
    use gtk::prelude::WidgetExt;

    if !gtk::is_initialized_main_thread() {
        return Err("Wayland subsurface must be created on the main thread".to_string());
    }

    if let Ok(gtk_window) = window.gtk_window() {
        gtk_window.realize();
    }
    let wayland = crate::platform::wayland_subsurface::create(window)?;
    let surface = wayland.create_surface(instance)?;
    Ok((surface, Some(wayland)))
}

#[cfg(not(any(target_os = "android", target_os = "linux")))]
pub fn surface_for_window(
    instance: &wgpu::Instance,
    window: &tauri::WebviewWindow,
) -> Result<(wgpu::Surface<'static>, PlatformSurface), String> {
    let surface = instance
        .create_surface(window.clone())
        .map_err(|e| e.to_string())?;
    Ok((surface, ()))
}

#[cfg(not(target_os = "android"))]
pub fn create_window_surface(
    instance: &wgpu::Instance,
    window: &tauri::WebviewWindow,
) -> Result<(wgpu::Surface<'static>, PlatformSurface), String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let instance = instance.clone();
    let target = window.clone();
    window
        .run_on_main_thread(move || {
            let _ = tx.send(surface_for_window(&instance, &target));
        })
        .map_err(|e| e.to_string())?;
    rx.recv().map_err(|e| e.to_string())?
}

pub fn resize_display(window: &tauri::Window, size: tauri::PhysicalSize<u32>) {
    update_display_size(window, size, false);
}

pub fn redraw_display(window: &tauri::Window, size: tauri::PhysicalSize<u32>) {
    update_display_size(window, size, true);
}

fn update_display_size(window: &tauri::Window, size: tauri::PhysicalSize<u32>, wait: bool) {
    let Some(ctx) = current_gpu_context(&window.state::<AppState>()) else {
        return;
    };
    let Some(display) = ctx.display_for_label(window.label()) else {
        return;
    };
    let mut lock = if wait {
        display.lock().unwrap_or_else(|e| e.into_inner())
    } else {
        match display.try_lock() {
            Ok(lock) => lock,
            Err(_) => return,
        }
    };
    if let Some(display) = lock.as_mut() {
        display.resize(&ctx.device, size);
        display.flush(&ctx.device, &ctx.queue);
    }
}

pub static WGPU_DISPLAY_FAILED: AtomicBool = AtomicBool::new(false);

pub fn wgpu_display_failed() -> bool {
    WGPU_DISPLAY_FAILED.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::DisplayTransform;

    // the shader's `Transform` block is 208 bytes (the last five vec4 sit on 16-byte boundaries, from 128
    // on), and the card reads exactly what is written from `DisplayTransform`
    #[test]
    fn display_transform_matches_the_shader_block() {
        assert_eq!(std::mem::size_of::<DisplayTransform>(), 208);
        assert_eq!(std::mem::offset_of!(DisplayTransform, peak), 128);
        assert_eq!(std::mem::offset_of!(DisplayTransform, peak_color), 144);
        assert_eq!(std::mem::offset_of!(DisplayTransform, peak_map), 160);
        assert_eq!(std::mem::offset_of!(DisplayTransform, peak_u), 176);
        assert_eq!(std::mem::offset_of!(DisplayTransform, peak_v), 192);
    }

    // WGSL compiles when it first runs, so run it: build the display shader on a real device and fail on
    // any validation error
    #[test]
    fn display_shader_compiles() {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no wgpu adapter (install a Vulkan driver, e.g. mesa lavapipe)");
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&Default::default())).expect("device");
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/display.wgsl").into()),
        });
        let error = pollster::block_on(scope.pop());
        assert!(error.is_none(), "display.wgsl failed to compile: {error:?}");
    }
}
