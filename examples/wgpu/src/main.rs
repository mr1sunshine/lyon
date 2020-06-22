use lyon::extra::rust_logo::build_logo_path;
use lyon::math::*;
use lyon::path::builder::*;
use lyon::path::Path;
use lyon::path::Polygon;
use lyon::tessellation;
use lyon::tessellation::geometry_builder::*;
use lyon::tessellation::{FillOptions, FillTessellator};
use lyon::tessellation::{StrokeOptions, StrokeTessellator};

use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

use futures::executor::block_on;
use std::ops::Rem;

#[macro_use]
use log;

const PRIM_BUFFER_LEN: usize = 2;

#[repr(C)]
#[derive(Copy, Clone)]
struct Globals {
    resolution: [f32; 2],

    scroll_offset: [f32; 2],

    zoom: f32,
}

unsafe impl bytemuck::Pod for Globals {}
unsafe impl bytemuck::Zeroable for Globals {}

#[repr(C)]
#[derive(Copy, Clone)]
struct GpuVertex {
    position: [f32; 2],

    normal: [f32; 2],

    prim_id: i32,
}
unsafe impl bytemuck::Pod for GpuVertex {}
unsafe impl bytemuck::Zeroable for GpuVertex {}

#[repr(C)]
#[derive(Copy, Clone)]
struct Primitive {
    color: [f32; 4],

    translate: [f32; 2],

    z_index: i32,
    width: f32,
}
unsafe impl bytemuck::Pod for Primitive {}
unsafe impl bytemuck::Zeroable for Primitive {}

#[repr(C)]
#[derive(Copy, Clone)]
struct BgPoint {
    point: [f32; 2],
}
unsafe impl bytemuck::Pod for BgPoint {}
unsafe impl bytemuck::Zeroable for BgPoint {}

const DEFAULT_WINDOW_WIDTH: f32 = 800.0;
const DEFAULT_WINDOW_HEIGHT: f32 = 800.0;

/// Creates a texture that uses MSAA and fits a given swap chain
fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
    sample_count: u32,
) -> wgpu::TextureView {
    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
        label: Some("Multisampled frame descriptor"),
        size: wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth: 1,
        },
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: sc_desc.format,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
    };

    device
        .create_texture(multisampled_frame_descriptor)
        .create_default_view()
}

fn main() {
    env_logger::init();
    println!("== wgpu example ==");
    println!("Controls:");
    println!("  Arrow keys: scrolling");
    println!("  PgUp/PgDown: zoom in/out");
    println!("  w: toggle wireframe mode");
    println!("  b: toggle drawing the background");
    println!("  a/z: increase/decrease the stroke width");

    // Number of samples for anti-aliasing
    // Set to 1 to disable
    let sample_count = 4;

    let num_instances: u32 = PRIM_BUFFER_LEN as u32 - 1;
    let tolerance = 0.02;

    let stroke_prim_id = 0;
    let fill_prim_id = 1;

    let mut geometry: VertexBuffers<GpuVertex, u16> = VertexBuffers::new();

    let mut fill_tess = FillTessellator::new();
    let mut stroke_tess = StrokeTessellator::new();

    // Build a Path for the rust logo.
    let mut builder = Path::builder().with_svg();
    build_polygons(&mut builder);
    let path = builder.build();

    let fill_count = fill_tess
        .tessellate_path(
            &path,
            &FillOptions::tolerance(tolerance).with_fill_rule(tessellation::FillRule::NonZero),
            &mut BuffersBuilder::new(&mut geometry, WithId(fill_prim_id as i32)),
        )
        .unwrap();

    stroke_tess
        .tessellate_path(
            &path,
            &StrokeOptions::tolerance(tolerance),
            &mut BuffersBuilder::new(&mut geometry, WithId(stroke_prim_id as i32)),
        )
        .unwrap();

    let fill_range = 0..fill_count.indices;
    let stroke_range = fill_range.end..(geometry.indices.len() as u32);
    let mut bg_geometry: VertexBuffers<BgPoint, u16> = VertexBuffers::new();

    fill_tess
        .tessellate_rectangle(
            &Rect::new(point(-1.0, -1.0), size(2.0, 2.0)),
            &FillOptions::DEFAULT,
            &mut BuffersBuilder::new(&mut bg_geometry, Custom),
        )
        .unwrap();

    let mut cpu_primitives = Vec::with_capacity(PRIM_BUFFER_LEN);
    for _ in 0..PRIM_BUFFER_LEN {
        cpu_primitives.push(Primitive {
            color: [1.0, 0.0, 0.0, 1.0],

            z_index: 0,
            width: 0.0,
            translate: [0.0, 0.0],
        });
    }

    // Stroke primitive
    cpu_primitives[stroke_prim_id] = Primitive {
        color: [0.0, 0.0, 0.0, 1.0],

        z_index: num_instances as i32 + 2,
        width: 1.0,
        translate: [0.0, 0.0],
    };
    // Main fill primitive
    cpu_primitives[fill_prim_id] = Primitive {
        color: [1.0, 1.0, 1.0, 1.0],

        z_index: num_instances as i32 + 1,
        width: 0.0,
        translate: [0.0, 0.0],
    };
    // Instance primitives
    for idx in (fill_prim_id + 1)..(fill_prim_id + num_instances as usize) {
        cpu_primitives[idx].z_index = (idx as u32 + 1) as i32;
        cpu_primitives[idx].color = [
            (0.1 * idx as f32).rem(1.0),
            (0.5 * idx as f32).rem(1.0),
            (0.9 * idx as f32).rem(1.0),
            1.0,
        ];
    }

    let mut scene = SceneParams {
        target_zoom: 5.0,
        zoom: 5.0,
        target_scroll: vector(70.0, 70.0),
        scroll: vector(70.0, 70.0),
        show_points: false,
        show_wireframe: false,
        stroke_width: 1.0,
        target_stroke_width: 1.0,
        draw_background: true,
        cursor_position: (0.0, 0.0),
        window_size: LogicalSize::new(DEFAULT_WINDOW_WIDTH as f64, DEFAULT_WINDOW_HEIGHT as f64),
        size_changed: true,
    };

    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let surface = wgpu::Surface::create(&window);

    let adapter = block_on(wgpu::Adapter::request(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
        },
        wgpu::BackendBit::PRIMARY,
    ))
    .unwrap();

    let (device, mut queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    }));

    let vbo = device.create_buffer_with_data(
        bytemuck::cast_slice(&geometry.vertices),
        wgpu::BufferUsage::VERTEX,
    );

    let ibo = device.create_buffer_with_data(
        bytemuck::cast_slice(&geometry.indices),
        wgpu::BufferUsage::INDEX,
    );

    let bg_vbo = device.create_buffer_with_data(
        bytemuck::cast_slice(&bg_geometry.vertices),
        wgpu::BufferUsage::VERTEX,
    );

    let bg_ibo = device.create_buffer_with_data(
        bytemuck::cast_slice(&bg_geometry.indices),
        wgpu::BufferUsage::INDEX,
    );

    let prim_buffer_byte_size = (PRIM_BUFFER_LEN * std::mem::size_of::<Primitive>()) as u64;
    let globals_buffer_byte_size = std::mem::size_of::<Globals>() as u64;

    let prims_ubo = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Prims ubo"),
        size: prim_buffer_byte_size,
        usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    });

    let globals_ubo = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Globals ubo"),
        size: globals_buffer_byte_size,
        usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    });

    let vs_bytes = include_bytes!("./../shaders/geometry.vert.spv");
    let fs_bytes = include_bytes!("./../shaders/geometry.frag.spv");
    let vs_spv = wgpu::read_spirv(std::io::Cursor::new(&vs_bytes[..])).unwrap();
    let fs_spv = wgpu::read_spirv(std::io::Cursor::new(&fs_bytes[..])).unwrap();
    let bg_vs_bytes = include_bytes!("./../shaders/background.vert.spv");
    let bg_fs_bytes = include_bytes!("./../shaders/background.frag.spv");
    let bg_vs_spv = wgpu::read_spirv(std::io::Cursor::new(&bg_vs_bytes[..])).unwrap();
    let bg_fs_spv = wgpu::read_spirv(std::io::Cursor::new(&bg_fs_bytes[..])).unwrap();
    let vs_module = device.create_shader_module(&vs_spv);
    let fs_module = device.create_shader_module(&fs_spv);
    let bg_vs_module = device.create_shader_module(&bg_vs_spv);
    let bg_fs_module = device.create_shader_module(&bg_fs_spv);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind group layout"),
        bindings: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind group"),
        layout: &bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &globals_ubo,
                    range: 0..globals_buffer_byte_size,
                },
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &prims_ubo,
                    range: 0..prim_buffer_byte_size,
                },
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let depth_stencil_state = Some(wgpu::DepthStencilStateDescriptor {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Greater,
        stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
        stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
        stencil_read_mask: 0,
        stencil_write_mask: 0,
    });

    let mut render_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8Unorm,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],

        depth_stencil_state: depth_stencil_state.clone(),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<GpuVertex>() as u64,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttributeDescriptor {
                        offset: 0,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 0,
                    },
                    wgpu::VertexAttributeDescriptor {
                        offset: 8,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 1,
                    },
                    wgpu::VertexAttributeDescriptor {
                        offset: 16,
                        format: wgpu::VertexFormat::Int,
                        shader_location: 2,
                    },
                ],
            }],
        },
        sample_count: sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    };

    let render_pipeline = device.create_render_pipeline(&render_pipeline_descriptor);

    // TODO: this isn't what we want: we'd need the equivalent of VK_POLYGON_MODE_LINE,
    // but it doesn't seem to be exposed by wgpu?
    render_pipeline_descriptor.primitive_topology = wgpu::PrimitiveTopology::LineList;
    let wireframe_render_pipeline = device.create_render_pipeline(&render_pipeline_descriptor);

    let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &bg_vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &bg_fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8Unorm,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],

        depth_stencil_state: depth_stencil_state.clone(),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<Point>() as u64,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    format: wgpu::VertexFormat::Float2,
                    shader_location: 0,
                }],
            }],
        },
        sample_count: sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    });

    let size = window.inner_size().to_physical(window.hidpi_factor());

    let mut swap_chain_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: size.width.round() as u32,
        height: size.height.round() as u32,
        present_mode: wgpu::PresentMode::Fifo,
    };

    let mut multisampled_render_target = None;

    let window_surface = wgpu::Surface::create(&window);
    let mut swap_chain = device.create_swap_chain(&window_surface, &swap_chain_desc);

    let mut depth_texture_view = None;

    let mut frame_count: f32 = 0.0;
    event_loop.run(move |event, _, control_flow| {
        if update_inputs(event, control_flow, &mut scene) {
            // keep polling inputs.
            return;
        }

        if scene.size_changed {
            scene.size_changed = false;
            let physical = scene.window_size.to_physical(window.hidpi_factor());
            swap_chain_desc.width = physical.width.round() as u32;
            swap_chain_desc.height = physical.height.round() as u32;
            swap_chain = device.create_swap_chain(&window_surface, &swap_chain_desc);

            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth texture"),
                size: wgpu::Extent3d {
                    width: swap_chain_desc.width,
                    height: swap_chain_desc.height,
                    depth: 1,
                },
                array_layer_count: 1,
                mip_level_count: 1,
                sample_count: sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            });

            depth_texture_view = Some(depth_texture.create_default_view());

            multisampled_render_target = if sample_count > 1 {
                Some(create_multisampled_framebuffer(
                    &device,
                    &swap_chain_desc,
                    sample_count,
                ))
            } else {
                None
            };
        }

        let frame = swap_chain.get_next_texture().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder"),
        });

        cpu_primitives[stroke_prim_id as usize].width = scene.stroke_width;
        cpu_primitives[stroke_prim_id as usize].color = [
            (frame_count * 0.008 - 1.6).sin() * 0.1 + 0.1,
            (frame_count * 0.005 - 1.6).sin() * 0.1 + 0.1,
            (frame_count * 0.01 - 1.6).sin() * 0.1 + 0.1,
            1.0,
        ];

        for idx in 2..(num_instances + 1) {
            cpu_primitives[idx as usize].translate = [
                (frame_count * 0.001 * idx as f32).sin() * (100.0 + idx as f32 * 10.0),
                (frame_count * 0.002 * idx as f32).sin() * (100.0 + idx as f32 * 10.0),
            ];
        }

        let globals_transfer_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&[Globals {
                resolution: [
                    scene.window_size.width as f32,
                    scene.window_size.height as f32,
                ],

                zoom: scene.zoom,
                scroll_offset: scene.scroll.to_array(),
            }]),
            wgpu::BufferUsage::COPY_SRC,
        );

        let prim_transfer_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&cpu_primitives),
            wgpu::BufferUsage::COPY_SRC,
        );

        encoder.copy_buffer_to_buffer(
            &globals_transfer_buffer,
            0,
            &globals_ubo,
            0,
            globals_buffer_byte_size,
        );

        encoder.copy_buffer_to_buffer(
            &prim_transfer_buffer,
            0,
            &prims_ubo,
            0,
            prim_buffer_byte_size,
        );

        {
            // A resolve target is only supported if the attachment actually uses anti-aliasing
            // So if sample_count == 1 then we must render directly to the swapchain's buffer
            let color_attachment = if let Some(msaa_target) = &multisampled_render_target {
                wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: msaa_target,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::WHITE,
                    resolve_target: Some(&frame.view),
                }
            } else {
                wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::WHITE,
                    resolve_target: None,
                }
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[color_attachment],

                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: depth_texture_view.as_ref().unwrap(),
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 0.0,
                    clear_stencil: 0,
                }),
            });

            if scene.show_wireframe {
                pass.set_pipeline(&wireframe_render_pipeline);
            } else {
                pass.set_pipeline(&render_pipeline);
            }
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_index_buffer(&ibo, 0, 0);
            pass.set_vertex_buffer(0, &vbo, 0, 0);

            pass.draw_indexed(fill_range.clone(), 0, 0..(num_instances as u32));
            pass.draw_indexed(stroke_range.clone(), 0, 0..1);

            if scene.draw_background {
                pass.set_pipeline(&bg_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_index_buffer(&bg_ibo, 0, 0);
                pass.set_vertex_buffer(0, &bg_vbo, 0, 0);

                pass.draw_indexed(0..6, 0, 0..1);
            }
        }

        queue.submit(&[encoder.finish()]);

        frame_count += 1.0;
    });
}

/// This vertex constructor forwards the positions and normals provided by the
/// tessellators and add a shape id.
pub struct WithId(pub i32);

impl FillVertexConstructor<GpuVertex> for WithId {
    fn new_vertex(&mut self, vertex: tessellation::FillVertex) -> GpuVertex {
        GpuVertex {
            position: vertex.position().to_array(),
            normal: [0.0, 0.0],

            prim_id: self.0,
        }
    }
}

impl StrokeVertexConstructor<GpuVertex> for WithId {
    fn new_vertex(&mut self, vertex: tessellation::StrokeVertex) -> GpuVertex {
        GpuVertex {
            position: vertex.position_on_path().to_array(),
            normal: vertex.normal().to_array(),
            prim_id: self.0,
        }
    }
}

pub struct Custom;

impl FillVertexConstructor<BgPoint> for Custom {
    fn new_vertex(&mut self, vertex: tessellation::FillVertex) -> BgPoint {
        BgPoint {
            point: vertex.position().to_array(),
        }
    }
}

struct SceneParams {
    target_zoom: f32,
    zoom: f32,
    target_scroll: Vector,
    scroll: Vector,
    show_points: bool,
    show_wireframe: bool,
    stroke_width: f32,
    target_stroke_width: f32,
    draw_background: bool,
    cursor_position: (f32, f32),
    window_size: LogicalSize,
    size_changed: bool,
}

fn update_inputs(
    event: Event<()>,
    control_flow: &mut ControlFlow,
    scene: &mut SceneParams,
) -> bool {
    match event {
        Event::EventsCleared => {
            return false;
        }
        Event::WindowEvent {
            event: WindowEvent::Destroyed,
            ..
        }
        | Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
            return false;
        }
        Event::WindowEvent {
            event: WindowEvent::CursorMoved { position, .. },
            ..
        } => {
            scene.cursor_position = (position.x as f32, position.y as f32);
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            scene.window_size = size;
            scene.size_changed = true
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(key),
                            ..
                        },
                    ..
                },
            ..
        } => match key {
            VirtualKeyCode::Escape => {
                *control_flow = ControlFlow::Exit;
                return false;
            }
            VirtualKeyCode::PageDown => {
                scene.target_zoom *= 0.8;
            }
            VirtualKeyCode::PageUp => {
                scene.target_zoom *= 1.25;
            }
            VirtualKeyCode::Left => {
                scene.target_scroll.x -= 50.0 / scene.target_zoom;
            }
            VirtualKeyCode::Right => {
                scene.target_scroll.x += 50.0 / scene.target_zoom;
            }
            VirtualKeyCode::Up => {
                scene.target_scroll.y -= 50.0 / scene.target_zoom;
            }
            VirtualKeyCode::Down => {
                scene.target_scroll.y += 50.0 / scene.target_zoom;
            }
            VirtualKeyCode::P => {
                scene.show_points = !scene.show_points;
            }
            VirtualKeyCode::W => {
                scene.show_wireframe = !scene.show_wireframe;
            }
            VirtualKeyCode::B => {
                scene.draw_background = !scene.draw_background;
            }
            VirtualKeyCode::A => {
                scene.target_stroke_width /= 0.8;
            }
            VirtualKeyCode::Z => {
                scene.target_stroke_width *= 0.8;
            }
            _key => {}
        },
        _evt => {
            //println!("{:?}", _evt);
        }
    }
    //println!(" -- zoom: {}, scroll: {:?}", scene.target_zoom, scene.target_scroll);

    scene.zoom += (scene.target_zoom - scene.zoom) / 3.0;
    scene.scroll = scene.scroll + (scene.target_scroll - scene.scroll) / 3.0;
    scene.stroke_width =
        scene.stroke_width + (scene.target_stroke_width - scene.stroke_width) / 5.0;

    *control_flow = ControlFlow::Poll;

    return true;
}

pub fn build_polygons<Builder: SvgPathBuilder>(path: &mut Builder) {
    let mut x_offset = 4096;
    let mut y_offset = 0;
    let polygon1 = [
        [4224, -128],
        [4224, 2046],
        [4214, 2040],
        [4214, 1994],
        [4198, 1974],
        [4154, 1946],
        [4120, 1910],
        [4096, 1900],
        [4066, 1870],
        [4050, 1868],
        [4012, 1846],
        [3940, 1844],
        [3930, 1836],
        [3882, 1828],
        [3874, 1860],
        [3884, 1866],
        [3890, 1896],
        [3868, 1912],
        [3858, 1910],
        [3852, 1892],
        [3830, 1882],
        [3830, 1860],
        [3850, 1862],
        [3856, 1850],
        [3856, 1844],
        [3838, 1832],
        [3826, 1832],
        [3798, 1866],
        [3744, 1860],
        [3738, 1852],
        [3692, 1856],
        [3666, 1866],
        [3636, 1846],
        [3642, 1804],
        [3622, 1776],
        [3572, 1758],
        [3540, 1760],
        [3480, 1778],
        [3466, 1776],
        [3444, 1738],
        [3432, 1736],
        [3414, 1718],
        [3396, 1716],
        [3396, 1708],
        [3416, 1706],
        [3406, 1682],
        [3382, 1670],
        [3340, 1670],
        [3318, 1654],
        [3286, 1644],
        [3226, 1638],
        [3206, 1628],
        [3210, 1650],
        [3196, 1658],
        [3176, 1656],
        [3168, 1670],
        [3170, 1680],
        [3186, 1678],
        [3178, 1696],
        [3184, 1728],
        [3176, 1732],
        [3168, 1732],
        [3162, 1720],
        [3120, 1728],
        [3088, 1720],
        [3068, 1736],
        [3052, 1738],
        [3030, 1722],
        [3018, 1702],
        [3008, 1718],
        [2996, 1772],
        [2982, 1782],
        [2954, 1756],
        [2938, 1722],
        [2942, 1710],
        [2930, 1708],
        [2930, 1694],
        [2946, 1686],
        [2946, 1668],
        [2936, 1656],
        [2942, 1648],
        [2944, 1614],
        [2908, 1576],
        [2892, 1576],
        [2878, 1584],
        [2876, 1578],
        [2858, 1578],
        [2836, 1558],
        [2826, 1558],
        [2806, 1574],
        [2812, 1598],
        [2808, 1618],
        [2788, 1616],
        [2788, 1624],
        [2782, 1626],
        [2772, 1620],
        [2750, 1622],
        [2736, 1606],
        [2710, 1610],
        [2696, 1600],
        [2702, 1574],
        [2672, 1572],
        [2656, 1564],
        [2624, 1562],
        [2612, 1570],
        [2596, 1570],
        [2584, 1578],
        [2582, 1588],
        [2582, 1562],
        [2574, 1548],
        [2566, 1560],
        [2548, 1560],
        [2532, 1542],
        [2504, 1536],
        [2492, 1554],
        [2494, 1562],
        [2520, 1564],
        [2502, 1580],
        [2464, 1594],
        [2454, 1604],
        [2416, 1606],
        [2416, 1598],
        [2430, 1594],
        [2444, 1570],
        [2462, 1566],
        [2476, 1542],
        [2498, 1528],
        [2502, 1510],
        [2514, 1498],
        [2534, 1482],
        [2546, 1480],
        [2552, 1466],
        [2572, 1452],
        [2588, 1426],
        [2582, 1396],
        [2592, 1388],
        [2592, 1372],
        [2578, 1342],
        [2566, 1346],
        [2562, 1306],
        [2542, 1304],
        [2530, 1290],
        [2460, 1290],
        [2454, 1310],
        [2426, 1312],
        [2444, 1280],
        [2444, 1268],
        [2432, 1260],
        [2384, 1254],
        [2414, 1228],
        [2414, 1218],
        [2408, 1208],
        [2368, 1190],
        [2358, 1200],
        [2348, 1200],
        [2308, 1248],
        [2296, 1276],
        [2304, 1292],
        [2300, 1308],
        [2284, 1318],
        [2250, 1318],
        [2266, 1348],
        [2256, 1340],
        [2244, 1340],
        [2220, 1362],
        [2208, 1356],
        [2198, 1364],
        [2188, 1362],
        [2188, 1350],
        [2204, 1336],
        [2176, 1334],
        [2170, 1348],
        [2156, 1346],
        [2120, 1356],
        [2114, 1366],
        [2124, 1370],
        [2124, 1376],
        [2024, 1414],
        [2004, 1442],
        [1982, 1440],
        [1984, 1454],
        [1972, 1454],
        [1966, 1468],
        [1956, 1470],
        [1954, 1488],
        [1938, 1492],
        [1942, 1500],
        [1960, 1498],
        [1956, 1510],
        [1972, 1518],
        [1970, 1546],
        [1946, 1554],
        [1944, 1560],
        [1864, 1566],
        [1832, 1576],
        [1838, 1658],
        [1844, 1666],
        [1870, 1674],
        [1880, 1700],
        [1894, 1704],
        [1898, 1714],
        [1854, 1712],
        [1834, 1684],
        [1808, 1666],
        [1782, 1664],
        [1782, 1654],
        [1766, 1648],
        [1750, 1670],
        [1766, 1674],
        [1762, 1686],
        [1774, 1688],
        [1778, 1700],
        [1764, 1704],
        [1752, 1690],
        [1742, 1690],
        [1730, 1708],
        [1742, 1732],
        [1762, 1746],
        [1780, 1750],
        [1782, 1766],
        [1762, 1752],
        [1724, 1748],
        [1714, 1740],
        [1720, 1648],
        [1712, 1630],
        [1702, 1630],
        [1706, 1682],
        [1672, 1708],
        [1662, 1734],
        [1688, 1786],
        [1688, 1806],
        [1678, 1818],
        [1674, 1842],
        [1678, 1888],
        [1682, 1894],
        [1706, 1894],
        [1718, 1884],
        [1728, 1884],
        [1746, 1892],
        [1766, 1908],
        [1774, 1946],
        [1762, 1954],
        [1760, 1964],
        [1760, 1930],
        [1748, 1918],
        [1742, 1900],
        [1698, 1912],
        [1692, 1942],
        [1700, 1954],
        [1700, 1980],
        [1682, 2000],
        [1680, 2020],
        [1648, 2044],
        [1640, 2062],
        [1594, 2054],
        [1596, 2044],
        [1616, 2048],
        [1628, 2040],
        [1628, 2030],
        [1650, 1990],
        [1666, 1972],
        [1664, 1948],
        [1674, 1934],
        [1674, 1926],
        [1650, 1898],
        [1650, 1812],
        [1656, 1806],
        [1656, 1770],
        [1636, 1736],
        [1658, 1674],
        [1656, 1640],
        [1630, 1626],
        [1578, 1624],
        [1568, 1646],
        [1556, 1714],
        [1518, 1756],
        [1516, 1776],
        [1532, 1784],
        [1532, 1830],
        [1520, 1840],
        [1520, 1858],
        [1528, 1860],
        [1526, 1868],
        [1532, 1862],
        [1542, 1864],
        [1556, 1900],
        [1570, 1906],
        [1558, 1944],
        [1526, 1908],
        [1476, 1882],
        [1462, 1864],
        [1436, 1852],
        [1380, 1846],
        [1346, 1804],
        [1336, 1806],
        [1332, 1814],
        [1334, 1832],
        [1370, 1858],
        [1386, 1896],
        [1384, 1908],
        [1362, 1920],
        [1360, 1938],
        [1348, 1936],
        [1352, 1912],
        [1338, 1900],
        [1312, 1914],
        [1304, 1926],
        [1270, 1922],
        [1256, 1928],
        [1248, 1946],
        [1220, 1942],
        [1228, 1938],
        [1226, 1908],
        [1216, 1904],
        [1194, 1920],
        [1196, 1936],
        [1186, 1928],
        [1168, 1930],
        [1130, 1962],
        [1110, 1970],
        [1104, 1980],
        [1092, 1982],
        [1084, 2022],
        [1046, 2028],
        [1040, 2014],
        [1024, 2002],
        [1024, 1992],
        [1034, 1978],
        [1062, 1970],
        [1056, 1948],
        [1038, 1928],
        [988, 1922],
        [1004, 1940],
        [996, 2008],
        [1010, 2020],
        [1010, 2046],
        [1002, 2066],
        [986, 2054],
        [960, 2048],
        [940, 2074],
        [926, 2078],
        [904, 2100],
        [910, 2134],
        [902, 2150],
        [872, 2144],
        [854, 2126],
        [840, 2124],
        [832, 2148],
        [846, 2166],
        [862, 2168],
        [864, 2186],
        [856, 2192],
        [830, 2186],
        [818, 2168],
        [798, 2162],
        [794, 2132],
        [786, 2118],
        [794, 2092],
        [790, 2076],
        [764, 2056],
        [742, 2028],
        [742, 2020],
        [756, 2034],
        [796, 2046],
        [806, 2056],
        [846, 2064],
        [864, 2074],
        [892, 2072],
        [926, 2050],
        [942, 2020],
        [934, 1980],
        [908, 1956],
        [876, 1938],
        [818, 1886],
        [792, 1884],
        [780, 1876],
        [764, 1878],
        [746, 1868],
        [752, 1850],
        [732, 1838],
        [720, 1852],
        [676, 1840],
        [672, 1830],
        [688, 1828],
        [706, 1814],
        [706, 1806],
        [686, 1786],
        [670, 1784],
        [662, 1774],
        [654, 1774],
        [644, 1788],
        [648, 1766],
        [640, 1760],
        [620, 1764],
        [610, 1798],
        [608, 1772],
        [600, 1772],
        [574, 1820],
        [574, 1798],
        [592, 1764],
        [588, 1756],
        [576, 1760],
        [574, 1770],
        [560, 1766],
        [554, 1786],
        [538, 1784],
        [534, 1796],
        [526, 1796],
        [532, 1782],
        [510, 1786],
        [504, 1796],
        [518, 1798],
        [518, 1808],
        [484, 1818],
        [496, 1838],
        [490, 1832],
        [478, 1834],
        [474, 1820],
        [466, 1820],
        [470, 1838],
        [450, 1848],
        [456, 1828],
        [450, 1818],
        [442, 1828],
        [428, 1828],
        [424, 1846],
        [388, 1874],
        [386, 1894],
        [394, 1896],
        [394, 1904],
        [386, 1914],
        [378, 1912],
        [376, 1900],
        [370, 1906],
        [362, 1904],
        [362, 1882],
        [350, 1902],
        [340, 1902],
        [330, 1912],
        [338, 1930],
        [324, 1942],
        [310, 1942],
        [296, 1958],
        [298, 1964],
        [320, 1948],
        [368, 1938],
        [362, 1950],
        [340, 1964],
        [342, 1986],
        [298, 2040],
        [294, 2062],
        [282, 2072],
        [284, 2090],
        [274, 2122],
        [264, 2134],
        [246, 2136],
        [256, 2150],
        [238, 2164],
        [216, 2206],
        [196, 2200],
        [180, 2214],
        [182, 2230],
        [162, 2236],
        [138, 2266],
        [124, 2272],
        [112, 2294],
        [118, 2308],
        [110, 2348],
        [122, 2372],
        [118, 2408],
        [134, 2412],
        [126, 2440],
        [152, 2464],
        [174, 2468],
        [190, 2460],
        [218, 2426],
        [234, 2422],
        [240, 2414],
        [252, 2422],
        [258, 2460],
        [282, 2516],
        [292, 2526],
        [284, 2546],
        [258, 2556],
        [252, 2564],
        [254, 2582],
        [266, 2588],
        [266, 2594],
        [250, 2596],
        [258, 2606],
        [276, 2600],
        [288, 2558],
        [296, 2576],
        [322, 2576],
        [330, 2550],
        [362, 2546],
        [368, 2534],
        [378, 2534],
        [384, 2518],
        [378, 2510],
        [384, 2444],
        [418, 2422],
        [432, 2394],
        [428, 2372],
        [392, 2338],
        [390, 2306],
        [396, 2300],
        [400, 2266],
        [440, 2214],
        [474, 2192],
        [490, 2166],
        [484, 2136],
        [494, 2112],
        [510, 2102],
        [512, 2090],
        [554, 2090],
        [570, 2100],
        [576, 2132],
        [562, 2136],
        [552, 2158],
        [530, 2188],
        [516, 2194],
        [502, 2220],
        [484, 2222],
        [480, 2262],
        [490, 2300],
        [486, 2354],
        [510, 2380],
        [524, 2384],
        [540, 2384],
        [594, 2362],
        [640, 2354],
        [648, 2354],
        [662, 2372],
        [678, 2372],
        [682, 2380],
        [662, 2380],
        [640, 2392],
        [636, 2404],
        [584, 2396],
        [538, 2412],
        [532, 2420],
        [532, 2442],
        [542, 2454],
        [556, 2456],
        [554, 2500],
        [542, 2510],
        [532, 2508],
        [514, 2480],
        [490, 2490],
        [478, 2520],
        [482, 2588],
        [478, 2594],
        [458, 2592],
        [446, 2616],
        [426, 2614],
        [416, 2598],
        [400, 2598],
        [328, 2636],
        [308, 2622],
        [306, 2608],
        [254, 2630],
        [252, 2614],
        [232, 2612],
        [222, 2588],
        [228, 2584],
        [244, 2592],
        [248, 2584],
        [246, 2574],
        [230, 2566],
        [236, 2546],
        [248, 2540],
        [248, 2530],
        [236, 2524],
        [240, 2486],
        [226, 2484],
        [214, 2502],
        [194, 2506],
        [188, 2512],
        [184, 2548],
        [186, 2570],
        [196, 2578],
        [194, 2600],
        [202, 2630],
        [192, 2644],
        [166, 2640],
        [158, 2650],
        [128, 2654],
        [122, 2662],
        [124, 2686],
        [118, 2688],
        [118, 2674],
        [108, 2672],
        [90, 2714],
        [74, 2730],
        [38, 2744],
        [34, 2772],
        [6, 2786],
        [-2, 2802],
        [-22, 2800],
        [-30, 2790],
        [-40, 2790],
        [-38, 2824],
        [-62, 2826],
        [-78, 2818],
        [-108, 2830],
        [-100, 2852],
        [-60, 2864],
        [-48, 2880],
        [-46, 2894],
        [-28, 2910],
        [-24, 2932],
        [-34, 2992],
        [-46, 3000],
        [-128, 2992],
        [-128, 2774],
        [-118, 2778],
        [-108, 2768],
        [-84, 2770],
        [-70, 2754],
        [-48, 2758],
        [18, 2748],
        [32, 2736],
        [20, 2722],
        [40, 2696],
        [34, 2672],
        [8, 2670],
        [-4, 2624],
        [-26, 2604],
        [-36, 2568],
        [-54, 2550],
        [-66, 2548],
        [-40, 2488],
        [-46, 2480],
        [-88, 2480],
        [-88, 2468],
        [-70, 2452],
        [-70, 2442],
        [-114, 2442],
        [-128, 2470],
        [-128, -128],
    ];

    let points: Vec<Point> = polygon1
        .iter()
        .cloned()
        .map(|dot| {
            point(
                (dot[0] + x_offset) as f32 * 0.5,
                (dot[1] + y_offset) as f32 * 0.5,
            )
        })
        .collect();
    path.add_polygon(Polygon {
        points: &points,
        closed: true,
    });

    x_offset = 0;
    let polygon2 = [
        [4224, -128],
        [4224, 2272],
        [4208, 2294],
        [4214, 2308],
        [4206, 2342],
        [4212, 2368],
        [4224, 2372],
        [4214, 2382],
        [4214, 2408],
        [4224, 2412],
        [4224, 2660],
        [4218, 2664],
        [4224, 2686],
        [4214, 2688],
        [4214, 2674],
        [4204, 2672],
        [4186, 2714],
        [4170, 2730],
        [4134, 2744],
        [4130, 2772],
        [4102, 2786],
        [4094, 2802],
        [4074, 2800],
        [4066, 2790],
        [4056, 2790],
        [4058, 2824],
        [4034, 2826],
        [4018, 2818],
        [3988, 2830],
        [3994, 2850],
        [4036, 2864],
        [4048, 2880],
        [4050, 2894],
        [4068, 2910],
        [4068, 2968],
        [4056, 2998],
        [3912, 2988],
        [3886, 3006],
        [3886, 3016],
        [3894, 3024],
        [3898, 3076],
        [3880, 3138],
        [3894, 3150],
        [3894, 3186],
        [3938, 3184],
        [3968, 3220],
        [3960, 3224],
        [3944, 3266],
        [3888, 3306],
        [3872, 3340],
        [3870, 3360],
        [3876, 3376],
        [3866, 3394],
        [3832, 3424],
        [3796, 3438],
        [3788, 3462],
        [3766, 3478],
        [3756, 3516],
        [3736, 3534],
        [3722, 3572],
        [3710, 3582],
        [3708, 3606],
        [3724, 3622],
        [3722, 3652],
        [3730, 3664],
        [3730, 3688],
        [3718, 3736],
        [3704, 3752],
        [3704, 3762],
        [3716, 3776],
        [3714, 3812],
        [3730, 3826],
        [3742, 3828],
        [3764, 3860],
        [3786, 3876],
        [3804, 3922],
        [3910, 3994],
        [3932, 3996],
        [3984, 3978],
        [4026, 3978],
        [4052, 3988],
        [4132, 3954],
        [4198, 3952],
        [4224, 3986],
        [4224, 4224],
        [3294, 4224],
        [3288, 4212],
        [3264, 4212],
        [3246, 4204],
        [3224, 4182],
        [3188, 4160],
        [3154, 4162],
        [3118, 4150],
        [3086, 4152],
        [3078, 4132],
        [3046, 4118],
        [3024, 4110],
        [2994, 4114],
        [2992, 4102],
        [2972, 4098],
        [2956, 4086],
        [2960, 4056],
        [2942, 4044],
        [2934, 4006],
        [2926, 3996],
        [2890, 3970],
        [2862, 3960],
        [2816, 3962],
        [2788, 3952],
        [2768, 3938],
        [2752, 3912],
        [2714, 3896],
        [2712, 3880],
        [2698, 3870],
        [2708, 3864],
        [2706, 3848],
        [2696, 3850],
        [2694, 3868],
        [2676, 3866],
        [2670, 3850],
        [2638, 3852],
        [2622, 3864],
        [2600, 3862],
        [2590, 3852],
        [2552, 3856],
        [2536, 3836],
        [2510, 3830],
        [2506, 3818],
        [2498, 3818],
        [2496, 3836],
        [2464, 3844],
        [2460, 3832],
        [2476, 3822],
        [2476, 3814],
        [2462, 3812],
        [2430, 3836],
        [2408, 3836],
        [2404, 3844],
        [2382, 3848],
        [2374, 3878],
        [2360, 3886],
        [2346, 3906],
        [2322, 3884],
        [2294, 3876],
        [2252, 3894],
        [2228, 3888],
        [2204, 3866],
        [2190, 3842],
        [2202, 3748],
        [2168, 3728],
        [2078, 3728],
        [2088, 3710],
        [2092, 3672],
        [2104, 3658],
        [2106, 3628],
        [2120, 3612],
        [2116, 3594],
        [2084, 3592],
        [2046, 3602],
        [2038, 3610],
        [2030, 3648],
        [2014, 3666],
        [1994, 3662],
        [1944, 3674],
        [1936, 3664],
        [1910, 3654],
        [1870, 3578],
        [1874, 3502],
        [1886, 3492],
        [1878, 3462],
        [1886, 3434],
        [1904, 3416],
        [1924, 3412],
        [1938, 3392],
        [1964, 3386],
        [1992, 3392],
        [2008, 3388],
        [2020, 3400],
        [2032, 3400],
        [2056, 3394],
        [2058, 3378],
        [2070, 3370],
        [2134, 3370],
        [2148, 3376],
        [2156, 3388],
        [2188, 3380],
        [2214, 3408],
        [2212, 3436],
        [2220, 3460],
        [2252, 3504],
        [2264, 3504],
        [2274, 3486],
        [2276, 3464],
        [2242, 3366],
        [2246, 3344],
        [2260, 3320],
        [2288, 3302],
        [2306, 3278],
        [2324, 3274],
        [2330, 3262],
        [2354, 3252],
        [2356, 3238],
        [2372, 3228],
        [2358, 3154],
        [2370, 3158],
        [2372, 3168],
        [2382, 3162],
        [2388, 3150],
        [2382, 3126],
        [2396, 3128],
        [2408, 3114],
        [2412, 3084],
        [2442, 3078],
        [2434, 3066],
        [2496, 3052],
        [2484, 3034],
        [2486, 3010],
        [2496, 2992],
        [2522, 2980],
        [2528, 2970],
        [2542, 2972],
        [2568, 2958],
        [2572, 2946],
        [2616, 2930],
        [2624, 2936],
        [2590, 2968],
        [2592, 2986],
        [2608, 2994],
        [2628, 2978],
        [2636, 2962],
        [2648, 2964],
        [2684, 2950],
        [2700, 2942],
        [2704, 2930],
        [2720, 2928],
        [2732, 2918],
        [2732, 2908],
        [2722, 2904],
        [2720, 2882],
        [2708, 2890],
        [2694, 2922],
        [2654, 2920],
        [2626, 2902],
        [2618, 2878],
        [2622, 2856],
        [2602, 2856],
        [2596, 2850],
        [2612, 2848],
        [2634, 2832],
        [2634, 2818],
        [2622, 2806],
        [2578, 2808],
        [2512, 2848],
        [2526, 2822],
        [2546, 2806],
        [2564, 2802],
        [2570, 2784],
        [2584, 2772],
        [2730, 2770],
        [2758, 2740],
        [2780, 2728],
        [2804, 2726],
        [2828, 2704],
        [2826, 2652],
        [2812, 2640],
        [2798, 2638],
        [2782, 2620],
        [2790, 2612],
        [2786, 2604],
        [2760, 2598],
        [2752, 2586],
        [2730, 2578],
        [2722, 2562],
        [2698, 2540],
        [2702, 2520],
        [2688, 2484],
        [2690, 2476],
        [2672, 2460],
        [2672, 2448],
        [2640, 2394],
        [2630, 2362],
        [2620, 2360],
        [2604, 2392],
        [2606, 2404],
        [2596, 2428],
        [2582, 2434],
        [2570, 2450],
        [2558, 2452],
        [2536, 2428],
        [2522, 2426],
        [2520, 2412],
        [2512, 2406],
        [2510, 2350],
        [2518, 2336],
        [2472, 2324],
        [2464, 2314],
        [2466, 2302],
        [2422, 2264],
        [2388, 2272],
        [2332, 2256],
        [2318, 2268],
        [2318, 2294],
        [2328, 2308],
        [2318, 2342],
        [2326, 2348],
        [2334, 2386],
        [2308, 2428],
        [2308, 2440],
        [2346, 2480],
        [2354, 2542],
        [2332, 2574],
        [2284, 2602],
        [2296, 2632],
        [2308, 2700],
        [2298, 2720],
        [2288, 2720],
        [2280, 2732],
        [2266, 2726],
        [2232, 2678],
        [2258, 2678],
        [2260, 2672],
        [2248, 2660],
        [2228, 2666],
        [2222, 2586],
        [2158, 2578],
        [2146, 2566],
        [2106, 2550],
        [2096, 2532],
        [2076, 2516],
        [2038, 2500],
        [1992, 2504],
        [1992, 2486],
        [1976, 2436],
        [1950, 2434],
        [1940, 2422],
        [1944, 2354],
        [1970, 2292],
        [1994, 2260],
        [2002, 2258],
        [1998, 2248],
        [2018, 2246],
        [2034, 2236],
        [2032, 2212],
        [2046, 2202],
        [2052, 2184],
        [2080, 2186],
        [2094, 2174],
        [2118, 2128],
        [2112, 2112],
        [2140, 2076],
        [2140, 2066],
        [2124, 2052],
        [2150, 2048],
        [2160, 2064],
        [2174, 2068],
        [2184, 2092],
        [2198, 2094],
        [2192, 2078],
        [2176, 2066],
        [2188, 2066],
        [2190, 2052],
        [2204, 2056],
        [2244, 2018],
        [2248, 1990],
        [2228, 1964],
        [2224, 1934],
        [2248, 1922],
        [2248, 1910],
        [2240, 1904],
        [2246, 1888],
        [2228, 1880],
        [2218, 1860],
        [2222, 1846],
        [2242, 1840],
        [2276, 1864],
        [2290, 1854],
        [2288, 1842],
        [2306, 1838],
        [2304, 1816],
        [2296, 1810],
        [2302, 1806],
        [2310, 1818],
        [2326, 1822],
        [2328, 1844],
        [2342, 1846],
        [2342, 1858],
        [2374, 1882],
        [2372, 1894],
        [2352, 1898],
        [2352, 1916],
        [2360, 1918],
        [2378, 1904],
        [2392, 1904],
        [2394, 1916],
        [2384, 1916],
        [2380, 1930],
        [2388, 1938],
        [2396, 1928],
        [2410, 1930],
        [2416, 1942],
        [2436, 1950],
        [2450, 2012],
        [2402, 2066],
        [2418, 2102],
        [2386, 2116],
        [2334, 2108],
        [2334, 2120],
        [2318, 2134],
        [2316, 2154],
        [2336, 2172],
        [2358, 2172],
        [2374, 2160],
        [2406, 2160],
        [2418, 2166],
        [2434, 2178],
        [2446, 2200],
        [2466, 2204],
        [2462, 2224],
        [2478, 2236],
        [2488, 2258],
        [2510, 2248],
        [2522, 2266],
        [2538, 2274],
        [2586, 2292],
        [2594, 2290],
        [2596, 2274],
        [2534, 2210],
        [2532, 2202],
        [2538, 2200],
        [2582, 2236],
        [2606, 2244],
        [2622, 2268],
        [2630, 2266],
        [2618, 2244],
        [2624, 2240],
        [2620, 2224],
        [2634, 2214],
        [2632, 2194],
        [2616, 2178],
        [2612, 2152],
        [2576, 2128],
        [2566, 2096],
        [2552, 2092],
        [2552, 2086],
        [2568, 2080],
        [2558, 2062],
        [2580, 2060],
        [2604, 2080],
        [2620, 2118],
        [2650, 2134],
        [2654, 2100],
        [2672, 2096],
        [2684, 2076],
        [2680, 2064],
        [2696, 2056],
        [2702, 2038],
        [2684, 2020],
        [2664, 2020],
        [2662, 2002],
        [2642, 2002],
        [2642, 1986],
        [2620, 1972],
        [2620, 1960],
        [2596, 1964],
        [2588, 1948],
        [2576, 1946],
        [2574, 1932],
        [2544, 1924],
        [2540, 1912],
        [2546, 1912],
        [2552, 1900],
        [2546, 1886],
        [2578, 1886],
        [2572, 1870],
        [2552, 1862],
        [2552, 1852],
        [2568, 1850],
        [2564, 1826],
        [2552, 1814],
        [2538, 1822],
        [2524, 1818],
        [2536, 1810],
        [2542, 1796],
        [2520, 1780],
        [2506, 1782],
        [2500, 1776],
        [2490, 1782],
        [2490, 1760],
        [2480, 1758],
        [2472, 1764],
        [2476, 1740],
        [2462, 1724],
        [2436, 1720],
        [2422, 1726],
        [2418, 1710],
        [2398, 1716],
        [2406, 1710],
        [2410, 1692],
        [2392, 1682],
        [2384, 1654],
        [2360, 1648],
        [2344, 1636],
        [2328, 1636],
        [2312, 1646],
        [2308, 1666],
        [2276, 1658],
        [2260, 1690],
        [2258, 1672],
        [2270, 1634],
        [2264, 1628],
        [2262, 1604],
        [2250, 1596],
        [2252, 1584],
        [2240, 1560],
        [2210, 1560],
        [2158, 1594],
        [2144, 1640],
        [2148, 1658],
        [2162, 1672],
        [2150, 1676],
        [2142, 1696],
        [2156, 1718],
        [2170, 1722],
        [2170, 1732],
        [2154, 1730],
        [2132, 1696],
        [2134, 1658],
        [2124, 1644],
        [2124, 1630],
        [2140, 1590],
        [2162, 1568],
        [2164, 1556],
        [2134, 1548],
        [2098, 1560],
        [2066, 1602],
        [2062, 1632],
        [2048, 1666],
        [2048, 1736],
        [2060, 1746],
        [2096, 1750],
        [2106, 1762],
        [2102, 1768],
        [2088, 1768],
        [2084, 1762],
        [2062, 1766],
        [2070, 1794],
        [2092, 1814],
        [2122, 1812],
        [2142, 1832],
        [2208, 1834],
        [2212, 1850],
        [2194, 1852],
        [2176, 1842],
        [2150, 1848],
        [2152, 1884],
        [2162, 1902],
        [2146, 1918],
        [2140, 1958],
        [2132, 1960],
        [2128, 1944],
        [2118, 1948],
        [2118, 1972],
        [2126, 1978],
        [2128, 1990],
        [2110, 2004],
        [2086, 1964],
        [2088, 1948],
        [2098, 1944],
        [2094, 1908],
        [2072, 1882],
        [2064, 1882],
        [2052, 1898],
        [2048, 1934],
        [2040, 1938],
        [2034, 1884],
        [2044, 1878],
        [2044, 1870],
        [2022, 1856],
        [2004, 1864],
        [1992, 1852],
        [2012, 1814],
        [1984, 1778],
        [1984, 1748],
        [1978, 1734],
        [1960, 1710],
        [1946, 1708],
        [1970, 1662],
        [1960, 1638],
        [2002, 1638],
        [2042, 1554],
        [2042, 1544],
        [2028, 1536],
        [1998, 1538],
        [1976, 1522],
        [1934, 1532],
        [1926, 1542],
        [1930, 1556],
        [1918, 1562],
        [1916, 1626],
        [1930, 1658],
        [1930, 1700],
        [1914, 1720],
        [1920, 1740],
        [1910, 1736],
        [1900, 1744],
        [1898, 1780],
        [1906, 1798],
        [1898, 1820],
        [1910, 1846],
        [1926, 1858],
        [1944, 1860],
        [1954, 1872],
        [1944, 1912],
        [1954, 1914],
        [1964, 1904],
        [1964, 1928],
        [1940, 1956],
        [1922, 1962],
        [1926, 2002],
        [1910, 2000],
        [1906, 1992],
        [1912, 1950],
        [1886, 1936],
        [1886, 1930],
        [1906, 1932],
        [1928, 1912],
        [1916, 1902],
        [1910, 1882],
        [1912, 1874],
        [1926, 1874],
        [1926, 1862],
        [1894, 1864],
        [1874, 1842],
        [1862, 1844],
        [1852, 1880],
        [1832, 1892],
        [1836, 1906],
        [1872, 1922],
        [1862, 1938],
        [1852, 1940],
        [1852, 1972],
        [1806, 1968],
        [1792, 1976],
        [1774, 1976],
        [1752, 1966],
        [1742, 1954],
        [1726, 1958],
        [1700, 1936],
        [1694, 1908],
        [1678, 1902],
        [1656, 1916],
        [1630, 1922],
        [1622, 1944],
        [1632, 1952],
        [1644, 1952],
        [1646, 1942],
        [1670, 1940],
        [1670, 1946],
        [1642, 1958],
        [1636, 1986],
        [1620, 1986],
        [1614, 1974],
        [1602, 1974],
        [1590, 1960],
        [1574, 1970],
        [1538, 1978],
        [1476, 1970],
        [1474, 1958],
        [1480, 1950],
        [1502, 1944],
        [1500, 1926],
        [1484, 1908],
        [1470, 1900],
        [1436, 1904],
        [1414, 1898],
        [1396, 1882],
        [1360, 1872],
        [1344, 1854],
        [1320, 1844],
        [1296, 1846],
        [1292, 1866],
        [1284, 1874],
        [1266, 1874],
        [1270, 1852],
        [1262, 1824],
        [1242, 1854],
        [1242, 1874],
        [1234, 1874],
        [1214, 1852],
        [1208, 1826],
        [1188, 1798],
        [1178, 1802],
        [1184, 1820],
        [1176, 1824],
        [1174, 1838],
        [1164, 1848],
        [1148, 1844],
        [1132, 1852],
        [1120, 1868],
        [1118, 1860],
        [1128, 1848],
        [1150, 1834],
        [1148, 1822],
        [1136, 1828],
        [1122, 1824],
        [1096, 1848],
        [1074, 1856],
        [1066, 1872],
        [1052, 1874],
        [1050, 1862],
        [1040, 1856],
        [1018, 1868],
        [998, 1884],
        [998, 1902],
        [986, 1906],
        [954, 1890],
        [936, 1874],
        [932, 1862],
        [882, 1856],
        [840, 1826],
        [794, 1834],
        [774, 1822],
        [734, 1818],
        [702, 1800],
        [646, 1804],
        [628, 1792],
        [628, 1776],
        [616, 1772],
        [582, 1778],
        [578, 1766],
        [566, 1760],
        [550, 1774],
        [554, 1754],
        [530, 1742],
        [502, 1776],
        [464, 1778],
        [436, 1810],
        [412, 1814],
        [388, 1846],
        [384, 1874],
        [364, 1900],
        [314, 1908],
        [308, 1940],
        [362, 1984],
        [370, 2010],
        [396, 2020],
        [398, 2034],
        [418, 2044],
        [416, 2054],
        [424, 2056],
        [424, 2062],
        [410, 2074],
        [372, 2072],
        [370, 2046],
        [350, 2044],
        [328, 2054],
        [322, 2068],
        [304, 2070],
        [272, 2100],
        [308, 2118],
        [302, 2130],
        [308, 2150],
        [320, 2158],
        [342, 2162],
        [372, 2156],
        [394, 2164],
        [406, 2150],
        [434, 2142],
        [436, 2152],
        [424, 2158],
        [436, 2190],
        [432, 2206],
        [422, 2212],
        [402, 2210],
        [384, 2232],
        [366, 2222],
        [354, 2224],
        [342, 2256],
        [324, 2278],
        [316, 2306],
        [324, 2322],
        [338, 2330],
        [342, 2342],
        [336, 2356],
        [356, 2374],
        [362, 2388],
        [382, 2388],
        [404, 2376],
        [412, 2394],
        [414, 2440],
        [436, 2440],
        [442, 2426],
        [466, 2432],
        [476, 2448],
        [484, 2446],
        [484, 2432],
        [494, 2440],
        [518, 2434],
        [510, 2450],
        [508, 2482],
        [486, 2504],
        [484, 2514],
        [454, 2530],
        [442, 2554],
        [418, 2552],
        [382, 2590],
        [354, 2594],
        [344, 2614],
        [376, 2606],
        [384, 2594],
        [414, 2586],
        [418, 2574],
        [440, 2574],
        [458, 2560],
        [486, 2552],
        [496, 2532],
        [534, 2510],
        [538, 2496],
        [570, 2468],
        [584, 2464],
        [592, 2446],
        [606, 2436],
        [606, 2426],
        [590, 2418],
        [590, 2410],
        [614, 2392],
        [634, 2352],
        [664, 2322],
        [680, 2322],
        [686, 2332],
        [670, 2332],
        [652, 2344],
        [652, 2364],
        [640, 2386],
        [648, 2396],
        [638, 2408],
        [642, 2416],
        [664, 2412],
        [694, 2382],
        [730, 2378],
        [728, 2344],
        [754, 2336],
        [774, 2356],
        [796, 2362],
        [820, 2380],
        [856, 2376],
        [890, 2392],
        [918, 2388],
        [920, 2406],
        [950, 2422],
        [976, 2452],
        [996, 2456],
        [988, 2470],
        [1028, 2538],
        [1034, 2528],
        [1032, 2508],
        [1046, 2500],
        [1054, 2482],
        [1058, 2504],
        [1048, 2506],
        [1036, 2522],
        [1042, 2528],
        [1042, 2544],
        [1056, 2534],
        [1058, 2574],
        [1070, 2600],
        [1092, 2600],
        [1094, 2582],
        [1088, 2570],
        [1094, 2570],
        [1096, 2582],
        [1110, 2584],
        [1118, 2600],
        [1128, 2602],
        [1128, 2652],
        [1136, 2660],
        [1150, 2660],
        [1156, 2682],
        [1178, 2702],
        [1192, 2744],
        [1212, 2752],
        [1208, 2756],
        [1182, 2746],
        [1176, 2758],
        [1218, 2800],
        [1264, 2832],
        [1258, 2850],
        [1272, 2880],
        [1274, 2946],
        [1262, 3020],
        [1270, 3048],
        [1266, 3094],
        [1278, 3108],
        [1280, 3134],
        [1308, 3164],
        [1310, 3184],
        [1322, 3192],
        [1322, 3208],
        [1346, 3236],
        [1350, 3256],
        [1400, 3274],
        [1426, 3296],
        [1428, 3312],
        [1464, 3388],
        [1498, 3418],
        [1498, 3434],
        [1484, 3436],
        [1482, 3442],
        [1492, 3454],
        [1520, 3464],
        [1540, 3480],
        [1544, 3514],
        [1580, 3540],
        [1592, 3560],
        [1606, 3554],
        [1598, 3532],
        [1578, 3522],
        [1578, 3508],
        [1560, 3468],
        [1534, 3440],
        [1516, 3400],
        [1490, 3380],
        [1484, 3338],
        [1508, 3340],
        [1520, 3348],
        [1538, 3396],
        [1536, 3410],
        [1546, 3410],
        [1564, 3432],
        [1578, 3438],
        [1580, 3448],
        [1608, 3470],
        [1606, 3488],
        [1630, 3502],
        [1690, 3568],
        [1700, 3596],
        [1694, 3632],
        [1742, 3674],
        [1778, 3682],
        [1804, 3702],
        [1896, 3736],
        [1942, 3720],
        [1962, 3728],
        [2010, 3774],
        [2040, 3778],
        [2078, 3794],
        [2102, 3792],
        [2106, 3806],
        [2142, 3838],
        [2146, 3870],
        [2168, 3872],
        [2190, 3888],
        [2192, 3902],
        [2228, 3908],
        [2240, 3920],
        [2252, 3922],
        [2256, 3932],
        [2272, 3928],
        [2268, 3908],
        [2286, 3892],
        [2302, 3894],
        [2314, 3906],
        [2312, 3918],
        [2334, 3946],
        [2338, 4012],
        [2324, 4034],
        [2308, 4038],
        [2300, 4064],
        [2274, 4078],
        [2272, 4096],
        [2256, 4120],
        [2256, 4150],
        [2274, 4164],
        [2274, 4170],
        [2246, 4194],
        [2252, 4224],
        [-128, 4224],
        [-128, 2294],
        [-68, 2260],
        [-20, 2268],
        [-10, 2258],
        [-8, 2246],
        [-14, 2240],
        [-14, 2226],
        [-24, 2216],
        [-32, 2176],
        [-56, 2164],
        [-56, 2150],
        [-24, 2150],
        [10, 2122],
        [16, 2096],
        [6, 2086],
        [8, 2072],
        [30, 2066],
        [26, 2084],
        [36, 2106],
        [58, 2108],
        [76, 2102],
        [88, 2108],
        [102, 2142],
        [120, 2144],
        [140, 2166],
        [156, 2172],
        [172, 2164],
        [174, 2136],
        [180, 2128],
        [176, 2106],
        [204, 2106],
        [216, 2096],
        [218, 2082],
        [234, 2074],
        [198, 2040],
        [188, 2022],
        [158, 2024],
        [148, 2014],
        [134, 2016],
        [136, 2042],
        [134, 2052],
        [128, 2052],
        [118, 2040],
        [118, 1998],
        [112, 1996],
        [108, 1982],
        [98, 1974],
        [90, 1978],
        [88, 1966],
        [52, 1944],
        [32, 1926],
        [26, 1912],
        [0, 1900],
        [-28, 1870],
        [-68, 1858],
        [-88, 1842],
        [-128, 1842],
        [-128, -128],
    ];

    let points: Vec<Point> = polygon2
        .iter()
        .cloned()
        .map(|dot| {
            point(
                (dot[0] + x_offset) as f32 * 0.5,
                (dot[1] + y_offset) as f32 * 0.5,
            )
        })
        .collect();
    path.add_polygon(Polygon {
        points: &points,
        closed: true,
    });
}
