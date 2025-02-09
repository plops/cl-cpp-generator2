//
// Created by martin on 2/7/25.
//

/**
 * needs ~/src/Daxa and ~/vulkan VulkanMemory...
 */

#include <array>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>
#include <iostream>
#include <thread>

#include "shared.inl"
#include "window.h"

using namespace daxa;

void upload_vertex_data_task(TaskGraph& tg, const TaskBufferView vertices)
{
    // Task that will send data to the GPU
    tg.add_task({
        .attachments{{inl_attachment(TaskBufferAccess::TRANSFER_WRITE, vertices)}}
      , .task{[vertices](const TaskInterface& ti)
        {
            // The triangle coordinates are fixed here
            constexpr float n{-.5f}, p{.5f}, z{.0f}, o{1.f};
            constexpr auto data{std::array{
                MyVertex{.position{n
                                 , p
                                 , z}
                       , .color{o
                              , z
                              , z}}
              , MyVertex{.position{p
                                 , p
                                 , z}
                       , .color{z
                              , o
                              , z}}
              , MyVertex{.position{z
                                 , n
                                 , z}
                       , .color{z
                              , z
                              , o}}
               ,
            }};
            const auto staging_buffer_id{ti.device.create_buffer({.size{sizeof(data)}
                                                                , .allocate_info{MemoryFlagBits::HOST_ACCESS_RANDOM}
                                                                , .name{"my_staging_buffer"}})};
            // Defer destruction of the buffer until after it is on the GPU
            ti.recorder.destroy_buffer_deferred(staging_buffer_id);
            auto* buffer_ptr{
                ti.device.buffer_host_address_as<std::array<MyVertex, 3>>(staging_buffer_id).value()};
            *buffer_ptr = data;
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer{staging_buffer_id}
              , .dst_buffer{ti.get(vertices).ids[0]}
              , .size{sizeof(data)}
               ,
            });
        }}
      , .name{"upload_vertex_data_task"}
       ,
    });
}

void draw_vertices_task(TaskGraph& tg, const std::shared_ptr<RasterPipeline>& pipeline, const TaskBufferView vertices,
                        const TaskImageView& render_target
    )
{
    // Create Rendering task
    tg.add_task(
    {.attachments{inl_attachment(TaskBufferAccess::VERTEX_SHADER_READ, vertices)
                , inl_attachment(TaskImageAccess::COLOR_ATTACHMENT, ImageViewType::REGULAR_2D, render_target)}
   , .task{[render_target, pipeline, vertices](const TaskInterface& ti)
     {
         // Get screen dimensions from the target image
         auto size{ti.device.info(ti.get(render_target).ids[0]).value().size};
         // std::cout << "size: " << size.x << ", " << size.y << std::endl;
         // Record the actual renderpass
         auto render_recorder{std::move(ti.recorder)
            .begin_renderpass({
                 .color_attachments{std::array{RenderAttachmentInfo{
                     .image_view{ti.get(render_target).ids[0]}
                   , .load_op{AttachmentLoadOp::CLEAR}
                   , .clear_value{std::array{.1f
                                           , .0f
                                           , .5f
                                           , 1.f}}}}}
               , .render_area{.width{size.x}
                            , .height{size.y}}
                ,
             })};
         render_recorder.set_pipeline(*pipeline);
         render_recorder.push_constant(
             MyPushConstant{.my_vertex_ptr{ti.device.device_address(ti.get(vertices).ids[0]).value()}});
         render_recorder.draw({.vertex_count{3}});
         ti.recorder = std::move(render_recorder).end_renderpass();
     }}
   , .name{"draw_vertices_task"}});
}

int main(const int argc, char const* argv[])
{
    std::cout << argc << " " << argv[0] << std::endl;
    // Create a window
    auto window{AppWindow("Learn Daxa", 860, 640)};
    std::cout << "App window created" << std::endl;

    auto instance{create_instance({})};
    auto device{instance.create_device_2(instance.choose_device({}, {}))};
    auto swapchain{device.create_swapchain({.native_window{window.get_native_handle()}
                                          , .native_window_platform{AppWindow::get_native_platform()}
                                           ,
                                            // .surface_format_selector{[](Format format)
                                            //                          {
                                            //                              switch (format)
                                            //                              {
                                            //                              case Format::R8G8B8A8_UINT:
                                            //                                  return 100;
                                            //                              default:
                                            //                                  return default_format_score(format);
                                            //                              }
                                            //                          }},
                                            .present_mode{PresentMode::FIFO}
                                          , .image_usage{ImageUsageFlagBits::TRANSFER_DST}
                                          , .name{"my swapchain"}})};

    auto pipeline_manager{
        PipelineManager({.device{device}
                       , .shader_compile_options{.root_paths{"/home/martin/src/Daxa/include"
                                                           , "."
                                                           , "../src"}
                                               , .language{ShaderLanguage::GLSL}
                                               , .enable_debug_info{true}}
                       , .name{"my pipeline manager"}})};

    std::shared_ptr<RasterPipeline> pipeline;
    {
        constexpr auto fn{"/home/martin/stage/cl-cpp-generator2/example/164_daxa/src/main.glsl"};
        const auto shaderFile{ShaderFile{fn}};
        const auto result = pipeline_manager.add_raster_pipeline({
            .vertex_shader_info{ShaderCompileInfo{.source{shaderFile}}}
          , .fragment_shader_info{ShaderCompileInfo{.source{shaderFile}}}
          , .color_attachments{{{.format{swapchain.get_format()}}}}
          , .raster{{}}
          , .push_constant_size{sizeof(MyPushConstant)}
          , .name{"my pipeline"}
           ,
        });
        if (result.is_err())
        {
            std::cerr << result.message() << std::endl;
            return -1;
        }
        pipeline = result.value();
        std::cout << "pipeline created" << std::endl;
    }

    auto buffer_id{device.create_buffer({.size{3 * sizeof(MyVertex)}
                                       , .name{"my vertex data"}})};

    auto task_swapchain_image{TaskImage({.swapchain_image{true}
                                       , .name{"task swapchain image"}})};
    const auto task_vertex_buffer{
        TaskBuffer({.initial_buffers =
                    {
                        .buffers{std::span{&buffer_id
                                         , 1}}
                    }
                  , .name{"my task vertex buffer"}})};

    auto loop_task_graph{TaskGraph({.device{device}
                                  , .swapchain{swapchain}
                                  , .name{"my loop"}})};

    // Manually mark used resources (this is needed to detect errors in graph)
    loop_task_graph.use_persistent_buffer(task_vertex_buffer);
    loop_task_graph.use_persistent_image(task_swapchain_image);

    // Fill the rendering task graph
    draw_vertices_task(loop_task_graph, pipeline, task_vertex_buffer, task_swapchain_image);

    // Tell the task graph that we are done filling it
    loop_task_graph.submit({});
    // Do the present step
    loop_task_graph.present({});
    // Compile the dependency graph between tasks
    loop_task_graph.complete({});
    std::cout << "render task graph completed" << std::endl;

    // Secondary task graph that transfers the vertices. Only runs once
    {
        auto upload_task_graph{TaskGraph({
            .device{device}
          , .name{"upload vertex buffer task graph"}
           ,
        })};
        upload_task_graph.use_persistent_buffer(task_vertex_buffer);
        upload_vertex_data_task(upload_task_graph, task_vertex_buffer);
        upload_task_graph.submit({});
        upload_task_graph.complete({});
        upload_task_graph.execute({});
        std::cout << "upload vertex buffer task graph executed" << std::endl;
    }


    // Main loop
    while (!window.should_close())
    {
        window.update();
        if (window.swapchain_out_of_date)
        {
            swapchain.resize();
            window.swapchain_out_of_date = false;
        }

        try
        {
            // Acquire the next image
            auto swapchain_image{swapchain.acquire_next_image()};
            if (swapchain_image.is_empty()) { continue; }
            // Update image id
            task_swapchain_image.set_images({.images{std::span{&swapchain_image
                                                             , 1}}});
        }
        catch (const std::logic_error&) { continue; }

        // Execute render task graph
        loop_task_graph.execute({});
        device.collect_garbage();
    }

    device.destroy_buffer(buffer_id);

    device.wait_idle();
    device.collect_garbage();

    std::cout << "cleaned up" << std::endl;

    return 0;
}
