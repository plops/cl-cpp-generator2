//
// Created by martin on 2/7/25.
//

/**
 * needs ~/src/Daxa and ~/vulkan VulkanMemory...
 */

#include <iostream>
#include <memory> // shared_ptr
#include <string>
#include <vector>
#include <array>
#include <exception>
#include <stdexcept>
#include <daxa/instance.hpp>
#include <daxa/types.hpp>
#include <daxa/pipeline.hpp>
#include <daxa/utils/task_graph_types.hpp>
#include <daxa/utils/pipeline_manager.hpp> //
#include <daxa/gpu_resources.hpp>
#include <daxa/utils/task_graph.hpp> //
#include <utility> // move

#include <daxa/utils/imgui.hpp>
#include <backends/imgui_impl_glfw.h>
#include <imgui.h>
#include "daxa/command_recorder.hpp"
#include "shared.inl"
#include "window.h"

using namespace daxa;

namespace
{
    /**
     * @brief Define daxa task to upload vertex coordinates to GPU
     * @param tg
     * @param vertices
     */
    void upload_vertex_data_task(TaskGraph & tg, const TaskBufferView vertices)
    {
        // Task that will send data to the GPU
        tg.add_task({
            .attachments{{inl_attachment(TaskBufferAccess::TRANSFER_WRITE, vertices)}}
          , .task{[vertices](const TaskInterface & ti)
            {
                // The triangle coordinates are fixed here
                constexpr auto n{-.5f};
                constexpr auto p{.5f};
                constexpr auto z{.0f};
                constexpr auto o{1.f};
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
                // Defer destruction of the buffer until after it is on the GPU (when garbage_collect
                // is called on the device, which happens once per frame)
                ti.recorder.destroy_buffer_deferred(staging_buffer_id);
                auto * buffer_ptr{
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

    /**
     * @brief Render task
     * @param tg
     * @param pipeline
     * @param vertices
     * @param render_target
     */
    void draw_vertices_task(TaskGraph &   tg, const std::shared_ptr<RasterPipeline> & pipeline, const TaskBufferView vertices,
                            TaskImageView render_target
        )
    {
        // Create Rendering task
        tg.add_task(
        {.attachments{inl_attachment(TaskBufferAccess::VERTEX_SHADER_READ, vertices)
                    , inl_attachment(TaskImageAccess::COLOR_ATTACHMENT, ImageViewType::REGULAR_2D, render_target)}
       , .task{[render_target, pipeline, vertices](const TaskInterface & ti)
         {
             // Get screen dimensions from the target image
             auto size{ti.device.info(ti.get(render_target).ids[0]).value().size};
             // std::cout << "size: " << size.x << ", " << size.y << std::endl;
             // Record the actual renderpass
             constexpr auto color{std::array{.1f
                                           , .0f
                                           , .5f
                                           , 1.f}};
             auto render_recorder{std::move(ti.recorder)
                .begin_renderpass({
                     .color_attachments{std::array{RenderAttachmentInfo{
                         .image_view{ti.get(render_target).ids[0]}
                       , .load_op{AttachmentLoadOp::CLEAR}
                       , .clear_value{color}}}}
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

    void msg(const std::string & message)
    {
        constexpr auto debug{true};
        if constexpr (debug) { std::cout << message << '\n'; }
    }
}

int main(const int argc, char const * argv[])
{
    std::cout << argc << " " << argv[0] << '\n';
    // Create a window
    constexpr auto w{860};
    constexpr auto h{640};
    auto           window{AppWindow("Learn Daxa", w, h)};
    msg("App window created");

    auto instance{create_instance({})};
    auto device{instance.create_device_2(instance.choose_device({}, {}))};
    // auto di{device.info()};
    // auto dp{device.properties()};
    // std::cout << "device: " << dp.device_name << std::endl;
    auto swapchain{device.create_swapchain({.native_window{window.get_native_handle()}
                                          , .native_window_platform{AppWindow::get_native_platform()}
                                          , .present_mode{PresentMode::FIFO}
                                          , .image_usage{ImageUsageFlagBits::TRANSFER_DST}
                                          , .name{"my swapchain"}})};

    auto pipeline_manager{
        PipelineManager({.device{device}
                       , .shader_compile_options{.root_paths{DAXA_INCLUDE
                                                           , "."
                                                           , "../src"}
                                               , .language{ShaderLanguage::GLSL}
                                               , .enable_debug_info{true}}
                       , .name{"my pipeline manager"}})};

    auto imgui_renderer{[&device, &swapchain, &window]() -> ImGuiRenderer
    {
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForVulkan(window.glfw_window_ptr, true);
        return ImGuiRenderer({
            .device = device
          , .format = swapchain.get_format()
        });
    }()};
    std::vector<TaskAttachmentInfo> imgui_task_attachments{};

    try
    {
        const auto pipeline
        {[&pipeline_manager,&swapchain]
        {
            const std::string fn{SHADER_PATH};
            const auto        shaderFile{ShaderFile{fn + "/main.glsl"}};
            const auto        result = pipeline_manager.add_raster_pipeline({
                .vertex_shader_info{ShaderCompileInfo{.source{shaderFile}}}
              , .fragment_shader_info{ShaderCompileInfo{.source{shaderFile}}}
              , .color_attachments{{{.format{swapchain.get_format()}}}}
              , .raster = {}
              , .push_constant_size{sizeof(MyPushConstant)}
              , .name{"my pipeline"}
            });
            if (result.is_err())
            {
                std::cerr << result.message() << '\n';
                throw std::runtime_error("failed to create raster pipeline");
            }

            msg("pipeline created");
            // This returns a shared pointer of the pipeline, the pipeline manager retains ownership. Note that the pipeline
            // manager is meant to be used during development only. The pipeline manager provides hot-reloading via its
            // reload_all() method. Don't use in shipped code.
            return result.value();
        }()};

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

        imgui_task_attachments.push_back(daxa::inl_attachment(daxa::TaskImageAccess::COLOR_ATTACHMENT, task_swapchain_image));
        auto imgui_task_info = InlineTaskInfo{
            .attachments = std::move(imgui_task_attachments)
          , .task = [&](daxa::TaskInterface const & ti)
            {
                auto size{ti.device.info(ti.get(task_swapchain_image).ids[0]).value().size};
                imgui_renderer.record_commands(ImGui::GetDrawData(), ti.recorder, ti.get(task_swapchain_image).ids[0], size.x, size.y);
            }
          , .name = "ImGui Task"
           ,
        };
        loop_task_graph.add_task(imgui_task_info);

        // Tell the task graph that we are done filling it
        loop_task_graph.submit({});
        // Do the present step
        loop_task_graph.present({});
        // Compile the dependency graph between tasks
        loop_task_graph.complete({});
        msg("render task graph prepared");

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
            msg("upload vertex buffer task graph executed");
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

            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::ShowDemoWindow();
            ImGui::Begin("Settings");

            // ImGui::Image(
            //     imgui_renderer.create_texture_id({
            //         .image_view_id = render_image.default_view(),
            //         .sampler_id = sampler,
            //     }),
            //     ImVec2(200, 200));
            //
            // if (ImGui::Checkbox("MY_TOGGLE", &my_toggle))
            // {
            //     update_virtual_shader();
            // }
            ImGui::End();
            ImGui::Render();

            try
            {
                // Acquire the next image
                auto swapchain_image{swapchain.acquire_next_image()};
                if (swapchain_image.is_empty()) { continue; }
                // Update image id
                task_swapchain_image.set_images({.images{std::span{&swapchain_image
                                                                 , 1}}});
            }
            catch (const std::logic_error &) { continue; }

            // Execute render task graph
            loop_task_graph.execute({});
            device.collect_garbage();
        }
        ImGui_ImplGlfw_Shutdown();

        device.destroy_buffer(buffer_id);

        device.wait_idle();
        device.collect_garbage();

        msg("cleaned up");
    }
    catch (const std::exception & e) { std::cerr << e.what() << '\n'; }
    return 0;
}
