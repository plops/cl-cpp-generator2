#include "VWindow.h"

#include <iostream>

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>

#include "shader/shared.inl"

using namespace daxa::task_resource_uses;
using namespace daxa::types;

struct UploadVertexDataTask {
    struct Uses {
        daxa::BufferTransferWrite vertex_buffer{};
    } uses = {};

    std::string_view name = "upload vertices";

    void callback(daxa::TaskInterface ti) {
        // [...]
        auto &recorder = ti.get_recorder();
        auto data = std::array{
                MyVertex{.position = {-0.5f, +0.5f, 0.0f}, .color = {1.0f, 0.0f, 0.0f}},
                MyVertex{.position = {+0.5f, +0.5f, 0.0f}, .color = {0.0f, 1.0f, 0.0f}},
                MyVertex{.position = {+0.0f, -0.5f, 0.0f}, .color = {0.0f, 0.0f, 1.0f}},
        };
        auto staging_buffer_id = ti.get_device().create_buffer({
                                                                       .size = sizeof(data),
                                                                       .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                                                                       .name = "my staging buffer",
                                                               });

        recorder.destroy_buffer_deferred(staging_buffer_id);
        auto *buffer_ptr = ti.get_device().get_host_address_as<std::array<MyVertex, 3>>(staging_buffer_id).value();
        *buffer_ptr = data;
        recorder.copy_buffer_to_buffer({
                                               .src_buffer = staging_buffer_id,
                                               .dst_buffer = uses.vertex_buffer.buffer(),
                                               .size = sizeof(data),
                                       });

    }

    int main(int argc, char const *argv[]) {
        // Create a window
        auto window = VWindow("Learn Daxa", 860, 640);

        auto instance = daxa::create_instance({});
        auto device = instance.create_device({
                                                     .selector = [](
                                                             daxa::DeviceProperties const &device_props) -> daxa::i32 {
                                                         daxa::i32 score = 0;
                                                         switch (device_props.device_type) {
                                                             case daxa::DeviceType::DISCRETE_GPU:
                                                                 score += 10000;
                                                                 break;
                                                             case daxa::DeviceType::VIRTUAL_GPU:
                                                                 score += 1000;
                                                                 break;
                                                             case daxa::DeviceType::INTEGRATED_GPU:
                                                                 score += 100;
                                                                 break;
                                                             default:
                                                                 break;
                                                         }
                                                         score += static_cast<daxa::i32>(
                                                                 device_props.limits.max_memory_allocation_count /
                                                                 100000);
                                                         return score;
                                                     },
                                                     .name = "my device",
                                             });

        auto swapchain = device.create_swapchain({
                                                         .native_window = window.get_native_handle(),
                                                         .native_window_platform = window.get_native_platform(),
                                                         // Here we can supply a user-defined surface format selection
                                                         // function, to rate formats. If you don't care what format the
                                                         // swapchain images are in, then you can just omit this argument
                                                         // because it defaults to `daxa::default_format_score(...)`
                                                         .surface_format_selector = [](daxa::Format format) {
                                                             switch (format) {
                                                                 case daxa::Format::R8G8B8A8_UINT:
                                                                     return 100;
                                                                 default:
                                                                     return daxa::default_format_score(format);
                                                             }
                                                         },
                                                         .present_mode = daxa::PresentMode::MAILBOX,
                                                         .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
                                                         .name = "my swapchain",
                                                 });

        auto swapchain_image = swapchain.acquire_next_image();

        auto pipeline_manager = daxa::PipelineManager({
                                                              .device = device,
                                                              .shader_compile_options = {
                                                                      .root_paths = {
                                                                              DAXA_SHADER_INCLUDE_DIR,
                                                                              "./src/shader",
                                                                      },
                                                                      .language = daxa::ShaderLanguage::GLSL,
                                                                      .enable_debug_info = true,
                                                              },
                                                              .name = "my pipeline manager",
                                                      });

        std::shared_ptr<daxa::RasterPipeline> pipeline;
        {
            auto result = pipeline_manager.add_raster_pipeline({
                                                                       .vertex_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{
                                                                               "main.glsl"}},
                                                                       .fragment_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{
                                                                               "main.glsl"}},
                                                                       .color_attachments = {
                                                                               {.format = swapchain.get_format()}},
                                                                       .raster = {},
                                                                       .push_constant_size = sizeof(MyPushConstant),
                                                                       .name = "my pipeline"
                                                               });
            if (result.is_err()) {
                std::cerr << result.message() << std::endl;
                return -1;
            }
            pipeline = result.value();
        }

        auto buffer_id = device.create_buffer({
                                                      .size = sizeof(MyVertex) * 3,
                                                      .name = "my vertex data",
                                              });

        while (!window.should_close()) {
            window.update();
        }

        return 0;
    }
