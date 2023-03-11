// sudo pacman -S vulkan-headers vulkan-devel
#include <shaderc/shaderc.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>
using namespace vk;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto info = ApplicationInfo("hello world", 0, nullptr, 0, VK_API_VERSION_1_3);
  auto instance = createInstanceUnique(InstanceCreateInfo({}, &info));
  auto physicalDevice = instance->enumeratePhysicalDevices()[0];
  auto qProps = physicalDevice.getQueueFamilyProperties();
  auto family = 0;
  for (auto qProp : qProps) {
    if (((qProp.queueFlags) & (QueueFlagBits::eCompute))) {
      break;
    }
    family++;
  }
  constexpr float priority[1] = {(1.0F)};
  auto qInfo = DeviceQueueCreateInfo({}, family, 1, priority);
  auto device = physicalDevice.createDeviceUnique(DeviceCreateInfo({}, qInfo));
  const std::string printShader = R"(#version 460
#extension GL_EXT_debug_printf : require

void main ()        {
        debugPrintfEXT("hello from thread %d\n", gl_GlobalInvocationID.x);
}

)";
  auto compiled = shaderc::Compiler().CompileGlslToSpv(
      printShader, shaderc_compute_shader, "hello_world.comp");
  auto spirv = std::vector<uint32_t>(compiled.cbegin(), compiled.cend());
  auto shaderModule =
      device->createShaderModuleUnique(ShaderModuleCreateInfo({}, spirv));
  auto stageInfo = PipelineShaderStageCreateInfo(
      {}, ShaderStageFlagBits::eCompute, *shaderModule, "main");
  auto pipelineLayout =
      device->createPipelineLayoutUnique(PipelineLayoutCreateInfo());
  auto pipelineInfo = ComputePipelineCreateInfo({}, stageInfo, *pipelineLayout);
  auto [status, pipeline] = device->createComputePipelineUnique(
      *(device->createPipelineCacheUnique({})), pipelineInfo);
  auto pool =
      device->createCommandPoolUnique(CommandPoolCreateInfo({}, family));
  auto allocateInfo =
      CommandBufferAllocateInfo(*pool, CommandBufferLevel::ePrimary, 1);
  auto cmdBuffers = device->allocateCommandBuffersUnique(allocateInfo);
  cmdBuffers[0]->begin(CommandBufferBeginInfo{});
  cmdBuffers[0]->bindPipeline(PipelineBindPoint::eCompute, *pipeline);
  cmdBuffers[0]->dispatch(8, 1, 1);
  cmdBuffers[0]->end();
  device->getQueue(family, 0).submit(SubmitInfo({}, {}, *(cmdBuffers[0])));
  device->waitIdle();

  return 0;
}
