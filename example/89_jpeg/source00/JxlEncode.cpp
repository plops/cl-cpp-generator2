// no preamble

#include <spdlog/spdlog.h>

#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>
#include <vector>

#include "JxlEncode.h"
JxlEncode::JxlEncode() {
  spdlog::info("JxlEncode constructor");
  auto enc{JxlEncoderMake(nullptr)};
  auto runner{JxlThreadParallelRunnerMake(nullptr, 4)};
  if (!((JXL_ENC_SUCCESS) ==
        (JxlEncoderSetParallelRunner(enc.get(), JxlThreadParallelRunner,
                                     runner.get())))) {
    spdlog::info("parallel runner setting failed");
  }
  {
    JxlBasicInfo basic_info;
    JxlEncoderInitBasicInfo(&basic_info);
    (basic_info.xsize) = (512);
    (basic_info.ysize) = (256);
    (basic_info.bits_per_sample) = (32);
    (basic_info.exponent_bits_per_sample) = (8);
    (basic_info.uses_original_profile) = (JXL_FALSE);
    if (!((JXL_ENC_SUCCESS) ==
          (JxlEncoderSetBasicInfo(enc.get(), &basic_info)))) {
      spdlog::info("basic info failed");
    }
    auto *frame_settings{JxlEncoderFrameSettingsCreate(enc.get(), nullptr)};
    auto pixel_format{
        JxlPixelFormat({3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0})};
    if (!((JXL_ENC_SUCCESS) ==
          (JxlEncoderAddImageFrame(frame_settings, &pixel_format, nullptr,
                                   0)))) {
      spdlog::info("adding image frame failed");
    }
    JxlEncoderCloseInput(enc.get());
  }
}
void JxlEncode::Encode() {}
JxlEncode::~JxlEncode() {}