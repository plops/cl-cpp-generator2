
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

bool CheckCL(cl_int result, const char *file, int line) {
  switch (result) {
  case CL_SUCCESS: {
    return true;
    break;
  };
  case CL_DEVICE_NOT_FOUND: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: DEVICE_NOT_FOUND") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_DEVICE_NOT_AVAILABLE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: DEVICE_NOT_AVAILABLE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_COMPILER_NOT_AVAILABLE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: COMPILER_NOT_AVAILABLE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_MEM_OBJECT_ALLOCATION_FAILURE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: MEM_OBJECT_ALLOCATION_FAILURE")
                  << (" ") << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_OUT_OF_RESOURCES: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: OUT_OF_RESOURCES") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_OUT_OF_HOST_MEMORY: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: OUT_OF_HOST_MEMORY") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_PROFILING_INFO_NOT_AVAILABLE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: PROFILING_INFO_NOT_AVAILABLE")
                  << (" ") << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_MEM_COPY_OVERLAP: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: MEM_COPY_OVERLAP") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_IMAGE_FORMAT_MISMATCH: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: IMAGE_FORMAT_MISMATCH") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_IMAGE_FORMAT_NOT_SUPPORTED: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: IMAGE_FORMAT_NOT_SUPPORTED") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_BUILD_PROGRAM_FAILURE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: BUILD_PROGRAM_FAILURE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_MAP_FAILURE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: MAP_FAILURE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_MISALIGNED_SUB_BUFFER_OFFSET: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: MISALIGNED_SUB_BUFFER_OFFSET")
                  << (" ") << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ")
                  << ("CL error: EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST")
                  << (" ") << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_VALUE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_VALUE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_DEVICE_TYPE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_DEVICE_TYPE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_PLATFORM: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_PLATFORM") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_DEVICE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_DEVICE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_CONTEXT: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_CONTEXT") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_QUEUE_PROPERTIES: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_QUEUE_PROPERTIES") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_COMMAND_QUEUE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_COMMAND_QUEUE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_HOST_PTR: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_HOST_PTR") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_MEM_OBJECT: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_MEM_OBJECT") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_IMAGE_FORMAT_DESCRIPTOR")
                  << (" ") << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_IMAGE_SIZE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_IMAGE_SIZE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_SAMPLER: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_SAMPLER") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_BINARY: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_BINARY") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_BUILD_OPTIONS: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_BUILD_OPTIONS") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_PROGRAM: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_PROGRAM") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_PROGRAM_EXECUTABLE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_PROGRAM_EXECUTABLE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_KERNEL_NAME: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_KERNEL_NAME") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_KERNEL_DEFINITION: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_KERNEL_DEFINITION") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_KERNEL: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_KERNEL") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_ARG_INDEX: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_ARG_INDEX") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_ARG_VALUE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_ARG_VALUE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_ARG_SIZE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_ARG_SIZE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_KERNEL_ARGS: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_KERNEL_ARGS") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_WORK_DIMENSION: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_WORK_DIMENSION") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_WORK_GROUP_SIZE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_WORK_GROUP_SIZE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_WORK_ITEM_SIZE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_WORK_ITEM_SIZE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_GLOBAL_OFFSET: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_GLOBAL_OFFSET") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_EVENT_WAIT_LIST: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_EVENT_WAIT_LIST") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_EVENT: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_EVENT") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_OPERATION: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_OPERATION") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_GL_OBJECT: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_GL_OBJECT") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_BUFFER_SIZE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_BUFFER_SIZE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_MIP_LEVEL: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_MIP_LEVEL") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  case CL_INVALID_GLOBAL_WORK_SIZE: {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("CL error: INVALID_GLOBAL_WORK_SIZE") << (" ")
                  << (std::setw(8)) << (" file='") << (file) << ("'")
                  << (std::setw(8)) << (" line='") << (line) << ("'")
                  << (std::endl) << (std::flush);
    }
    break;
  };
  }
  return false;
}

cl_device_id getFirstDevice(cl_context context) {
  auto dataSize{static_cast<size_t>(0)};
  auto devices{static_cast<cl_device_id *>(nullptr)};
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &dataSize);
  (devices) = (static_cast<cl_device_id *>(malloc(dataSize)));
  clGetContextInfo(context, CL_CONTEXT_DEVICES, dataSize, devices, nullptr);
  auto first{(devices)[(0)]};
  free(devices);
  return first;
}

cl_int getPlatformID(cl_platform_id *platform) {
  char chBuffer[1024];
  auto num_platforms{static_cast<cl_uint>(0)};
  auto devCount{static_cast<cl_uint>(0)};
  auto ids{static_cast<cl_platform_id *>(nullptr)};
  auto err{static_cast<cl_int>(0)};
  CHECKCL((err) = (clGetPlatformIDs(0, nullptr, &num_platforms)));
  if ((0) == (num_platforms)) {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("no platforms") << (" ") << (std::endl)
                  << (std::flush);
    }
  }
  (ids) = (static_cast<cl_platform_id *>(
      malloc((num_platforms) * (sizeof(cl_platform_id)))));
  (err) = (clGetPlatformIDs(num_platforms, ids, nullptr));
  cl_uint deviceType[2]{{CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU}};
  char *deviceOrder[2][3]{{{"NVIDIA", "AMD", ""}, {"", "", ""}}};
  for (auto i = 0; (i) < (num_platforms); (i) += (1)) {
    CHECKCL((err) = (clGetPlatformInfo((ids)[(i)], CL_PLATFORM_NAME, 1024,
                                       &chBuffer, nullptr)));
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("opencl platform") << (" ") << (std::setw(8))
                  << (" i='") << (i) << ("'") << (std::setw(8))
                  << (" chBuffer='") << (chBuffer) << ("'") << (std::endl)
                  << (std::flush);
    }
  }
  for (auto j = 0; (j) < (2); (j) += (1)) {
    for (auto k = 0; (k) < (3); (k) += (1)) {
      for (auto i = 0; (i) < (num_platforms); (i) += (1)) {
        (err) = (clGetDeviceIDs((ids)[(i)], (deviceType)[(j)], 0, nullptr,
                                &devCount));
        if (((CL_SUCCESS) != (err)) | ((0) == (devCount))) {
          continue;
        }
        CHECKCL((err) = (clGetPlatformInfo((ids)[(i)], CL_PLATFORM_NAME, 1024,
                                           &chBuffer, nullptr)));
        if ((deviceOrder)[(j)][(k)][(0)]) {
          if (!(strstr(chBuffer, (deviceOrder)[(j)][(k)]))) {
            continue;
          }
        }
        {

          auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
          (std::cout) << (std::setw(10))
                      << (std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count())
                      << (" ") << (std::this_thread::get_id()) << (" ")
                      << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ") << ("opencl device") << (" ")
                      << (std::setw(8)) << (" chBuffer='") << (chBuffer)
                      << ("'") << (std::endl) << (std::flush);
        }
        (*platform) = ((ids)[(i)]);
        (j) = (2);
        (k) = (3);
        break;
      }
    }
  }
  free(ids);
  return CL_SUCCESS;
}
