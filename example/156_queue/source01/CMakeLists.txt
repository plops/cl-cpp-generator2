cmake_minimum_required(VERSION 3.28)  

include(FetchContent)

project(main)

# add option for ThreadSanitizer
option(THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
if(THREAD_SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -ggdb")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
endif()

FetchContent_Declare(
  readerwriterqueue
  GIT_REPOSITORY    https://github.com/cameron314/readerwriterqueue
  GIT_TAG           master
)

FetchContent_MakeAvailable(readerwriterqueue)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC readerwriterqueue)
