
# Example usage:
#  mkdir build
#  cd build
#  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..

cmake_minimum_required(VERSION 3.28)
project(ryzen_mon_glgui CXX C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
enable_testing()

# GLFW
set(GLFW_DIR /home/martin/src/glfw) # Set this to point to an up-to-date GLFW repo
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
add_subdirectory(${GLFW_DIR} binary_dir EXCLUDE_FROM_ALL)
include_directories(${GLFW_DIR}/include)

# Dear ImGui
set(IMGUI_DIR /home/martin/src/imgui)
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

# ImPlot
set(IMPLOT_DIR /home/martin/src/implot)
include_directories(${IMPLOT_DIR})

# Ryzen
set(RYZEN_DIR /home/martin/src/ryzen_monitor/src)
include_directories(${RYZEN_DIR} ${RYZEN_DIR}/lib)

# Libraries
find_package(OpenGL REQUIRED)
set(LIBRARIES "glfw;OpenGL")

include_directories(${GLFW_DIR}/deps)

set(sources src/main.cpp
	    src/CpuAffinityManagerBase.cpp
	    src/CpuAffinityManagerWithGui.cpp)

#file(GLOB sources src/*.cpp)

add_executable(${PROJECT_NAME}
	${sources}
	${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
	${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
	${IMGUI_DIR}/imgui.cpp
	${IMGUI_DIR}/imgui_draw.cpp
	${IMGUI_DIR}/imgui_demo.cpp
	${IMGUI_DIR}/imgui_tables.cpp
	${IMGUI_DIR}/imgui_widgets.cpp
	${IMPLOT_DIR}/implot.cpp
	${IMPLOT_DIR}/implot_items.cpp
	${IMPLOT_DIR}/implot_demo.cpp
	${RYZEN_DIR}/lib/libsmu.c
	${RYZEN_DIR}/pm_tables.c
	${RYZEN_DIR}/readinfo.c
	${RYZEN_DIR}/ryzen_monitor.c
	)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PUBLIC -DImTextureID=ImU64)



# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.10.0  # specify the commit hash or tag here
)
FetchContent_MakeAvailable(googletest)

# Define the test executable
add_executable(unit_tests
  tests/test_CpuAffinityManagerBase.cpp
  tests/test_DiagramBase.cpp
  src/CpuAffinityManagerBase.cpp  # Include the source file with actual implementation
  src/DiagramBase.cpp
  )

# Link the test executable with Google Test and the necessary libraries
target_link_libraries(unit_tests gtest_main)
target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Register the test with CTest  
add_test(NAME unit_tests COMMAND unit_tests)  

# Optionally, you can add a custom command to run the tests after building
add_custom_command(
  TARGET unit_tests
  COMMENT "Running unit tests"
  POST_BUILD
  COMMAND unit_tests
)
