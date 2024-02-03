
# Example usage:
#  mkdir build
#  cd build
#  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..

cmake_minimum_required(VERSION 3.28)
project(ryzen_mon_glgui CXX C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# GCC 13, CLANG 14 and MSVC 16.10/VS 2019 all have the {fmt} based std::format 

set(CMAKE_CXX_STANDARD 20)

# Set DEP_DIR with a command line flag
if(NOT DEP_DIR)
	set(DEP_DIR "/home/runner/work/cl-cpp-generator2/cl-cpp-generator2/" CACHE PATH "Path to dependencies directory")
endif()

# Create a CMake Option to enable adress sanitizer
option(ENABLE_ASAN "Enable Address Sanitizer" OFF)
if(ENABLE_ASAN)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

# GLFW
set(GLFW_DIR ${DEP_DIR}/glfw) # Set this to point to an up-to-date GLFW repo
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
add_subdirectory(${GLFW_DIR} binary_dir EXCLUDE_FROM_ALL)
include_directories(${GLFW_DIR}/include)

# Dear ImGui
set(IMGUI_DIR ${DEP_DIR}/imgui)
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

# ImPlot
set(IMPLOT_DIR ${DEP_DIR}/implot)
include_directories(${IMPLOT_DIR})

# Ryzen
set(RYZEN_DIR ${DEP_DIR}/ryzen_monitor/src)
include_directories(${RYZEN_DIR} ${RYZEN_DIR}/lib)

# Libraries
find_package(OpenGL REQUIRED)
set(LIBRARIES "glfw;OpenGL")

include_directories(${GLFW_DIR}/deps)

set(sources 
	src/main.cpp
	src/CpuAffinityManagerBase.cpp
	src/CpuAffinityManagerWithGui.cpp
	src/DiagramBase.cpp
	src/DiagramWithGui.cpp	
)

#file(GLOB sources src/*.cpp)

include(FetchContent)
FetchContent_Declare(
	popl
	GIT_REPOSITORY https://github.com/badaix/popl.git
	GIT_TAG v1.3.0
	)
FetchContent_MakeAvailable(popl)

include_directories(${popl_SOURCE_DIR}/include)

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


# Define the test executable
option(ENABLE_RYZEN_TESTS "Enable unit tests for the ryzen monitor gui" OFF)  # Declare the ENABLE_RYZEN_TESTS option

if(ENABLE_RYZEN_TESTS)
	enable_testing()

	set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")
	include(coverage)
	add_coverage_target("*/tests/*")

	# Fetch Google Test
	include(FetchContent)
	FetchContent_Declare(
		googletest
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG release-1.10.0  # specify the commit hash or tag here
	)
	FetchContent_MakeAvailable(googletest)

	add_executable(unit_tests
		tests/test_CpuAffinityManagerBase.cpp
		src/CpuAffinityManagerBase.cpp  # Include the source file with actual implementation 
		src/DiagramBase.cpp
		tests/test_DiagramBase.cpp
		tests/main.cpp
	)
	# Link the test executable with Google Test and the necessary libraries
	target_link_libraries(unit_tests gtest_main)
	target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

	# Register the test with CTest  
	add_test(NAME unit_tests COMMAND unit_tests)  
endif()