# i don't have glew and implot

# Example usage:
#  mkdir build
#  cd build
#  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..

cmake_minimum_required(VERSION 3.27)
project(ryzen_mon_glgui CXX C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
	# enable link time optimization
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
	add_compile_options(-march=native)
endif()

# GCC 13, CLANG 14 and MSVC 16.10/VS 2019 all have the {fmt} based std::format 

set(CMAKE_CXX_STANDARD 20)


# Set DEP_DIR with a command line flag
if(NOT DEP_DIR)
	set(DEP_DIR "/tmp/deps143/" CACHE PATH "Path to dependencies directory")
endif()

# Create a CMake Option to enable adress sanitizer
option(ENABLE_ASAN "Enable Address Sanitizer" OFF)
if(ENABLE_ASAN)
	add_compile_options(-fsanitize=address)
	add_link_options(-fsanitize=address)
endif()

find_package(GLFW)

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
# the files  in sources can be merged into one for a unity build
# place them into the same UNITY_GROUP
set_source_files_properties(${sources} PROPERTIES UNITY_GROUP "Sources")

set(imgui_sources_backends
	${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
	${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp

)
#set_source_files_properties(${imgui_sources_backends} PROPERTIES UNITY_GROUP "ImGui_backends")

set(imgui_sources_separate
		${IMGUI_DIR}/imgui.cpp
		#${IMGUI_DIR}/imgui_demo.cpp

		${IMGUI_DIR}/imgui_draw.cpp

		${IMGUI_DIR}/imgui_widgets.cpp
		${IMGUI_DIR}/imgui_tables.cpp
)
set_source_files_properties(${imgui_sources_separate} PROPERTIES UNITY_GROUP "ImPlotImGui")

set(implot_sources
		${IMPLOT_DIR}/implot.cpp
		${IMPLOT_DIR}/implot_items.cpp
		#${IMPLOT_DIR}/implot_demo.cpp
)
set_source_files_properties(${implot_sources} PROPERTIES UNITY_GROUP "ImPlotImGui")

set(ryzen_sources
		${RYZEN_DIR}/lib/libsmu.c
		${RYZEN_DIR}/pm_tables.c
		${RYZEN_DIR}/readinfo.c
		${RYZEN_DIR}/ryzen_monitor.c
)
set_source_files_properties(${ryzen_sources} PROPERTIES UNITY_GROUP "Ryzen")
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
		${imgui_sources_backends}
		${imgui_sources_separate}
		${implot_sources}
		${ryzen_sources}
)

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PUBLIC -DImTextureID=ImU64)


# Define the test executable
option(ENABLE_RYZEN_TESTS "Enable unit tests for the ryzen monitor gui" OFF)  # Declare the ENABLE_RYZEN_TESTS option

if(ENABLE_RYZEN_TESTS)
	enable_testing()

	set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")
	#include(coverage)
	#add_coverage_target("*/tests/*")

	# Fetch Google Test
	include(FetchContent)
	FetchContent_Declare(
		googletest
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG v1.14.0  # specify the commit hash or tag here
	)
	FetchContent_MakeAvailable(googletest)

	set(test_sources
		tests/test_CpuAffinityManagerBase.cpp
		tests/test_DiagramBase.cpp
		src/CpuAffinityManagerBase.cpp  # Include the source file with actual implementation
		src/DiagramBase.cpp
	)
	set_source_files_properties(${test_sources} PROPERTIES UNITY_GROUP "Tests")

	add_executable(unit_tests
		${test_sources}
	)    
	
	# Set specific compiler options for the unit_tests target (turn of optimizations for coverage)
    target_compile_options(unit_tests PRIVATE
		-O0 -fprofile-arcs -ftest-coverage --coverage)  
	target_link_options(unit_tests PRIVATE --coverage)  # Link with the coverage library

	# Link the test executable with Google Test and the necessary libraries
	target_link_libraries(unit_tests gtest_main)
	target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

	# Register the test with CTest  
	add_test(NAME unit_tests COMMAND unit_tests)  
endif()
