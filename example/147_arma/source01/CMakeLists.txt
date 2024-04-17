
# Example usage:
#  mkdir build
#  cd build
#  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..

cmake_minimum_required(VERSION 3.27)
project(arma CXX C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
	# enable link time optimization
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# GCC 13, CLANG 14 and MSVC 16.10/VS 2019 all have the {fmt} based std::format 

set(CMAKE_CXX_STANDARD 20)

# Create a CMake Option to enable adress sanitizer
option(ENABLE_ASAN "Enable Address Sanitizer" OFF)
if(ENABLE_ASAN)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -march=native -O1")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

# Libraries
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(sources
  src/main.cpp
)

#file(GLOB sources src/*.cpp)

  
# the files  in sources can be merged into one for a unity build
# place them into the same UNITY_GROUP
set_source_files_properties(${sources} PROPERTIES UNITY_GROUP "Sources")

option(ENABLE_POPL "Enable Popl Argument Parser" OFF)
if(ENABLE_POPL)
include(FetchContent)
FetchContent_Declare(
	popl
	GIT_REPOSITORY https://github.com/badaix/popl.git
	GIT_TAG v1.3.0
	)
FetchContent_MakeAvailable(popl)

include_directories(${popl_SOURCE_DIR}/include)
endif()

add_executable(${PROJECT_NAME}
		${sources}
)

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})
#target_compile_definitions(${PROJECT_NAME} PUBLIC -DImTextureID=ImU64)

