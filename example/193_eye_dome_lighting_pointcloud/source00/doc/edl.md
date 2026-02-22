High-Performance Point Cloud Visualization: A Modern C++ Implementation of Eye-Dome Lighting
============================================================================================

1\. Introduction to Non-Photorealistic Point Cloud Rendering
------------------------------------------------------------

The visualization of dense, uncolored 3D point cloud data presents a highly specific and computationally demanding challenge within the domain of computer graphics and scientific visualization. Point clouds, typically acquired via Light Detection and Ranging (LiDAR) sensors, photogrammetric scanning arrays, or depth-sensing cameras, frequently consist of millions---or even billions---of discrete vertices. In their most raw and unadulterated format, such as the standard ASCII XYZ format, these datasets possess only spatial coordinates (x,y,z) mapping their position in three-dimensional space. When rendering these dense vertex arrays using standard graphical pipelines, the absence of inherent color data (RGB) and explicitly defined surface normals results in a visually flat, indistinguishable mass of pixels.   

Traditional physical illumination models, such as the Lambertian reflectance model, the Phong illumination model, or modern Physically Based Rendering (PBR) pipelines, require explicit normal vectors to compute the complex interactions between light sources and the geometric surface. To apply these traditional lighting models to a raw XYZ point cloud, a rendering engine must first estimate the normal vector for every single point. This is typically achieved by performing a k-nearest neighbors (k-NN) search to find adjacent points, calculating a local covariance matrix, and extracting the eigenvector corresponding to the smallest eigenvalue. Executing this neighborhood analysis and Principal Component Analysis (PCA) for millions of unstructured points is a computationally expensive operation that severely degrades real-time rendering performance, particularly for massive datasets.   

To resolve this critical limitation, graphics engineers and researchers employ non-photorealistic rendering (NPR) techniques. These techniques are designed to enhance depth perception and spatial comprehension without relying on explicit normal computation or the simulation of physical light transport. Eye-Dome Lighting (EDL) has emerged as the premier image-based shading technique specifically engineered to improve depth perception in the visualization of complex datasets.   

Originally developed by Christian Boucheny during his doctoral research for Electricité de France (EDF), the primary objective of EDL was to facilitate the visualization of massive 3D datasets representing complex industrial facilities. The technique was subsequently integrated into major visualization platforms, including Kitware's ParaView, the web-based Potree viewer developed by Markus Schütz at the Vienna University of Technology, and the open-source CloudCompare software.   

EDL functions entirely as a post-processing pass implemented via OpenGL Shading Language (GLSL) on the Graphics Processing Unit (GPU). It calculates local shading factors based exclusively on the projected depth information captured during the initial geometry rendering phase. This methodology ensures that the computational cost of the illumination scales strictly with the resolution of the screen rather than the geometric complexity of the scene. By bypassing physical light transport, EDL acts as a highly efficient, non-physical shortcut that gracefully accentuates the shapes, contours, and structural undulations of objects at interactive framerates, making it exceptionally well-suited for dense, uncolored point clouds.   

The following comprehensive report details the theoretical foundations, architectural rationale, dependency management, and complete source code implementation for a modular, modern C++ rendering engine utilizing the Eye-Dome Lighting algorithm. The system is designed for Linux environments, adheres strictly to the Almost Always Auto (AAA) idiom, and is prepared for seamless cross-platform deployment.

2\. Theoretical Foundations of the Eye-Dome Lighting Algorithm
--------------------------------------------------------------

### 2.1. Mechanism and Mathematical Formulation

The Eye-Dome Lighting algorithm achieves its visual enhancement by analyzing the spatial depth disparity between a central pixel and its immediate neighbors within screen space. Conceptually, the algorithm treats the hardware depth buffer as a continuous height map. It constructs a localized "dome" or virtual bounding hemisphere around each fragment. If a fragment's depth is significantly closer to the virtual camera than the depth of its neighboring fragments, the algorithm deduces that the fragment lies on an exposed surface, an edge, or a local peak. Conversely, if the fragment is deeper than its neighbors, it is assumed to reside in a crevice or occluded valley, and is thus darkened.   

The localized shading factor, S(p), for a given pixel p, is derived by summing the positive depth differences between p and a defined set of neighboring pixels N(p) situated at a specified radial distance. The fundamental mathematical formulation can be expressed as:

S(p)=∣N(p)∣1​i∈N(p)∑​max(0,Δzi​-τ)

Where:

-   Δzi​=z(p)-z(i) represents the difference in depth.

-   z(p) is the linearized depth value of the central pixel under evaluation.

-   z(i) is the linearized depth value of the neighboring pixel.

-   τ is a user-defined **offset** or threshold value.

The inclusion of the offset parameter (τ) serves a critical purpose. LiDAR scanners and photogrammetric reconstructions inherently produce data containing high-frequency sensor noise. On a flat, planar surface (such as a wall or a street), this micro-noise causes tiny depth disparities between adjacent pixels. Without an offset, the EDL algorithm would amplify this noise, resulting in an overly grainy image. The offset ensures that only depth disparities exceeding a certain geometric threshold contribute to the shadow calculation, effectively acting as a high-pass filter for structural features.   

The resulting sum S(p) quantifies the extent to which the central pixel is occluded. To convert this raw occlusion metric into a visually pleasing, non-photorealistic shadow value, an exponential decay function is applied. This function is modulated by a user-defined **strength** scalar, c:

Shading(p)=exp(-c⋅S(p))

This exponential mapping ensures that small structural changes produce subtle darkening, while steep drop-offs (such as the edge of a building against the ground) produce stark, highly contrasted outlines. The final step in the pipeline involves multiplying this shading coefficient by the base color of the fragment. For uncolored point clouds, a uniform base tint (such as a light gray or white) is applied, allowing the generated shadows to define the entire visual structure of the object.   

### 2.2. Non-Physical Shortcuts and Performance Enhancements

The implementation of EDL represents a deliberate architectural choice to favor perceptual clarity over physical accuracy. The algorithm takes several non-physical shortcuts that drastically simplify the codebase and enhance runtime performance:

1.  **Elimination of Normal Vectors:** As previously established, calculating normal vectors is computationally prohibitive. EDL circumvents this entirely by relying solely on the scalar depth buffer. This reduces the memory footprint of the point cloud by half (omitting normal data) and eliminates the need for complex pre-processing algorithms.   

2.  **Screen-Space Complexity:** Physical rendering complexity scales with O(G), where G is the volume of geometry. Raytracing or complex global illumination requires testing millions of points against light paths. EDL operates purely in screen space, meaning its complexity is O(P), where P is the number of pixels on the screen. Rendering a point cloud with 10 million points takes the exact same post-processing time as rendering a cloud with 100 million points, provided they cover the same screen area.   

3.  **Cross-Filtering over Kernel Convolutions:** A true physical ambient occlusion model samples a dense matrix or a randomized hemispherical kernel around each pixel, requiring dozens of texture fetches per fragment. To maximize performance, Potree and CloudCompare implementations utilize a sparse cross-filter. Instead of sampling a full grid, the shader only samples four orthogonal directions (North, South, East, West) at the specified radius. This reduces the texture fetch overhead significantly while maintaining the illusion of a continuous halo effect.   

### 2.3. Parameterization for Visual Tuning

The flexibility of the EDL algorithm relies on the runtime adjustability of its core parameters. Because point clouds vary wildly in scale---ranging from millimeter-accurate mechanical scans to city-wide aerial surveys---static parameters inevitably fail. An interface is required to manipulate these variables dynamically :   

| Algorithm Parameter | Mathematical Function | Visual Impact on Point Cloud |
| --- | --- | --- |
| **Radius** | Dictates the distance (in screen-space pixels) of the neighborhood N(p) from the central pixel. |

Controls the width and spread of the shadow effect. A larger radius creates thicker, more pronounced outlines, but extreme values may induce cache-miss penalties during texture sampling.

 |
| **Strength** | Acts as the scalar c within the exponential decay function. |

Determines the absolute darkness and contrast of the generated outlines. Higher values produce aggressive, stylized contours.

 |
| **Offset** | Acts as the threshold τ subtracted from Δzi​ prior to the max() function. |

Prevents self-shadowing on rough, planar surfaces by filtering out minor sensor noise, resulting in a cleaner visualization.

 |

3\. Architectural Design and Modern C++ Idioms
----------------------------------------------

### 3.1. The Almost Always Auto (AAA) Paradigm

To construct a robust and highly maintainable rendering engine, the source code strictly adheres to the Almost Always Auto (AAA) style, a modern C++ (C++11 and later) programming idiom. The AAA idiom mandates the use of the `auto` keyword for variable declarations, shifting the developer's focus away from explicit type definitions and toward the initialization expression itself.

The rationale for applying AAA in the context of OpenGL graphics programming is multifaceted and highly advantageous:

1.  **Guaranteed Initialization:** The C++ language permits uninitialized variables (e.g., `int count;`), which can lead to unpredictable state corruption. Using `auto` forces explicit initialization (e.g., `auto count = 0;`), systematically eliminating a vast category of undefined behaviors.

2.  **Mitigation of Implicit Conversions:** The OpenGL API is notorious for its strict and varied numeric types, frequently alternating between `GLuint`, `GLint`, `GLsizei`, `GLfloat`, and standard system integers. Manually specifying these types often leads to silent truncation or signed/unsigned mismatch conversions. Utilizing `auto` alongside specific type casting (e.g., `static_cast<GLsizei>(size)`) forces the compiler to explicitly resolve the correct type, preventing elusive memory corruption or graphical artifacts.

3.  **Enhanced Readability and Refactoring:** It streamlines the codebase by aligning variable names visually. When refactoring complex data structures (such as changing a standard `std::vector` to a customized buffer array), the variable declarations remain untouched, as the compiler automatically deduces the updated return types.

### 3.2. Data Ingestion: Simple XYZ Parsing

The user specifications dictate the use of a simple, straightforward method to read the raw XYZ text files, explicitly avoiding complex optimization research for the parser. While sophisticated rendering engines utilize out-of-core octree structures (such as Potree's proprietary formats) or binary LAS/LAZ formats to stream massive datasets from disk, a simple standard library approach is utilized here to maintain a minimal codebase footprint.   

The architecture relies on standard `std::ifstream` and `std::istringstream` to read the file sequentially. The parser iterates line by line, extracting the three floating-point values into a standard `std::vector<glm::vec3>`.

A critical step executed during this parsing phase is spatial normalization. LiDAR and GIS datasets are frequently georeferenced, meaning their absolute coordinates map to global positioning systems (e.g., coordinates in the millions of meters). If these massive floating-point numbers are passed directly to the GPU, the 32-bit floating-point precision of the OpenGL projection matrix will fail, resulting in catastrophic depth buffer inaccuracy known as Z-fighting. To prevent this, the parser simultaneously calculates the bounding box of the dataset, determines the central geometric centroid, and translates the entire point cloud to the local origin (0,0,0) before transferring the data to the GPU memory.   

4\. Dependency Management and Build Environment Configuration
-------------------------------------------------------------

Constructing a reliable build pipeline on Linux necessitates a robust toolchain capable of resolving multiple external graphical libraries. The project utilizes CMake as its build generator. Per the architectural constraints, the required dependencies---GLFW, Dear ImGui, GLAD, and GLM---are located in local user directories (`~/src/glfw`, `~/src/imgui`, `~/src/glad`, and `~/src/glm`) rather than global system paths.

### 4.1. GLAD Online Generator Configuration

GLAD functions as an OpenGL loader-generator based on the official Khronos XML specifications. Because OpenGL is heavily dependent on the specific graphics driver installed on the host Linux machine, its functions cannot be linked directly at compile time. Instead, GLAD dynamically queries the driver at runtime to map function pointers.   

Unlike monolithic libraries, GLAD requires the developer to generate a specific, minimized subset of the OpenGL API tailored strictly to the project's requirements.   

**Exact Generator Instructions for EDL Compliance:**

1.  Navigate to the official GLAD online service at `https://glad.dav1d.de/`.   

2.  **Language:** Select `C/C++`.

3.  **Specification:** Select `OpenGL`.

4.  **API:** Set `gl` to `Version 3.3` or higher (4.1 is recommended for broader modern compatibility). Version 3.3 is the absolute minimum required to natively support GLSL 330, Vertex Array Objects (VAOs), and Framebuffer Objects (FBOs) without relying on legacy extensions.

5.  **Profile:** Strictly select `Core`. The Core profile permanently disables the deprecated fixed-function pipeline (e.g., `glBegin`, `glEnd`, `glLight`), forcing the application to use the modern programmable shader pipeline necessary for EDL.

6.  **Extensions:** While FBOs and depth textures are integrated into the Core profile in versions 3.0 and above, for maximum safety across diverse Linux display servers, ensure the extensions `GL_ARB_framebuffer_object` and `GL_ARB_depth_texture` are intrinsically supported by the generated loader.   

7.  **Options:** Ensure the `Generate a loader` option is explicitly ticked. As stated in the GLAD documentation, this is required unless an external windowing library specifically handles the `GetProcAddress` functionality natively [User Query].

8.  Click **Generate** and download the resulting archive.

9.  Extract the contents. The resulting file structure should match the requested configuration: glad ├── src │ └── gl.c └── include ├── glad │ └── gl.h └── KHR └── khrplatform.h Place this entire structure into the `~/src/glad` directory.

As explicitly outlined in the provided documentation, a raw compilation using standard GCC would take the form: `gcc src/main.c glad/src/gl.c -Iglad/include -lglfw -ldl` This manual compilation string highlights the necessity of linking the `dl` (dynamic linking) library on Linux to allow GLAD to query the driver pointers.

### 4.2. Resolution of Local Dependencies via CMake

While the manual GCC string is functional for simple scripts, a scalable C++ project requires CMake. Standard `find_package` commands often fail when libraries are compiled locally and not formally installed into `/usr/local/lib` or `/usr/lib`. To guarantee a successful build utilizing the specific `~/src/*` pathways, the `CMakeLists.txt` file must employ specific directory inclusions.   

The strategy involves compiling GLAD and Dear ImGui directly from their source files as static libraries integrated into the CMake tree. GLFW, utilizing its own CMake framework, is included via `add_subdirectory`, while GLM, being entirely header-only, requires only an include directory declaration.   

CMake

```
cmake_minimum_required(VERSION 3.10)
project(PointcloudEDL CXX)

# Enforce Modern C++ Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Expand the home directory path to locate dependencies
set(SRC_DIR "$ENV{HOME}/src")

# 1. GLM (Header-only mathematics library)
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${SRC_DIR}/glm)

# 2. GLAD (OpenGL loader generated via web interface)
add_library(glad STATIC ${SRC_DIR}/glad/src/gl.c)
target_include_directories(glad PUBLIC ${SRC_DIR}/glad/include)

# 3. GLFW (Window and context management)
# We disable GLFW's tests and documentation to vastly accelerate build times
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(${SRC_DIR}/glfw glfw_build)

# 4. Dear ImGui (Immediate mode graphical user interface)
set(IMGUI_DIR ${SRC_DIR}/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
# ImGui backends require linking against GLFW
target_link_libraries(imgui glfw)

# Find System OpenGL dynamically based on the Linux distribution
find_package(OpenGL REQUIRED)

# Define Main Executable
add_executable(PointcloudEDL main.cpp)

# Link all dependencies to the main executable
# CMAKE_DL_LIBS is strictly required on Linux for GLAD to dynamically load OpenGL pointers
target_link_libraries(PointcloudEDL PRIVATE
    glad
    glfw
    glm
    imgui
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)

```

This configuration achieves strict modularity. By compiling GLAD and ImGui as localized static targets, the project bypasses complex Linux package manager discrepancies, ensuring identical compilation behavior across Fedora, Ubuntu, or Arch-based distributions.   

5\. The OpenGL Rendering Pipeline Architecture
----------------------------------------------

The implementation of Eye-Dome Lighting requires a multi-stage rendering pipeline. Modern graphics pipelines diverge from simply drawing directly to the monitor; instead, they utilize an off-screen pass to gather spatial data, followed by a post-processing pass to compute the visual effects.

### 5.1. Pass 1: The Geometry and Depth Generation Phase

The initial stage requires redirecting the output of the GPU away from the default physical screen (the backbuffer) into an off-screen memory construct known as a Framebuffer Object (FBO).   

For the EDL algorithm to function, this FBO must be configured with a specific depth attachment. While color information is absent in the base XYZ file, a color texture is attached to the FBO to store a flat base albedo (the default color of the points) and to differentiate between actual geometry and empty space.

The depth buffer allocation is the most critical technical nuance in this phase. The buffer must be allocated using the `GL_DEPTH_COMPONENT32F` format. A 32-bit floating-point depth texture is absolutely mandatory. Standard 16-bit or 24-bit integer depth formats lack the sub-pixel precision necessary to differentiate the minute undulations on dense point cloud surfaces. If insufficient depth precision is allocated, the Δzi​ calculations in the EDL shader will aggressively round to zero, obliterating the shading effect and leaving the dataset appearing flat.   

During this pass, the `glDrawArrays(GL_POINTS,...)` command is invoked. Each spatial coordinate is rasterized as a primitive point (whose scale can be adjusted via `glPointSize`). The fragment shader outputs a uniform color, but crucially, OpenGL's internal pipeline automatically calculates the fragment's distance from the camera and writes this value directly into the 32-bit floating-point `GL_DEPTH_ATTACHMENT` texture.   

### 5.2. Pass 2: EDL Post-Processing and Depth Linearization

Once the FBO is fully populated with the spatial map of the point cloud, the pipeline unbinds the FBO, redirecting all subsequent rendering back to the default monitor buffer. The engine then renders a single, screen-spanning quadrilateral (two triangles that cover the entire viewport). The previously generated depth texture and color texture are bound to the GPU as inputs (`sampler2D`) for the EDL fragment shader.   

#### The Necessity of Depth Linearization

The Z-values automatically generated and stored by OpenGL's perspective projection matrix are non-linear. The mathematics of frustum projection dictate that precision is heavily biased toward the near clipping plane to prevent Z-fighting in close-up objects, leaving far objects with highly compressed depth values.

To accurately compute absolute spatial disparities for the EDL shadow calculation, these hyperbolic depth values must be mathematically transformed back into a linear space (Eye Space) within the shader.   

The mathematical linearization relies on converting the fragment back to Normalized Device Coordinates (NDC) and applying the camera's near (n) and far (f) clipping planes:

zndc​=2.0⋅depth-1.0

zlinear​=f+n-zndc​⋅(f-n)2.0⋅n⋅f​

Only after the depth is linearized can the shader accurately calculate the spatial offsets for the surrounding neighborhood.

#### Shader Execution and Edge Detection

Once the linear depth is established for the central pixel, the shader samples the depth texture at neighboring coordinates. By utilizing the aforementioned non-physical cross-filter shortcut, the shader tests four orthogonal pixels at a distance dictated by the user's `radius` parameter.   

If a sampled neighboring fragment indicates empty space (i.e., its raw depth corresponds to the clear background, typically a value of `1.0`), the shader assigns an artificially massive depth penalty. This specific shortcut guarantees that the outer silhouettes and extreme boundaries of the point cloud generate thick, dark borders, separating the dataset vividly from the background skybox.   

The shader aggregates the valid spatial disparities, subtracts the `offset` parameter to filter out planar noise, calculates the average occlusion, applies the exponential decay via the `strength` parameter, and finally darkens the base color to render the final frame.

6\. Runtime Interaction: Dear ImGui Integration
-----------------------------------------------

Eye-Dome Lighting is an algorithm highly sensitive to the scale of the dataset and the viewer's current camera distance. A radius and strength parameter configuration that perfectly highlights structural micro-cracks in a close-up scan of a masonry wall will often appear entirely black or overly cluttered when the camera pulls back to view an entire city landscape.

Therefore, hardcoding these algorithmic variables is insufficient. Exposing them via a dynamic Graphical User Interface (GUI) is a mandatory architectural design pattern for any functional point cloud viewer.   

The Dear ImGui framework is leveraged explicitly for its immediate-mode paradigm. Unlike retained-mode interfaces (such as Qt or GTK), ImGui requires no complex stateful synchronization or event callbacks. During the main rendering loop, an ImGui overlay is seamlessly pushed to the pipeline. Sliders for `EDL Strength`, `EDL Radius`, `EDL Offset`, and `Point Size` are mapped directly to local C++ variables.

Immediately preceding the execution of the EDL full-screen pass, these floating-point variables are pushed to the GPU via `glUniform1f()` commands. This provides the user with instantaneous, real-time visual feedback, allowing them to fine-tune the algorithmic thresholds without requiring expensive shader recompilations or application restarts.   

7\. Complete Source Code Implementation
---------------------------------------

The following section comprises the entirety of the C++ source code necessary to compile the application. It utilizes the AAA style, incorporates the XYZ file parser, establishes the two-pass rendering pipeline, and defines the integrated EDL GLSL shaders featuring the strength, radius, and offset parameters.

To execute this, ensure the `CMakeLists.txt` is configured exactly as detailed in Section 4.2, and place an uncolored ASCII point cloud file named `cloud.txt` (formatted simply as `X Y Z` per line) in the same directory as the compiled executable.

### 7.1. `main.cpp` Source Code

C++

```
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>

// ============================================================================
// SHADER SOURCE CODE DEFINITIONS
// ============================================================================

// 1. Geometry Pass Vertex Shader
// Projects the raw 3D points into screen space.
const auto geometryVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(aPos, 1.0);
}
)";

// 2. Geometry Pass Fragment Shader
// Outputs a flat base color. The vital depth data is written automatically
// by the OpenGL pipeline into the attached depth texture.
const auto geometryFS = R"(
#version 330 core
layout (location = 0) out vec4 FragColor;
uniform vec3 baseColor;

void main() {
    FragColor = vec4(baseColor, 1.0);
}
)";

// 3. Post-Processing Vertex Shader (Fullscreen Quad)
// Renders a flat quad covering the entire screen to host the post-processing.
const auto edlVS = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main() {
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}
)";

// 4. Post-Processing Fragment Shader (Eye-Dome Lighting)
// Implements the depth-disparity non-photorealistic shading logic.
const auto edlFS = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D colorTexture;
uniform sampler2D depthTexture;

uniform float edlStrength;
uniform float edlRadius;
uniform float edlOffset;
uniform vec2 resolution;
uniform float zNear;
uniform float zFar;

// Converts hyperbolic hardware depth back into linear eye-space depth
float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * zNear * zFar) / (zFar + zNear - z * (zFar - zNear));
}

void main() {
    float rawDepth = texture(depthTexture, TexCoords).r;

    // If the pixel is the clear color (depth == 1.0), render the background
    if (rawDepth >= 1.0) {
        FragColor = vec4(0.15, 0.15, 0.15, 1.0); // Dark gray background
        return;
    }

    vec4 baseColor = texture(colorTexture, TexCoords);
    float depth = linearizeDepth(rawDepth);

    vec2 texelSize = 1.0 / resolution;

    // Orthogonal cross-filter sampling to optimize texture fetches
    vec2 offsets = vec2(
        vec2(edlRadius, 0.0),
        vec2(-edlRadius, 0.0),
        vec2(0.0, edlRadius),
        vec2(0.0, -edlRadius)
    );

    float sum = 0.0;

    for(int i = 0; i < 4; i++) {
        vec2 sampleCoords = TexCoords + offsets[i] * texelSize;
        float neighborRawDepth = texture(depthTexture, sampleCoords).r;

        // Massive penalty for borders against the skybox to create strong outlines
        if (neighborRawDepth >= 1.0) {
            sum += 10.0;
        } else {
            float neighborDepth = linearizeDepth(neighborRawDepth);
            // Calculate positive depth disparities, applying the noise-reduction offset
            sum += max(0.0, depth - neighborDepth - edlOffset);
        }
    }

    // Average the occlusion and apply exponential decay
    float factor = sum / 4.0;
    float shade = exp(-factor * 300.0 * edlStrength);

    FragColor = vec4(baseColor.rgb * shade, baseColor.a);
}
)";

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

auto compileShader(GLenum type, const char* source) -> GLuint {
    auto shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    auto success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        auto infoLog = std::string(512, '\0');
        glGetShaderInfoLog(shader, 512, nullptr, infoLog.data());
        std::cerr << "Shader Compilation Failed:\n" << infoLog << std::endl;
    }
    return shader;
}

auto createProgram(const char* vsSource, const char* fsSource) -> GLuint {
    auto vs = compileShader(GL_VERTEX_SHADER, vsSource);
    auto fs = compileShader(GL_FRAGMENT_SHADER, fsSource);

    auto program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

// Parses raw XYZ text data and translates the bounding volume to the origin
auto loadPointCloud(const std::string& filepath, glm::vec3& outCenter) -> std::vector<glm::vec3> {
    auto points = std::vector<glm::vec3>{};
    auto file = std::ifstream{filepath};

    if (!file.is_open()) {
        std::cerr << "Warning: Failed to open point cloud file: " << filepath << std::endl;
        return points;
    }

    auto line = std::string{};
    auto minBounds = glm::vec3(std::numeric_limits<float>::max());
    auto maxBounds = glm::vec3(std::numeric_limits<float>::lowest());

    while (std::getline(file, line)) {
        auto ss = std::istringstream{line};
        auto p = glm::vec3{};
        if (ss >> p.x >> p.y >> p.z) {
            points.push_back(p);
            minBounds = glm::min(minBounds, p);
            maxBounds = glm::max(maxBounds, p);
        }
    }

    outCenter = (minBounds + maxBounds) * 0.5f;
    return points;
}

// ============================================================================
// MAIN APPLICATION
// ============================================================================

int main() {
    // 1. Initialize GLFW Context
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    auto window = glfwCreateWindow(1280, 720, "Eye-Dome Lighting Point Cloud Viewer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable V-Sync

    // 2. Initialize GLAD Function Pointers
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // 3. Load Data & Normalize Coordinates
    auto center = glm::vec3(0.0f);
    auto points = loadPointCloud("cloud.txt", center);

    if(points.empty()) {
        std::cerr << "Generating dummy geometry for demonstration purposes." << std::endl;
        for(auto i = 0; i < 10000; i++) {
            points.push_back(glm::vec3(
                (rand() % 100) / 10.0f - 5.0f,
                (rand() % 100) / 10.0f - 5.0f,
                (rand() % 100) / 10.0f - 5.0f
            ));
        }
    } else {
        // Translation to origin prevents Z-fighting in georeferenced data
        for(auto& p : points) {
            p -= center;
        }
        std::cout << "Successfully loaded " << points.size() << " points." << std::endl;
    }

    // 4. Setup Geometry VAO/VBO for the Point Cloud
    auto vao = GLuint{};
    auto vbo = GLuint{};
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(glm::vec3), points.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
    glEnableVertexAttribArray(0);

    // Setup Fullscreen Quad for Post-Processing Pass
    float quadVertices = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    auto quadVAO = GLuint{};
    auto quadVBO = GLuint{};
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    // 5. Allocate the Framebuffer Object (FBO)
    auto fbo = GLuint{};
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    auto colorTex = GLuint{};
    glGenTextures(1, &colorTex);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1280, 720, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);

    auto depthTex = GLuint{};
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    // Crucial: 32-bit float requirement for depth evaluation precision
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 1280, 720, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER)!= GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Critical Error: Framebuffer architecture incomplete!" << std::endl;
        return -1;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 6. Compile Pipeline Shaders
    auto geomProgram = createProgram(geometryVS, geometryFS);
    auto edlProgram = createProgram(edlVS, edlFS);

    // 7. Initialize Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // Configurable Runtime Parameters
    auto edlStrength = 0.8f;
    auto edlRadius = 1.5f;
    auto edlOffset = 0.001f;
    auto pointSize = 3.0f;
    auto baseColor = glm::vec3(0.7f, 0.8f, 0.9f);

    auto cameraDist = 20.0f;
    auto rotationY = 0.0f;
    auto rotationX = 0.5f;

    auto zNear = 0.1f;
    auto zFar = 1000.0f;

    // 8. Primary Render Loop
    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        auto width = 0;
        auto height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if(width == 0 |

| height == 0) continue;

        glViewport(0, 0, width, height);

        // Dynamically reallocate FBO textures if the user resizes the window
        glBindTexture(GL_TEXTURE_2D, colorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, depthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

        // ====================================================================
        // PASS 1: Geometry Rendering
        // ====================================================================
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        // Clear color to pure white representing infinite depth space (1.0)
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(geomProgram);

        auto proj = glm::perspective(glm::radians(60.0f), static_cast<float>(width)/height, zNear, zFar);
        auto view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -cameraDist));
        view = glm::rotate(view, rotationX, glm::vec3(1.0f, 0.0f, 0.0f));
        view = glm::rotate(view, rotationY, glm::vec3(0.0f, 1.0f, 0.0f));
        auto mvp = proj * view;

        glUniformMatrix4fv(glGetUniformLocation(geomProgram, "mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform3fv(glGetUniformLocation(geomProgram, "baseColor"), 1, glm::value_ptr(baseColor));

        glPointSize(pointSize);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(points.size()));

        // ====================================================================
        // PASS 2: Post-Processing (Eye-Dome Lighting)
        // ====================================================================
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(edlProgram);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorTex);
        glUniform1i(glGetUniformLocation(edlProgram, "colorTexture"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthTex);
        glUniform1i(glGetUniformLocation(edlProgram, "depthTexture"), 1);

        // Push ImGui tuned variables to the GPU
        glUniform1f(glGetUniformLocation(edlProgram, "edlStrength"), edlStrength);
        glUniform1f(glGetUniformLocation(edlProgram, "edlRadius"), edlRadius);
        glUniform1f(glGetUniformLocation(edlProgram, "edlOffset"), edlOffset);
        glUniform2f(glGetUniformLocation(edlProgram, "resolution"), static_cast<float>(width), static_cast<float>(height));
        glUniform1f(glGetUniformLocation(edlProgram, "zNear"), zNear);
        glUniform1f(glGetUniformLocation(edlProgram, "zFar"), zFar);

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // ====================================================================
        // PASS 3: Immediate Mode UI Overlay
        // ====================================================================
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Eye-Dome Configuration");
        ImGui::Text("Non-Photorealistic Shading Settings");
        ImGui::Separator();
        ImGui::SliderFloat("EDL Strength", &edlStrength, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("EDL Radius", &edlRadius, 0.0f, 10.0f, "%.1f px");
        ImGui::SliderFloat("EDL Offset", &edlOffset, 0.0f, 0.01f, "%.4f");

        ImGui::Spacing();
        ImGui::Text("Geometry Settings");
        ImGui::Separator();
        ImGui::SliderFloat("Point Size", &pointSize, 1.0f, 15.0f, "%.1f px");
        ImGui::ColorEdit3("Base Color", glm::value_ptr(baseColor));

        ImGui::Spacing();
        ImGui::Text("Camera Settings");
        ImGui::Separator();
        ImGui::SliderFloat("Zoom Distance", &cameraDist, 1.0f, 200.0f);
        ImGui::SliderFloat("Yaw Rotation", &rotationY, 0.0f, 6.28f);
        ImGui::SliderFloat("Pitch Rotation", &rotationX, -1.5f, 1.5f);

        ImGui::Spacing();
        ImGui::Text("Performance: %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Safely deallocate resources
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteFramebuffers(1, &fbo);

    glfwTerminate();
    return 0;
}

```

8\. Conclusion
--------------

The comprehensive methodology delineated in this report establishes a fully optimized, modular architecture for deploying Eye-Dome Lighting within a modern C++ 17 graphical environment. By acknowledging the computational limitations of calculating traditional geometric normals for millions of scattered points, the integration of an image-based, non-photorealistic post-processing pass provides immense performance benefits.

The mechanism circumvents physical light transport rules, utilizing localized depth disparities driven by user-defined radius, strength, and offset thresholding matrices to force visual contrasts along geometric contours. Furthermore, utilizing the AAA coding paradigm alongside modular, localized CMake dependency resolution circumvents structural vulnerabilities. Integrating the Dear ImGui interface ensures that the user maintains complete, real-time control over the algorithmic parameters, guaranteeing optimal visual clarity regardless of the structural density or scale of the provided XYZ point cloud.   

[

![](https://t0.gstatic.com/faviconV2?url=https://www.mathworks.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

mathworks.com

pointCloud - Object for storing 3-D point cloud - MATLAB - MathWorks

Opens in a new window](https://www.mathworks.com/help/vision/ref/pointcloud.html)[

![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

mdpi.com

Impact-Detection Algorithm That Uses Point Clouds as Topographic Inputs for 3D Rockfall Simulations - MDPI

Opens in a new window](https://www.mdpi.com/2076-3263/11/5/188)[

![](https://t0.gstatic.com/faviconV2?url=https://www.kitware.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

kitware.com

Eye-Dome Lighting: a non-photorealistic shading technique - Kitware, Inc.

Opens in a new window](https://www.kitware.com/eye-dome-lighting-a-non-photorealistic-shading-technique/)[

![](https://t3.gstatic.com/faviconV2?url=https://viscircle.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

viscircle.com

What you should know about the Eye Dome Lighting. - VisCircle

Opens in a new window](https://viscircle.com/eye-dome-lighting-a-non-photorealistic-shading-technique/)[

![](https://t0.gstatic.com/faviconV2?url=http://potree.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

potree.org

Potree Viewer

Opens in a new window](http://potree.org/demo/potree_1.5/examples/lion_laz.html)[

![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

researchgate.net

Eye-Dome Lighting: a non-photorealistic shading technique - ResearchGate

Opens in a new window](https://www.researchgate.net/publication/320616607_Eye-Dome_Lighting_a_non-photorealistic_shading_technique)[

![](https://t2.gstatic.com/faviconV2?url=https://dev.epicgames.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

dev.epicgames.com

Eye-Dome Lighting Mode for Point Clouds in Unreal Engine - Epic Games Developers

Opens in a new window](https://dev.epicgames.com/documentation/en-us/unreal-engine/eye-dome-lighting-mode-for-point-clouds-in-unreal-engine)[

![](https://t0.gstatic.com/faviconV2?url=https://dev.luciad.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

dev.luciad.com

How to enhance depth perception in point cloud data - Luciad Developer Platform

Opens in a new window](https://dev.luciad.com/portal/productDocumentation/LuciadFusion/docs/articles/howto/view/eye_dome_lighting.html?subcategory=lls_ogc3dtiles)[

![](https://t0.gstatic.com/faviconV2?url=https://dev.luciad.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

dev.luciad.com

EyeDomeLightingEffect (LuciadCPillar Java Android API documentation)

Opens in a new window](https://dev.luciad.com/portal/productDocumentation/LuciadCPillar/docs/reference/android/com/luciad/maps/effects/EyeDomeLightingEffect.html)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

github.com

potree/src/materials/shaders/edl.fs at develop - GitHub

Opens in a new window](https://github.com/potree/potree/blob/develop/src/materials/shaders/edl.fs)[

![](https://t2.gstatic.com/faviconV2?url=https://publicdownload.qps.nl/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

publicdownload.qps.nl

Fledermaus 8.6 Documentation - QPS

Opens in a new window](https://publicdownload.qps.nl/Fledermaus/Fledermaus-8.6.0-Documentation.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://docs.qgis.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

docs.qgis.org

5\. QGIS Configuration --- QGIS Documentation documentation - QGIS resources

Opens in a new window](https://docs.qgis.org/latest/en/docs/user_manual/introduction/qgis_configuration.html)[

![](https://t3.gstatic.com/faviconV2?url=https://www.esri.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

esri.com

Eye-Dome Lighting Enhanced Point Cloud Rendering in ArcGIS Pro - Esri

Opens in a new window](https://www.esri.com/arcgis-blog/products/arcgis-pro/3d-gis/eye-dome-lighting)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

github.com

BA_PointCloud/PointCloudRenderer/Assets/Resources/Shaders/EDL.shader at master - GitHub

Opens in a new window](https://github.com/SFraissTU/BA_PointCloud/blob/master/PointCloudRenderer/Assets/Resources/Shaders/EDL.shader)[

![](https://t2.gstatic.com/faviconV2?url=https://www.cg.tuwien.ac.at/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

cg.tuwien.ac.at

Rendering Large Point Clouds in Unity - Research Unit of Computer Graphics | TU Wien

Opens in a new window](https://www.cg.tuwien.ac.at/research/publications/2017/FRAISS-2017-PCU/FRAISS-2017-PCU-thesis.pdf)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

github.com

DouglasViolante/Simple-PointCloud-Viewer: A basic point cloud viewer, made with C++ and powered by OpenGL 4. - GitHub

Opens in a new window](https://github.com/DouglasViolante/Simple-PointCloud-Viewer)[

![](https://t0.gstatic.com/faviconV2?url=https://www.mass.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

mass.gov

Development of a Visualization, Sharing, and Processing Platform for Large-Scale Highway Point Cloud Data - Mass.gov

Opens in a new window](https://www.mass.gov/doc/development-of-a-visualization-sharing-and-processing-platform-for-large-scale-highway-point-cloud-data-final-report/download)[

![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

reddit.com

Online GL/EGL/GLX/WGL loader generator, based on Glad (and the official specs) - Reddit

Opens in a new window](https://www.reddit.com/r/opengl/comments/35hx7g/online_gleglglxwgl_loader_generator_based_on_glad/)[

![](https://t2.gstatic.com/faviconV2?url=https://glad.dav1d.de/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

glad.dav1d.de

Glad

Opens in a new window](https://glad.dav1d.de/)[

![](https://t2.gstatic.com/faviconV2?url=https://community.roonlabs.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

community.roonlabs.com

GL_ARB_framebuffer_object OpenGL extension is required - Roon Labs Community

Opens in a new window](https://community.roonlabs.com/t/gl-arb-framebuffer-object-opengl-extension-is-required/3860)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

stackoverflow.com

cmake find_package cannot locate glfw after build and install - Stack Overflow

Opens in a new window](https://stackoverflow.com/questions/40076082/cmake-find-package-cannot-locate-glfw-after-build-and-install)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

github.com

CMake project for GLFW, Glad, ImGui and glm - GitHub

Opens in a new window](https://github.com/cmmw/imgui-glfw-glad-glm)[

![](https://t3.gstatic.com/faviconV2?url=https://discourse.glfw.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

discourse.glfw.org

How to build program with glfw compiled locally? - support

Opens in a new window](https://discourse.glfw.org/t/how-to-build-program-with-glfw-compiled-locally/1109)[

![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

reddit.com

I am using Linux and am trying to make a window with OpenGl, GLFW, and imgui but I am struggling with cmake - Reddit

Opens in a new window](https://www.reddit.com/r/cmake/comments/1bjw1ns/i_am_using_linux_and_am_trying_to_make_a_window/)[

![](https://t0.gstatic.com/faviconV2?url=https://learnopengl.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

learnopengl.com

Framebuffers - LearnOpenGL

Opens in a new window](https://learnopengl.com/Advanced-OpenGL/Framebuffers)[

![](https://t0.gstatic.com/faviconV2?url=https://learnopengl.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

learnopengl.com

Depth testing - LearnOpenGL

Opens in a new window](https://learnopengl.com/Advanced-OpenGL/Depth-testing)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

stackoverflow.com

Sample depth buffer in OpenGL compute shader - glsl - Stack Overflow

Opens in a new window](https://stackoverflow.com/questions/21384972/sample-depth-buffer-in-opengl-compute-shader)[

![](https://t3.gstatic.com/faviconV2?url=https://community.khronos.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

community.khronos.org

depth buffer and lighting - OpenGL: Basic Coding - Khronos Forums

Opens in a new window](https://community.khronos.org/t/depth-buffer-and-lighting/73015)[

![](https://t3.gstatic.com/faviconV2?url=https://community.khronos.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

community.khronos.org

sampling the depth buffer - OpenGL: Advanced Coding - Khronos Forums

Opens in a new window](https://community.khronos.org/t/sampling-the-depth-buffer/46549)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

stackoverflow.com

draw the depth value in opengl using shaders - Stack Overflow

Opens in a new window](https://stackoverflow.com/questions/6408851/draw-the-depth-value-in-opengl-using-shaders)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

github.com

Implement Eye Dome Lighting post-processing effect - Issue #6633 - godotengine/godot-proposals - GitHub

Opens in a new window](https://github.com/godotengine/godot-proposals/issues/6633)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

stackoverflow.com

How can i link GLFW and Dear ImGUI in CMAKE - c++ - Stack Overflow

Opens in a new window](https://stackoverflow.com/questions/68014365/how-can-i-link-glfw-and-dear-imgui-in-cmake)[

![](https://t1.gstatic.com/faviconV2?url=https://docs.opendronemap.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

docs.opendronemap.org

Tutorials --- OpenDroneMap 3.6.0 documentation

Opens in a new window](https://docs.opendronemap.org/tutorials/)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

github.com

AIS-Bonn/easy_pbr: Lightweight Physically-Based Renderer designed for ease of use and fast prototyping - GitHub

Opens in a new window](https://github.com/AIS-Bonn/easy_pbr)

[

![](https://t0.gstatic.com/faviconV2?url=https://dev.luciad.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://dev.luciad.com/portal/productDocumentation/LuciadRIA/docs/reference/LuciadRIA/interfaces/_luciad_ria_view_EyeDomeLightingEffect.EyeDomeLightingEffect.html)[

![](https://t1.gstatic.com/faviconV2?url=https://docs.pyvista.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://docs.pyvista.org/examples/02-plot/edl.html)[

![](https://t0.gstatic.com/faviconV2?url=https://dev.luciad.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://dev.luciad.com/portal/productDocumentation/LuciadFusion/docs/articles/howto/vectorpainting/pointcloud.html)[

![](https://t3.gstatic.com/faviconV2?url=https://ieeexplore.ieee.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://ieeexplore.ieee.org/iel8/2945/11151252/10972035.pdf)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/RobotLocomotion/director/blob/master/src/vtk/DRCFilters/edl_resources/Shaders/edl_shade.glsl)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/RaduAlexandru/easy_pbr)[

![](https://t0.gstatic.com/faviconV2?url=https://dev.to/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://dev.to/masanori_msl/using-potree-from-code-written-in-typescript-1-2p41)[

![](https://t3.gstatic.com/faviconV2?url=https://bcmpinc.wordpress.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://bcmpinc.wordpress.com/2015/08/18/creating-an-opengl-4-5-context-using-sdl2-and-glad/)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/30302966/opengl-4-5-buffer-texture-extensions-support)[

![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.reddit.com/r/cmake/comments/tgm6ys/learning_cmake_everything_working_but_failing_to/)[

![](https://t0.gstatic.com/faviconV2?url=https://www.youtube.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.youtube.com/watch?v=LTzXGBv5PLg)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/tentone/potree-core)[

![](https://t3.gstatic.com/faviconV2?url=https://community.khronos.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://community.khronos.org/t/glsl-point-light-shader/65735)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/ocornut/imgui/discussions/4915)[

![](https://t3.gstatic.com/faviconV2?url=https://discourse.glfw.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://discourse.glfw.org/t/glfw-on-linux-cmake-finds-no-libraries/2733)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/Dav1dde/glad)[

![](https://t0.gstatic.com/faviconV2?url=https://360-for-you.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://360-for-you.com/static/potree/potree_guide.pdf)[

![](https://t3.gstatic.com/faviconV2?url=https://forum.processing.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://forum.processing.org/one/topic/loading-a-huge-xyz-point-cload-file.html)[

![](https://t0.gstatic.com/faviconV2?url=https://www.youtube.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.youtube.com/watch?v=5jfYdqWYFCM)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/ocornut/imgui/discussions/7565)[

![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.researchgate.net/publication/309358171_Potree_Rendering_Large_Point_Clouds_in_Web_Browsers)[

![](https://t3.gstatic.com/faviconV2?url=https://etheses.whiterose.ac.uk/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://etheses.whiterose.ac.uk/id/eprint/38109/1/JS_thesis_final.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://cesium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://cesium.com/downloads/cesiumjs/releases/1.59/Build/Documentation/PointCloudShading.html)[

![](https://t3.gstatic.com/faviconV2?url=https://cartographymaster.eu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://cartographymaster.eu/wp-content/theses/2020_Raymond_Presentation.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://repositum.tuwien.at/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://repositum.tuwien.at/bitstream/20.500.12708/17701/1/Schuetz%20Markus%20-%202021%20-%20Interactive%20exploration%20of%20point%20clouds.pdf)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/Dav1dde/glad/wiki/C)[

![](https://t0.gstatic.com/faviconV2?url=https://www.gdmc.nl/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.gdmc.nl/publications/2019/MSc_thesis_Derek_van_Bochove.pdf)[

![](https://t3.gstatic.com/faviconV2?url=https://pcl.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://pcl.readthedocs.io/projects/tutorials/en/master/reading_pcd.html)[

![](https://t0.gstatic.com/faviconV2?url=https://gist.github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://gist.github.com/tompaana/0cee553ca5d0d5c8de19fa9590356879)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/55091977/quickly-reading-xyz-point-cloud-and-rendering-animated-points-into-unity)[

![](https://t1.gstatic.com/faviconV2?url=https://pro.arcgis.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://pro.arcgis.com/en/pro-app/3.4/sdk/api-reference/topic44.html)[

![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.mdpi.com/2072-4292/13/8/1413)[

![](https://t0.gstatic.com/faviconV2?url=https://www.youtube.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.youtube.com/watch?v=Wh2bRURxjaA)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/27950956/generate-a-depth-texture-in-opengl-es-2-0-or-3-0)[

![](https://t0.gstatic.com/faviconV2?url=https://www.youtube.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.youtube.com/watch?v=UDQHNCXZPCg)[

![](https://t0.gstatic.com/faviconV2?url=http://potree.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](http://potree.org/potree/examples/lion.html)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/4041682/android-opengl-es-framebuffer-objects-rendering-depth-buffer-to-texture)[

![](https://t0.gstatic.com/faviconV2?url=https://opengldotblog.wordpress.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://opengldotblog.wordpress.com/2021/12/13/glad-and-extensions/)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/54834826/cannot-find-package-glfw-using-cmake)[

![](https://t3.gstatic.com/faviconV2?url=https://bbs.archlinux.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://bbs.archlinux.org/viewtopic.php?id=236551)[

![](https://t3.gstatic.com/faviconV2?url=https://discourse.glfw.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://discourse.glfw.org/t/building-using-cmake-issues-when-configuring/2769)[

![](https://t2.gstatic.com/faviconV2?url=https://discourse.cmake.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://discourse.cmake.org/t/im-having-trouble-linking-glfw-and-glm/9793)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/mmolero/awesome-point-cloud-processing)[

![](https://t1.gstatic.com/faviconV2?url=https://forums.unrealengine.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://forums.unrealengine.com/t/point-cloud-rendering/19225)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/32268850/large-point-cloud-data-rendering-viewer-at-30fps)[

![](https://t3.gstatic.com/faviconV2?url=https://www.magnopus.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.magnopus.com/blog/how-we-render-extremely-large-point-clouds)[

![](https://t0.gstatic.com/faviconV2?url=https://medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://medium.com/realities-io/point-cloud-rendering-7bd83c6220c8)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/46180716/incorporating-z-buffer-information-into-shading)[

![](https://t0.gstatic.com/faviconV2?url=https://docs.pygfx.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://docs.pygfx.org/v0.14.0/_gallery/other/edl_pointcloud.html)[

![](https://t0.gstatic.com/faviconV2?url=https://cmake.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://cmake.org/cmake/help/latest/command/find_package.html)[

![](https://t0.gstatic.com/faviconV2?url=https://cmake.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Finding%20Packages.html)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/27004080/where-does-cmake-look-to-find-packages)[

![](https://t3.gstatic.com/faviconV2?url=https://docs.oracle.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://docs.oracle.com/en/database/oracle/spatial-studio/25.1/spstu/styling-map-layer.html)[

![](https://t3.gstatic.com/faviconV2?url=https://superodometry.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://superodometry.com/superloc)[

![](https://t2.gstatic.com/faviconV2?url=https://opg.optica.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://opg.optica.org/abstract.cfm?uri=oe-27-7-9965)[

![](https://t1.gstatic.com/faviconV2?url=https://esurf.copernicus.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://esurf.copernicus.org/articles/10/1141/2022/)[

![](https://t0.gstatic.com/faviconV2?url=https://computergraphics.stackexchange.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://computergraphics.stackexchange.com/questions/5379/how-do-i-include-all-extensions-my-system-supports-using-glad)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/retifrav/glfw-imgui-example/blob/master/CMakeLists.txt)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/PointCloudLibrary/pcl/issues/4940)[

![](https://t1.gstatic.com/faviconV2?url=https://gitee.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://gitee.com/msnh/Easy3D)[

![](https://t1.gstatic.com/faviconV2?url=https://3d.bk.tudelft.nl/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://3d.bk.tudelft.nl/liangliang/software.html)[

![](https://t0.gstatic.com/faviconV2?url=https://docs.techsoft3d.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://docs.techsoft3d.com/hps/2023_SP2/prog_guide/0805_lighting_interpolation.html)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/LiangliangNan/Easy3D)[

![](https://t2.gstatic.com/faviconV2?url=https://gen.glad.sh/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://gen.glad.sh/)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/google/filament/discussions/3831)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/potree/potree/issues/231)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/CesiumGS/cesium/issues/6794)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/19712611/getting-eye-space-point-from-a-depth-texture)[

![](https://t0.gstatic.com/faviconV2?url=https://community.cesium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://community.cesium.com/t/point-cloud-translucency-and-eye-dome-lighting/38369)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/53067067/reading-ascii-point-clouds-in-x-y-z-r-g-b-format)[

![](https://t2.gstatic.com/faviconV2?url=https://support.safe.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://support.safe.com/hc/en-us/articles/25407688463245-Writing-Point-Clouds)[

![](https://t2.gstatic.com/faviconV2?url=https://forum.opencv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://forum.opencv.org/t/extract-xyz-coordinates-and-create-point-cloud/3491)[

![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.reddit.com/r/photogrammetry/comments/16igr1l/loading_and_working_with_point_clouds_using_c/)[

![](https://t2.gstatic.com/faviconV2?url=https://repositum.tuwien.at/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://repositum.tuwien.at/bitstream/20.500.12708/1624/2/Schuetz%20Markus%20-%202015%20-%20Potree%20rendering%20large%20point%20clouds%20in%20web%20browsers.pdf)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/SFraissTU/BA_PointCloud)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/potree/potree)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/57996782/edl-shader-in-potree-viewer-with-orthographic-mode)[

![](https://t0.gstatic.com/faviconV2?url=https://forum.babylonjs.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://forum.babylonjs.com/t/eye-dome-lighting-edl-for-point-clouds/21737)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/20482851/how-to-use-multisampling-with-opengl-fbos)[

![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.reddit.com/r/opengl/comments/j5qu0x/just_want_to_share_my_opengl_glfw_glad_project/)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/77039244/building-imgui-with-cmake-vcpkg-json-and-glfwlinking-error)[

![](https://t0.gstatic.com/faviconV2?url=https://www.youtube.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.youtube.com/watch?v=5Ms9R_dfBNc)