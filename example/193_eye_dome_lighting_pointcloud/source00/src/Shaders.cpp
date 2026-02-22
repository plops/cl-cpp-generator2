#include "Shaders.hpp"

namespace Shaders {
    const char* geometryVS = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 mvp;

    void main() {
        gl_Position = mvp * vec4(aPos, 1.0);
    }
    )";

    const char* geometryFS = R"(
    #version 330 core
    layout (location = 0) out vec4 FragColor;
    uniform vec3 baseColor;

    void main() {
        FragColor = vec4(baseColor, 1.0);
    }
    )";

    const char* edlVS = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    }
    )";

    const char* edlFS = R"(
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
        vec2 offsets[4] = vec2[](
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
}
