#pragma once
#include <glad/glad.h>

// RAII Framebuffer Manager
class PostProcessingFBO {
public:
    GLuint fbo{0}, colorTex{0}, depthTex{0};
    int currentWidth = 0, currentHeight = 0;

    PostProcessingFBO();
    ~PostProcessingFBO();

    void resizeIfNeeded(int width, int height);
};
