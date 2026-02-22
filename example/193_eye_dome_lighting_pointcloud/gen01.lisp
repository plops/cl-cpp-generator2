;; ============================================================================
;; Minimal Modular Modern C++ Dense Point Cloud Viewer with Eye-Dome Lighting
;; Generated via cl-cpp-generator2 DSL
;; ============================================================================

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(progn
  (defparameter *source-dir* #P"example/pointcloud_edl/source/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname 'cl-cpp-generator2 *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  ;; Common Lisp Macro demonstrating the elimination of shader compilation boilerplate
  (defun gl-shader-info-log (var-msg)
    (destructuring-bind (&key var msg) var-msg
      `(let ((infoLog (std--vector<char> 512)))
         (glGetShaderInfoLog ,var (static_cast<GLsizei> (infoLog.size)) nullptr (infoLog.data))
         (let ((info (std--string (infoLog.begin) (infoLog.end))))
           (std--cout << (string ,msg) << (string " info='") << info << (string "'\n"))))))

  (write-source
   (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames "main.cpp" *source-dir*))
   `(do0
     (include<> glad/glad.h)
     (include<> GLFW/glfw3.h)
     (include<> glm/glm.hpp)
     (include<> glm/gtc/matrix_transform.hpp)
     (include<> glm/gtc/type_ptr.hpp)
     (include "imgui.h")
     (include "imgui_impl_glfw.h")
     (include "imgui_impl_opengl3.h")
     (include<> iostream)
     (include<> fstream)
     (include<> sstream)
     (include<> vector)
     (include<> string)
     (include<> cmath)

     (setf "const char *const geometry_vertex_shader_src"
           (string-r "#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }"))

     (setf "const char *const geometry_fragment_shader_src"
           (string-r "#version 330 core
out vec4 FragColor;
void main() { FragColor = vec4(0.85, 0.85, 0.85, 1.0); }"))

     (setf "const char *const edl_vertex_shader_src"
           (string-r "#version 330 core
out vec2 vTexCoords;
void main() {
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);
    vTexCoords.x = (x + 1.0) * 0.5;
    vTexCoords.y = (y + 1.0) * 0.5;
    gl_Position = vec4(x, y, 0.0, 1.0);
}"))

     (setf "const char *const edl_fragment_shader_src"
           (string-r "#version 330 core
out vec4 FragColor;
in vec2 vTexCoords;
uniform sampler2D uColorTexture;
uniform sampler2D uDepthTexture;
uniform float uScreenWidth;
uniform float uScreenHeight;
uniform float uEdlStrength;
uniform float uEdlRadius;
void main() {
    vec4 baseColor = texture(uColorTexture, vTexCoords);
    float depth = texture(uDepthTexture, vTexCoords).r;
    if (depth > 0.99999) { FragColor = baseColor; return; }
    vec2 texelSize = 1.0 / vec2(uScreenWidth, uScreenHeight);
    vec2 offsets[4] = vec2(vec2(0.0, 1.0), vec2(0.0, -1.0), vec2(1.0, 0.0), vec2(-1.0, 0.0));
    float sum = 0.0;
    for(int i = 0; i < 4; i++) {
        float neighborDepth = texture(uDepthTexture, vTexCoords + offsets[i] * texelSize * uEdlRadius).r;
        if (neighborDepth > 0.99999) neighborDepth = 0.0;
        if (neighborDepth!= 0.0) sum += max(0.0, depth - neighborDepth);
    }
    FragColor = vec4(baseColor.rgb * exp(-sum / 4.0 * 300.0 * uEdlStrength), baseColor.a);
}"))

     (defun load_xyz_point_cloud (filepath)
       (declare (type "const std::string&" filepath)
                (values "std::vector<float>"))
       (let ((points (std--vector<float>))
             (file (std--ifstream filepath)))
         (unless (file.is_open)
           (std--cerr << (string "Failed to open XYZ file\n"))
           (return points))
         (let ((line (std--string)))
           (while (std--getline file line)
             (if (line.empty) continue)
             (let ((ss (std--istringstream line))
                   (x 0.0f) (y 0.0f) (z 0.0f))
               (if (>> ss x y z)
                   (do0
                    (points.push_back x)
                    (points.push_back y)
                    (points.push_back z))))))
         (return points)))

     (defun main (argc argv)
       (declare (type int argc) (type char** argv) (values int))

       (unless (glfwInit) (return -1))
       (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 3)
       (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 3)
       (glfwWindowHint GLFW_OPENGL_PROFILE GLFW_OPENGL_CORE_PROFILE)

       (let ((window_width 1280)
             (window_height 720)
             (window (glfwCreateWindow window_width window_height (string "AAA EDL Viewer") nullptr nullptr)))
         (unless window
           (glfwTerminate)
           (return -1))

         (glfwMakeContextCurrent window)

         (unless (gladLoadGLLoader (reinterpret_cast<GLADloadproc> glfwGetProcAddress))
           (std--cerr << (string "Failed to initialize GLAD\n"))
           (return -1))

         (glEnable GL_DEPTH_TEST)
         (glDepthFunc GL_LESS)

         (let ((point_data (load_xyz_point_cloud (string "scan.txt"))))
           (if (point_data.empty)
               (do0
                (point_data.push_back 0.0f) (point_data.push_back 0.0f) (point_data.push_back 0.0f)
                (point_data.push_back 0.1f) (point_data.push_back 0.1f) (point_data.push_back -0.1f)
                (point_data.push_back -0.1f) (point_data.push_back 0.2f) (point_data.push_back 0.1f)))

           (let ((vao 0) (vbo 0))
             (glGenVertexArrays 1 &vao)
             (glGenBuffers 1 &vbo)

             (glBindVertexArray vao)
             (glBindBuffer GL_ARRAY_BUFFER vbo)
             (glBufferData GL_ARRAY_BUFFER (* (point_data.size) (sizeof float)) (point_data.data) GL_STATIC_DRAW)
             (glVertexAttribPointer 0 3 GL_FLOAT GL_FALSE (* 3 (sizeof float)) (reinterpret_cast<void*> 0))
             (glEnableVertexAttribArray 0)

             (let ((fbo 0)
                   (color_texture 0)
                   (depth_texture 0))
               (glGenFramebuffers 1 &fbo)
               (glBindFramebuffer GL_FRAMEBUFFER fbo)

               (glGenTextures 1 &color_texture)
               (glBindTexture GL_TEXTURE_2D color_texture)
               (glTexImage2D GL_TEXTURE_2D 0 GL_RGBA8 window_width window_height 0 GL_RGBA GL_UNSIGNED_BYTE nullptr)
               (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_NEAREST)
               (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_NEAREST)
               (glFramebufferTexture2D GL_FRAMEBUFFER GL_COLOR_ATTACHMENT0 GL_TEXTURE_2D color_texture 0)

               (glGenTextures 1 &depth_texture)
               (glBindTexture GL_TEXTURE_2D depth_texture)
               (glTexImage2D GL_TEXTURE_2D 0 GL_DEPTH_COMPONENT32F window_width window_height 0 GL_DEPTH_COMPONENT GL_FLOAT nullptr)
               (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_NEAREST)
               (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_NEAREST)
               (glFramebufferTexture2D GL_FRAMEBUFFER GL_DEPTH_ATTACHMENT GL_TEXTURE_2D depth_texture 0)

               (unless (== (glCheckFramebufferStatus GL_FRAMEBUFFER) GL_FRAMEBUFFER_COMPLETE)
                 (std--cerr << (string "FBO not complete!\n")))
               (glBindFramebuffer GL_FRAMEBUFFER 0)

               ;; Utilizing a generic lambda to compile shaders and reuse the Lisp macro seamlessly
               (let ((compile_and_link
                      (lambda (v_src f_src)
                        (declare (type "const char*" v_src f_src)
                                 (capture ""))
                        (let ((vs (glCreateShader GL_VERTEX_SHADER))
                              (fs (glCreateShader GL_FRAGMENT_SHADER))
                              (prog (glCreateProgram))
                              (success 0))
                          (glShaderSource vs 1 &v_src nullptr)
                          (glCompileShader vs)
                          (glGetShaderiv vs GL_COMPILE_STATUS &success)
                          (unless success ,(gl-shader-info-log `(:var vs :msg "VS Error")))

                          (glShaderSource fs 1 &f_src nullptr)
                          (glCompileShader fs)
                          (glGetShaderiv fs GL_COMPILE_STATUS &success)
                          (unless success ,(gl-shader-info-log `(:var fs :msg "FS Error")))

                          (glAttachShader prog vs)
                          (glAttachShader prog fs)
                          (glLinkProgram prog)
                          (glGetProgramiv prog GL_LINK_STATUS &success)
                          (unless success ,(gl-shader-info-log `(:var prog :msg "Link Error")))

                          (glDeleteShader vs)
                          (glDeleteShader fs)
                          (return prog))))
                     (geom_program (compile_and_link geometry_vertex_shader_src geometry_fragment_shader_src))
                     (edl_program (compile_and_link edl_vertex_shader_src edl_fragment_shader_src)))

                 (IMGUI_CHECKVERSION)
                 (ImGui--CreateContext)
                 (ImGui_ImplGlfw_InitForOpenGL window true)
                 (ImGui_ImplOpenGL3_Init (string "#version 330 core"))
                 (ImGui--StyleColorsDark)

                 (let ((edl_radius 1.5f)
                       (edl_strength 2.0f)
                       (angle 0.0f))

                   (while (!glfwWindowShouldClose window)
                     (glfwPollEvents)

                     (let ((current_w 0) (current_h 0))
                       (glfwGetFramebufferSize window &current_w &current_h)
                       (if (or (!= current_w window_width) (!= current_h window_height))
                           (do0
                            (setf window_width current_w
                                  window_height current_h)
                            (glBindTexture GL_TEXTURE_2D color_texture)
                            (glTexImage2D GL_TEXTURE_2D 0 GL_RGBA8 window_width window_height 0 GL_RGBA GL_UNSIGNED_BYTE nullptr)
                            (glBindTexture GL_TEXTURE_2D depth_texture)
                            (glTexImage2D GL_TEXTURE_2D 0 GL_DEPTH_COMPONENT32F window_width window_height 0 GL_DEPTH_COMPONENT GL_FLOAT nullptr))))

                     (ImGui_ImplOpenGL3_NewFrame)
                     (ImGui_ImplGlfw_NewFrame)
                     (ImGui--NewFrame)

                     (ImGui--Begin (string "EDL Settings"))
                     (ImGui--SliderFloat (string "Radius") &edl_radius 0.1f 5.0f)
                     (ImGui--SliderFloat (string "Strength") &edl_strength 0.1f 10.0f)
                     (ImGui--End)

                     (glBindFramebuffer GL_FRAMEBUFFER fbo)
                     (glViewport 0 0 window_width window_height)
                     (glClearColor 0.0f 0.0f 0.0f 1.0f)
                     (glClear (| GL_COLOR_BUFFER_BIT GL_DEPTH_BUFFER_BIT))

                     (glUseProgram geom_program)
                     (incf angle 0.01f)
                     (let ((view (glm--lookAt (glm--vec3 (* (sin angle) 5.0f) 2.0f (* (cos angle) 5.0f))
                                              (glm--vec3 0.0f 0.0f 0.0f)
                                              (glm--vec3 0.0f 1.0f 0.0f)))
                           (proj (glm--perspective (glm--radians 45.0f) (/ (static_cast<float> window_width) (static_cast<float> window_height)) 0.1f 100.0f))
                           (mvp (* proj view))
                           (mvp_loc (glGetUniformLocation geom_program (string "uMVP"))))
                       (glUniformMatrix4fv mvp_loc 1 GL_FALSE (glm--value_ptr mvp)))

                     (glBindVertexArray vao)
                     (glDrawArrays GL_POINTS 0 (/ (point_data.size) 3))

                     (glBindFramebuffer GL_FRAMEBUFFER 0)
                     (glViewport 0 0 window_width window_height)
                     (glClear (| GL_COLOR_BUFFER_BIT GL_DEPTH_BUFFER_BIT))

                     (glUseProgram edl_program)
                     (glActiveTexture GL_TEXTURE0)
                     (glBindTexture GL_TEXTURE_2D color_texture)
                     (glUniform1i (glGetUniformLocation edl_program (string "uColorTexture")) 0)

                     (glActiveTexture GL_TEXTURE1)
                     (glBindTexture GL_TEXTURE_2D depth_texture)
                     (glUniform1i (glGetUniformLocation edl_program (string "uDepthTexture")) 1)

                     (glUniform1f (glGetUniformLocation edl_program (string "uScreenWidth")) (static_cast<float> window_width))
                     (glUniform1f (glGetUniformLocation edl_program (string "uScreenHeight")) (static_cast<float> window_height))
                     (glUniform1f (glGetUniformLocation edl_program (string "uEdlRadius")) edl_radius)
                     (glUniform1f (glGetUniformLocation edl_program (string "uEdlStrength")) edl_strength)

                     (glDrawArrays GL_TRIANGLES 0 3)

                     (ImGui--Render)
                     (ImGui_ImplOpenGL3_RenderDrawData (ImGui--GetDrawData))
                     (glfwSwapBuffers window))

                   (ImGui_ImplOpenGL3_Shutdown)
                   (ImGui_ImplGlfw_Shutdown)
                   (ImGui--DestroyContext)

                   (glDeleteVertexArrays 1 &vao)
                   (glDeleteBuffers 1 &vbo)
                   (glDeleteFramebuffers 1 &fbo)
                   (glDeleteTextures 1 &color_texture)
                   (glDeleteTextures 1 &depth_texture)
                   (glDeleteProgram geom_program)
                   (glDeleteProgram edl_program)

                   (glfwDestroyWindow window)
                   (glfwTerminate)
                   (return 0))))))))))
  :omit-parens t
  :format t
  :tidy nil))
