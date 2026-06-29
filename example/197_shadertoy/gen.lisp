;;;; =========================================================================
;;;; LISP TO GLSL SHADER GENERATOR (EXAMPLE 197 - INTERACTIVE VERSION)
;;;; =========================================================================
;;;;
;;;; DESCRIPTION:
;;;; This file contains Common Lisp generator code that uses the
;;;; `cl-cpp-generator2` transpiler to emit a high-performance GLSL (OpenGL
;;;; Shading Language) fragment shader and a state-buffer shader.
;;;;
;;;; The transpiled code is split into two files:
;;;;  1. `buf0.glsl` (State buffer): Stores persistent interactive variables
;;;;     in the pixels of a double-buffered offscreen pass (iChannel0).
;;;;  2. `main_image.glsl` (Render pass): Reads state, renders the 3D scene,
;;;;     and overlays interactive 2D GUI sliders.
;;;;
;;;; INTERACTION DETAILS:
;;;; - KEYBOARD NAVIGATION:
;;;;   - Tab: Focus next slider widget (0 -> 1 -> 2 -> 0).
;;;;   - Shift+Tab: Focus previous slider widget.
;;;;   - Left/Right Arrows: Decrease/Increase the value of the active widget.
;;;; - MOUSE CONTROL:
;;;;   - Click and drag on any of the drawn sliders to change their value and
;;;;     focus them instantly.
;;;;
;;;; PARAMETERS CONTROLLED:
;;;;  - Widget 0: smax blend factor (0.0 to 2.0).
;;;;  - Widget 1: shadow k factor (1.0 to 100.0).
;;;;  - Widget 2: renderer maxDist (2.0 to 50.0).
;;;;
;;;; =========================================================================

;; Load the C++ code generator framework
(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  ;; Define output file paths
  (defparameter *buf0-file*
    (asdf:system-relative-pathname 'cl-cpp-generator2
                                   "example/197_shadertoy/vulkan-shadertoy-x11/launcher/shaders/shadertoy/buf0.glsl"))
  (defparameter *main-file*
    (asdf:system-relative-pathname 'cl-cpp-generator2
                                   "example/197_shadertoy/vulkan-shadertoy-x11/launcher/shaders/shadertoy/main_image.glsl"))

  ;; Make sure directory structure exists
  (ensure-directories-exist *buf0-file*)

  ;; =========================================================================
  ;; 1. STATE BUFFER GENERATION (buf0.glsl)
  ;; =========================================================================
  (let* ((buf-code
          `(do0
            "// --- transpiled interactive state buffer ---"
            
            ;; Helper function to check if a keyboard key is currently down (Row 0)
            (defun is_key_down (key)
              (declare (type int key)
                       (values bool))
              (return (> (dot (texelFetch iKeyboard (ivec2 key 0) 0) x) 0.5f0)))

            ;; Helper function to check if a keyboard key was pressed this frame (Row 1)
            (defun is_key_pressed (key)
              (declare (type int key)
                       (values bool))
              (return (> (dot (texelFetch iKeyboard (ivec2 key 1) 0) x) 0.5f0)))

            (defun mainImage (fragColor fragCoord)
              (declare (type "out vec4" fragColor)
                       (type "in vec2" fragCoord)
                       (values void))
              (let ((ipx (ivec2 fragCoord))
                    (state (vec4 0.2f0 16.0f0 0.0f0 10.0f0)))
                (declare (type ivec2 ipx)
                         (type vec4 state))
                
                ;; Only evaluate and store state in the first pixel (0, 0)
                (when (== ipx (ivec2 0 0))
                  ;; If it's not the first frame, fetch the previous state from itself
                  (when (> iFrame 0)
                    (setf state (texelFetch iChannel0 (ivec2 0 0) 0)))
                  
                  ;; Keyboard Navigation: Tab / Shift-Tab
                  (when (is_key_pressed 9) ; Tab
                    (if (is_key_down 16) ; Shift
                        (setf (dot state z) (mod (- (dot state z) 1.0f0) 3.0f0))
                        (setf (dot state z) (mod (+ (dot state z) 1.0f0) 3.0f0))))
                  
                  ;; Keyboard Adjustments: Left / Right Arrows
                  (let ((left (is_key_down 37))
                        (right (is_key_down 39)))
                    (declare (type bool left right))
                    (when left
                      (cond ((== (dot state z) 0.0f0)
                             (setf (dot state x) (max (- (dot state x) 0.005f0) 0.0f0)))
                            ((== (dot state z) 1.0f0)
                             (setf (dot state y) (max (- (dot state y) 0.2f0) 1.0f0)))
                            ((== (dot state z) 2.0f0)
                             (setf (dot state w) (max (- (dot state w) 0.1f0) 2.0f0)))))
                    (when right
                      (cond ((== (dot state z) 0.0f0)
                             (setf (dot state x) (min (+ (dot state x) 0.005f0) 2.0f0)))
                            ((== (dot state z) 1.0f0)
                             (setf (dot state y) (min (+ (dot state y) 0.2f0) 100.0f0)))
                            ((== (dot state z) 2.0f0)
                             (setf (dot state w) (min (+ (dot state w) 0.1f0) 50.0f0))))))
                  
                  ;; Mouse Interactions: Sliders
                  (when (> iMouse.z 0.0f0)
                    (let ((m iMouse.xy)
                          (res iResolution.xy))
                      (declare (type vec2 m res))
                      (let ((mx (/ m.x res.x))
                            (my (/ m.y res.y)))
                        (declare (type float mx my))
                        ;; Check if mouse click coordinates hit the slider bounding boxes (X span: 0.05 to 0.40)
                        (when (logand (>= mx 0.05f0) (<= mx 0.40f0))
                          (let ((val (/ (- mx 0.05f0) 0.35f0)))
                            (declare (type float val))
                            (cond
                              ;; Slider 0: smax blend (Y: 0.10 to 0.15)
                              ((logand (>= my 0.10f0) (<= my 0.15f0))
                               (setf (dot state z) 0.0f0
                                     (dot state x) (* val 2.0f0)))
                              ;; Slider 1: shadow k (Y: 0.18 to 0.23)
                              ((logand (>= my 0.18f0) (<= my 0.23f0))
                               (setf (dot state z) 1.0f0
                                     (dot state y) (+ 1.0f0 (* val 99.0f0))))
                              ;; Slider 2: maxDist (Y: 0.26 to 0.31)
                              ((logand (>= my 0.26f0) (<= my 0.31f0))
                               (setf (dot state z) 2.0f0
                                     (dot state w) (+ 2.0f0 (* val 48.0f0))))))))))
                  
                  (setf fragColor state))
                
                (unless (== ipx (ivec2 0 0))
                  (setf fragColor (vec4 0.0f0))))))))
    (write-source *buf0-file* buf-code :format nil :tidy nil))

  ;; =========================================================================
  ;; 2. MAIN RENDERER GENERATION (main_image.glsl)
  ;; =========================================================================
  (let* ((main-code
          `(do0
            "// --- transpiled raymarching shader with smin and shadows ---"
            
            ;; ---------------------------------------------------------------
            ;; FUNCTION: smin (Smooth Minimum)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Blends two distance fields (SDF values) together smoothly.
            ;;
            ;; PARAMETERS:
            ;; - a (float): Distance value to the first object.
            ;; - b (float): Distance value to the second object.
            ;; - k (float): Smoothing factor.
            ;;
            ;; RETURNS:
            ;; - float: The smoothly blended distance value.
            (defun smin (a b k)
              (declare (type float a b k)
                       (values float))
              (let (h)
                (declare (type float h))
                (setf h (clamp (+ 0.5f0 (* 0.5f0 (/ (- b a) k))) 0.0f0 1.0f0))
                (return (- (mix b a h) (* k h (- 1.0f0 h))))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: sdSphere (Sphere Signed Distance Function)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Calculates the shortest distance from point 'p' to the sphere boundary.
            (defun sdSphere (p s)
              (declare (type vec3 p)
                       (type float s)
                       (values float))
              (return (- (length p) s)))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: sdBox (Box Signed Distance Function)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Calculates the shortest distance from point 'p' to the box boundary.
            (defun sdBox (p b)
              (declare (type vec3 p)
                       (type vec3 b)
                       (values float))
              (let (q)
                (declare (type vec3 q))
                (setf q (- (abs p) b))
                (return (+ (length (max q 0.0f0)) (min (max q.x (max q.y q.z)) 0.0f0)))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: map (Scene Map / Distance Field Evaluator)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Defines all shapes in the 3D scene.
            ;; Now accepts a dynamic smax_blend parameter.
            (defun map (p smax_blend)
              (declare (type vec3 p)
                       (type float smax_blend)
                       (values float))
              (let (plane c s rot pRot box sphere blendedObject)
                (declare (type float plane c s blendedObject box sphere)
                         (type mat3 rot)
                         (type vec3 pRot))
                (setf plane (+ p.y 1.0f0)
                      c (cos iTime)
                      s (sin iTime)
                      rot (mat3 c 0.0f0 s 0.0f0 1.0f0 0.0f0 (- s) 0.0f0 c)
                      pRot (* rot p)
                      box (sdBox pRot (vec3 0.6f0))
                      sphere (sdSphere (- pRot (vec3 0.0f0 0.2f0 0.0f0)) 0.75f0)
                      blendedObject (smin box sphere smax_blend))
                (return (min plane blendedObject))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: getNormal (Calculate Surface Normal)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Estimates the surface normal using central differences.
            (defun getNormal (p smax_blend)
              (declare (type vec3 p)
                       (type float smax_blend)
                       (values vec3))
              (let (e d n)
                (declare (type vec2 e)
                         (type float d)
                         (type vec3 n))
                (setf e (vec2 0.001f0 0.0f0)
                      d (map p smax_blend)
                      n (- d (vec3 (map (- p e.xyy) smax_blend)
                                   (map (- p e.yxy) smax_blend)
                                   (map (- p e.yyx) smax_blend))))
                (return (normalize n))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: getShadow (Raymarched Soft Shadow Calculator)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Calculates soft shadows pointing to the light source.
            (defun getShadow (ro rd mint maxt k smax_blend)
              (declare (type vec3 ro rd)
                       (type float mint maxt k smax_blend)
                       (values float))
              (let (res tVal)
                (declare (type float res tVal))
                (setf res 1.0f0
                      tVal mint)
                (for ("int i = 0" (< i 32) (incf i))
                  (let (h)
                    (declare (type float h))
                    (setf h (map (+ ro (* tVal rd)) smax_blend))
                    (when (< h 0.001f0)
                      (return 0.0f0))
                    (setf res (min res (/ (* k h) tVal)))
                    (incf tVal (clamp h 0.01f0 0.2f0))
                    (when (> tVal maxt)
                      break)))
                (return (clamp res 0.0f0 1.0f0))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: mainImage (Main Viewport Render Entrypoint)
            ;; ---------------------------------------------------------------
            (defun mainImage (fragColor fragCoord)
              (declare (type "out vec4" fragColor)
                       (type "in vec2" fragCoord)
                       (values void))
              
              ;; Fetch interactive states from buf0 (iChannel0)
              (let ((state (texelFetch iChannel0 (ivec2 0 0) 0)))
                (declare (type vec4 state))
                
                (let ((smax_blend (dot state x))
                      (shadow_k (dot state y))
                      (focused_widget (dot state z))
                      (maxDist (dot state w)))
                  (declare (type float smax_blend shadow_k focused_widget maxDist))
                  
                  (let (uv ro rd tVal hit p n lightPos l dif shadow objectColor col)
                    (declare (type vec2 uv)
                             (type vec3 ro rd p n lightPos l objectColor col)
                             (type float tVal dif shadow)
                             (type bool hit))
                    
                    ;; Aspect ratio correction
                    (setf uv (/ (- fragCoord (* 0.5f0 iResolution.xy)) iResolution.y)
                          ro (vec3 0.0f0 1.0f0 -3.0f0)
                          rd (normalize (vec3 uv 1.0f0))
                          tVal 0.0f0
                          hit false)
                    
                    ;; Raymarching loop using dynamic maxDist
                    (for ("int i = 0" "i < 80" "i++")
                      (let (d)
                        (declare (type float d))
                        (setf d (map (+ ro (* tVal rd)) smax_blend))
                        (when (< d 0.001f0)
                          (setf hit true)
                          break)
                        (incf tVal d)
                        (when (> tVal maxDist)
                          break)))
                    
                    ;; Background color
                    (setf col (vec3 0.1f0 0.15f0 0.2f0))
                    (when hit
                      (setf p (+ ro (* tVal rd))
                            n (getNormal p smax_blend)
                            lightPos (vec3 2.0f0 4.0f0 -1.0f0)
                            l (normalize (- lightPos p))
                            dif (clamp ("dot" n l) 0.0f0 1.0f0)
                            shadow (getShadow (+ p (* n 0.01f0)) l 0.01f0 5.0f0 shadow_k smax_blend))
                      (if (> p.y -0.99f0)
                          (setf objectColor (vec3 0.9f0 0.4f0 0.1f0))
                          (setf objectColor (vec3 0.5f0)))
                      (setf col (* objectColor (+ (* dif shadow) 0.1f0))
                            col (pow col (vec3 0.4545f0))))
                    
                    ;; Overlay 2D GUI widgets (Sliders)
                    (let ((scr_uv (/ fragCoord iResolution.xy))
                          (bar_color (vec3 0.4f0))
                          (handle_color (vec3 0.8f0))
                          (focus_color (vec3 0.2f0 0.9f0 0.2f0)))
                      (declare (type vec2 scr_uv)
                               (type vec3 bar_color handle_color focus_color))
                      
                      ;; Slider 0: smax blend factor
                      (let ((val_smax (/ smax_blend 2.0f0))
                            (y_center_smax 0.125f0)
                            (is_focused_smax (== focused_widget 0.0f0)))
                        (declare (type float val_smax y_center_smax)
                                 (type bool is_focused_smax))
                        (let ((hx_smax (+ 0.05f0 (* val_smax 0.35f0))))
                          (declare (type float hx_smax))
                          (when (logand (>= scr_uv.x 0.05f0) (<= scr_uv.x 0.40f0) (< (abs (- scr_uv.y y_center_smax)) 0.006f0))
                            (setf col (mix col (? is_focused_smax focus_color bar_color) 0.8f0)))
                          (when (< (length (- scr_uv (vec2 hx_smax y_center_smax))) 0.012f0)
                            (setf col (mix col (? is_focused_smax focus_color handle_color) 1.0f0)))))
                      
                      ;; Slider 1: shadow k factor
                      (let ((val_shadow (/ (- shadow_k 1.0f0) 99.0f0))
                            (y_center_shadow 0.205f0)
                            (is_focused_shadow (== focused_widget 1.0f0)))
                        (declare (type float val_shadow y_center_shadow)
                                 (type bool is_focused_shadow))
                        (let ((hx_shadow (+ 0.05f0 (* val_shadow 0.35f0))))
                          (declare (type float hx_shadow))
                          (when (logand (>= scr_uv.x 0.05f0) (<= scr_uv.x 0.40f0) (< (abs (- scr_uv.y y_center_shadow)) 0.006f0))
                            (setf col (mix col (? is_focused_shadow focus_color bar_color) 0.8f0)))
                          (when (< (length (- scr_uv (vec2 hx_shadow y_center_shadow))) 0.012f0)
                            (setf col (mix col (? is_focused_shadow focus_color handle_color) 1.0f0)))))
                      
                      ;; Slider 2: maxDist factor
                      (let ((val_dist (/ (- maxDist 2.0f0) 48.0f0))
                            (y_center_dist 0.285f0)
                            (is_focused_dist (== focused_widget 2.0f0)))
                        (declare (type float val_dist y_center_dist)
                                 (type bool is_focused_dist))
                        (let ((hx_dist (+ 0.05f0 (* val_dist 0.35f0))))
                          (declare (type float hx_dist))
                          (when (logand (>= scr_uv.x 0.05f0) (<= scr_uv.x 0.40f0) (< (abs (- scr_uv.y y_center_dist)) 0.006f0))
                            (setf col (mix col (? is_focused_dist focus_color bar_color) 0.8f0)))
                          (when (< (length (- scr_uv (vec2 hx_dist y_center_dist))) 0.012f0)
                            (setf col (mix col (? is_focused_dist focus_color handle_color) 1.0f0))))))
                    
                    (setf fragColor (vec4 col 1.0f0)))))))))
    (write-source *main-file* main-code :format nil :tidy nil)))
