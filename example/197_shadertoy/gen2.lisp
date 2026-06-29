;;;; =========================================================================
;;;; LISP TO GLSL SHADER GENERATOR (EXAMPLE 197 - POINTCLOUD VERSION)
;;;; =========================================================================
;;;;
;;;; DESCRIPTION:
;;;; This file generates a 3D point cloud renderer with Screen-Space Shadows
;;;; (SSS) and Eye-Dome Lighting (EDL) shading using the cl-cpp-generator2 DSL.
;;;;
;;;; The transpiled code is split into two files:
;;;;  1. `buf0.glsl` (State & Point Cloud Rasterizer): Stores state in (0,0) and
;;;;     rasterizes the 3D point cloud into the texture buffer (iChannel0).
;;;;  2. `main_image.glsl` (Render Pass): Reconstructs 3D positions, estimates
;;;;     surface normals, applies diffuse shading, traces screen-space shadows,
;;;;     and applies EDL outlining.
;;;; =========================================================================

;; Load the C++ code generator framework
(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; Global metadata definition for widgets:
;; Widget 0: Point Size (1.0 to 15.0)
;; Widget 1: EDL Strength (0.0 to 5.0)
;; Widget 2: Shadow Strength (0.0 to 1.0)
(defparameter *widget-meta*
  '((0 x 0.10f0  1.0f0 15.0f0 0.10f0 0.15f0)      ; index, component, delta, min, max, y_min, y_max
    (1 y 0.05f0  0.0f0 5.0f0  0.18f0 0.23f0)
    (2 w 0.02f0  0.0f0 1.0f0  0.26f0 0.31f0)))

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
  ;; 1. CODE GENERATOR HELPER: horizontal sliders
  ;; =========================================================================
  (defun make-slider-overlay (index val-expr y-center min-val max-val)
    (let ((val-sym (intern (format nil "val_~a" index)))
          (y-sym (intern (format nil "y_center_~a" index)))
          (focused-sym (intern (format nil "is_focused_~a" index)))
          (hx-sym (intern (format nil "hx_~a" index))))
      `(let (,val-sym ,y-sym ,focused-sym)
         (declare (type float ,val-sym ,y-sym)
                  (type bool ,focused-sym))
         (setf ,val-sym (/ (- ,val-expr ,min-val) ,(- max-val min-val))
               ,y-sym ,y-center
               ,focused-sym (== focused_widget ,(float index 0.0f0)))
         (let (,hx-sym)
           (declare (type float ,hx-sym))
           (setf ,hx-sym (+ 0.05f0 (* ,val-sym 0.35f0)))
           (when (logand (>= scr_uv.x 0.05f0) (<= scr_uv.x 0.40f0) (< (abs (- scr_uv.y ,y-sym)) 0.006f0))
             (setf col (mix col (? ,focused-sym focus_color bar_color) 0.8f0)))
           (when (< (length (- scr_uv (vec2 ,hx-sym ,y-sym))) 0.012f0)
             (setf col (mix col (? ,focused-sym focus_color handle_color) 1.0f0)))))))

  ;; =========================================================================
  ;; 2. STATE BUFFER & POINT CLOUD RASTERIZER (buf0.glsl)
  ;; =========================================================================
  (let* ((buf-code
          `(do0
            "// --- transpiled interactive state buffer & point cloud rasterizer ---"
            
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
              
              (let (ipx state)
                (declare (type ivec2 ipx)
                         (type vec4 state))
                (setf ipx (ivec2 fragCoord)
                      state (vec4 5.0f0 1.5f0 0.0f0 0.7f0)) ;; Initial default values: size=5.0, edl=1.5, focused=0, shadow=0.7
                
                (when (> iFrame 0)
                  (setf state (texelFetch iChannel0 (ivec2 0 0) 0)))
                
                ;; Only process widget state logic in the first pixel
                (when (== ipx (ivec2 0 0))
                  ;; Keyboard Navigation: Tab / Shift-Tab
                  (when (is_key_pressed 9) ; Tab
                    (if (is_key_down 16) ; Shift
                        (setf (dot state z) (mod (- (dot state z) 1.0f0) 3.0f0))
                        (setf (dot state z) (mod (+ (dot state z) 1.0f0) 3.0f0))))
                  
                  ;; Keyboard Adjustments: Left / Right Arrows
                  (let (left right)
                    (declare (type bool left right))
                    (setf left (is_key_down 37)
                          right (is_key_down 39))
                    (when left
                      (cond
                        ,@(loop for (idx comp delta min-val max-val) in *widget-meta*
                                collect `((== (dot state z) ,(float idx 0.0f0))
                                          (setf (dot state ,comp) (max (- (dot state ,comp) ,delta) ,min-val))))))
                    (when right
                      (cond
                        ,@(loop for (idx comp delta min-val max-val) in *widget-meta*
                                collect `((== (dot state z) ,(float idx 0.0f0))
                                          (setf (dot state ,comp) (min (+ (dot state ,comp) ,delta) ,max-val)))))))
                  
                  ;; Mouse Interactions: Sliders
                  (when (> iMouse.z 0.0f0)
                    (let (m res)
                      (declare (type vec2 m res))
                      (setf m iMouse.xy
                            res iResolution.xy)
                      (let (mx my)
                        (declare (type float mx my))
                        (setf mx (/ m.x res.x)
                              my (/ m.y res.y))
                        ;; Check if mouse click coordinates hit the slider bounding boxes (X span: 0.05 to 0.40)
                        (when (logand (>= mx 0.05f0) (<= mx 0.40f0))
                          (let (val)
                            (declare (type float val))
                            (setf val (/ (- mx 0.05f0) 0.35f0))
                            (cond
                              ,@(loop for (idx comp delta min-val max-val y-min y-max) in *widget-meta*
                                      collect `((logand (>= my ,y-min) (<= my ,y-max))
                                                (setf (dot state z) ,(float idx 0.0f0)
                                                      (dot state ,comp) (+ ,min-val (* val ,(- max-val min-val))))))))))))
                  (setf fragColor state))
                
                (unless (== ipx (ivec2 0 0))
                  ;; Render Point Cloud into the buffer
                  (let (point_size)
                    (declare (type float point_size))
                    (setf point_size (dot state x))
                    
                    (let ((min_depth 100000.0f0)
                          (hitColor (vec3 0.0f0)))
                      (declare (type float min_depth)
                               (type vec3 hitColor))
                      
                      ;; Rasterize 1000 points
                      (for ("int i = 0" (< i 1000) (incf i))
                        (let (theta phi R r p cy sy rotY rotX proj p_pixel dist size depth)
                          (declare (type float theta phi R r dist size depth cy sy)
                                   (type vec3 p)
                                   (type vec2 proj p_pixel)
                                   (type mat2 rotY rotX))
                          ;; Generate a rotating torus knot point cloud
                          (setf theta (* (float i) 0.006283f0) ;; 2*pi/1000
                                phi (* (float i) 0.03f0)       ;; frequency
                                R 1.6f0
                                r 0.6f0
                                p (vec3 (* (+ R (* r (cos (* phi 3.0f0)))) (cos (* phi 2.0f0)))
                                        (* r (sin (* phi 3.0f0)))
                                        (* (+ R (* r (cos (* phi 3.0f0)))) (sin (* phi 2.0f0)))))
                          
                          ;; Rotation
                          (setf cy (cos (* iTime 0.3f0))
                                sy (sin (* iTime 0.3f0))
                                rotY (mat2 cy (- sy) sy cy)
                                (dot p xz) (* rotY (dot p xz))
                                
                                cy (cos (* iTime 0.15f0))
                                sy (sin (* iTime 0.15f0))
                                rotX (mat2 cy (- sy) sy cy)
                                (dot p yz) (* rotX (dot p yz)))
                          
                          ;; Translate point in front of camera
                          (incf (dot p z) 5.0f0)
                          
                          ;; Project to screen coordinates
                          (setf proj (/ (dot p xy) (dot p z))
                                p_pixel (* (+ (* proj 0.5f0) 0.5f0) (dot iResolution xy))
                                dist (length (- fragCoord p_pixel))
                                size (/ point_size (dot p z)))
                          
                          (when (< dist size)
                            (setf depth (dot p z))
                            (when (< depth min_depth)
                              (setf min_depth depth
                                    hitColor (+ (vec3 0.5f0 0.5f0 0.5f0)
                                                (* 0.5f0 (vec3 (cos phi) (sin (* phi 2.0f0)) (cos (* phi 3.0f0))))))))))
                      
                      (if (< min_depth 10000.0f0)
                          (setf fragColor (vec4 hitColor min_depth))
                          (setf fragColor (vec4 0.1f0 0.15f0 0.2f0 100000.0f0)))))))))))
    (write-source *buf0-file* buf-code :format nil :tidy nil))

  ;; =========================================================================
  ;; 3. MAIN RENDERER (main_image.glsl)
  ;; =========================================================================
  (let* ((main-code
          `(do0
            "// --- transpiled main renderer: reconstruction, normals, shading, screen-space shadows, and EDL ---"
            
            ;; 3D coordinate reconstruction from pixel coordinate and depth
            (defun reconstructP (pixel depth)
              (declare (type ivec2 pixel)
                       (type float depth)
                       (values vec3))
              (let (uv proj)
                (declare (type vec2 uv proj))
                (setf uv (/ (- (vec2 pixel) (* 0.5f0 (dot iResolution xy))) (dot iResolution y))
                      proj (* uv 2.0f0))
                (return (vec3 (* proj depth) depth))))

            (defun mainImage (fragColor fragCoord)
              (declare (type "out vec4" fragColor)
                       (type "in vec2" fragCoord)
                       (values void))
              
              ;; Fetch state from (0,0) of iChannel0
              (let (state)
                (declare (type vec4 state))
                (setf state (texelFetch iChannel0 (ivec2 0 0) 0))
                
                (let (point_size edl_strength focused_widget shadow_strength)
                  (declare (type float point_size edl_strength focused_widget shadow_strength))
                  (setf point_size (dot state x)
                        edl_strength (dot state y)
                        focused_widget (dot state z)
                        shadow_strength (dot state w))
                  
                  ;; Fetch current pixel color & depth
                  (let (centerData baseColor depth)
                    (declare (type vec4 centerData)
                             (type vec3 baseColor)
                             (type float depth))
                    (setf centerData (texelFetch iChannel0 (ivec2 fragCoord) 0)
                          baseColor (dot centerData rgb)
                          depth (dot centerData w))
                    
                    (let (col)
                      (declare (type vec3 col))
                      
                      (if (> depth 10000.0f0)
                          ;; Render background with vignette
                          (let ((uv (/ (- fragCoord (* 0.5f0 (dot iResolution xy))) (dot iResolution y))))
                            (declare (type vec2 uv))
                            (setf col (- (vec3 0.1f0 0.15f0 0.2f0) (* 0.08f0 (length uv)))))
                          
                          ;; Render shaded and shadowed point
                          (let (P depth_R depth_U P_R P_U dPdx dPdy normal normal_cross light_pos L dif shadow_factor V R_ref spec)
                            (declare (type vec3 P P_R P_U dPdx dPdy normal normal_cross light_pos L V R_ref)
                                     (type float depth_R depth_U dif shadow_factor spec))
                            
                            ;; 1. Reconstruct 3D Position
                            (setf P (reconstructP (ivec2 fragCoord) depth))
                            
                            ;; 2. Reconstruct normal from neighbor depths (Screen-Space Normal)
                            (setf depth_R (dot (texelFetch iChannel0 (+ (ivec2 fragCoord) (ivec2 1 0)) 0) w)
                                  depth_U (dot (texelFetch iChannel0 (+ (ivec2 fragCoord) (ivec2 0 1)) 0) w))
                            (if (> depth_R 10000.0f0) (setf depth_R depth))
                            (if (> depth_U 10000.0f0) (setf depth_U depth))
                            
                            (setf P_R (reconstructP (+ (ivec2 fragCoord) (ivec2 1 0)) depth_R)
                                  P_U (reconstructP (+ (ivec2 fragCoord) (ivec2 0 1)) depth_U)
                                  dPdx (- P_R P)
                                  dPdy (- P_U P)
                                  normal_cross (cross dPdx dPdy))
                            (if (< (length normal_cross) 0.0001f0)
                                (setf normal (vec3 0.0f0 0.0f0 -1.0f0))
                                (setf normal (normalize normal_cross)))
                            
                            ;; 3. Shading (Diffuse & Specular Phong)
                            (setf light_pos (vec3 (* 2.5f0 (cos (* iTime 0.5f0)))
                                                  2.5f0
                                                  (+ (* 2.5f0 (sin (* iTime 0.5f0))) 4.5f0))
                                  L (normalize (- light_pos P))
                                  dif (clamp ("dot" normal L) 0.0f0 1.0f0)
                                  
                                  V (normalize (- P))
                                  R_ref (reflect (- L) normal)
                                  spec (* (pow (max ("dot" R_ref V) 0.0f0) 16.0f0) 0.3f0))
                            
                            ;; 4. Screen-Space Shadows (Raymarching towards light in depth buffer)
                            (let ((ray_dir (normalize (- light_pos P)))
                                  (light_dist (length (- light_pos P)))
                                  (t_max (min light_dist 5.0f0))
                                  (steps 24))
                              (declare (type float light_dist t_max)
                                       (type vec3 ray_dir)
                                       (type int steps))
                              (setf shadow_factor 1.0f0)
                              (for ("int step_idx = 1" (<= step_idx steps) (incf step_idx))
                                (let (tVal P_curr proj_curr uv_curr pixel_curr map_depth)
                                  (declare (type float tVal map_depth)
                                           (type vec3 P_curr)
                                           (type vec2 proj_curr uv_curr)
                                           (type ivec2 pixel_curr))
                                  (setf tVal (* (/ (float step_idx) (float steps)) t_max)
                                        P_curr (+ P (* ray_dir tVal))
                                        proj_curr (/ (dot P_curr xy) (dot P_curr z))
                                        uv_curr (* proj_curr 0.5f0)
                                        pixel_curr (ivec2 (+ (* uv_curr (dot iResolution y)) (* 0.5f0 (dot iResolution xy)))))
                                  
                                  (when (logior (< (dot pixel_curr x) 0) (>= (dot pixel_curr x) (dot iResolution x))
                                                (< (dot pixel_curr y) 0) (>= (dot pixel_curr y) (dot iResolution y)))
                                    break)
                                  
                                  (setf map_depth (dot (texelFetch iChannel0 pixel_curr 0) w))
                                  ;; Ray is behind the depth surface (occlusion) and neighbor is valid
                                  (when (logand (< map_depth 1000.0f0) (> (dot P_curr z) (+ map_depth 0.08f0)))
                                    (setf shadow_factor (- 1.0f0 shadow_strength))
                                    break))))
                            
                            (setf col (+ (* baseColor (+ (* dif shadow_factor) 0.15f0))
                                         (* (vec3 spec) shadow_factor)))))
                      
                      ;; 5. EDL Outlining / Shading
                      (let ((sum 0.0f0)
                            (edl_radius 2.0f0))
                        (declare (type float sum edl_radius))
                        "vec2 offsets[4] = vec2[](vec2(0.0f, 1.0f), vec2(0.0f, -1.0f), vec2(1.0f, 0.0f), vec2(-1.0f, 0.0f));"
                        (for ("int idx = 0" (< idx 4) (incf idx))
                          (let (neighborDepth)
                            (declare (type float neighborDepth))
                            (setf neighborDepth (dot (texelFetch iChannel0 (clamp (ivec2 (+ (ivec2 fragCoord) (ivec2 (* (aref offsets idx) edl_radius)))) (ivec2 0) (- (ivec2 (dot iResolution xy)) 1)) 0) w))
                            (when (> neighborDepth 10000.0f0)
                              (setf neighborDepth depth))
                            (setf sum (+ sum (max 0.0f0 (- depth neighborDepth))))))
                        ;; darken the color based on EDL difference
                        (setf col (* col (exp (* (- sum) 150.0f0 edl_strength)))))
                      
                      ;; Apply Gamma Correction
                      (setf col (pow col (vec3 0.4545f0)))
                      
                      ;; 6. Overlay 2D GUI widgets (Sliders)
                      (let (scr_uv bar_color handle_color focus_color)
                        (declare (type vec2 scr_uv)
                                 (type vec3 bar_color handle_color focus_color))
                        (setf scr_uv (/ fragCoord iResolution.xy)
                              bar_color (vec3 0.4f0)
                              handle_color (vec3 0.8f0)
                              focus_color (vec3 0.2f0 0.9f0 0.2f0))
                        
                        ,@(loop for (idx comp delta min-val max-val y-min y-max) in *widget-meta*
                                for val-expr = (case idx
                                                 (0 'point_size)
                                                 (1 'edl_strength)
                                                 (2 'shadow_strength))
                                for y-center = (/ (+ y-min y-max) 2.0f0)
                                collect (make-slider-overlay idx val-expr y-center min-val max-val)))
                      
                      (setf fragColor (vec4 col 1.0f0))))))))))
    (write-source *main-file* main-code :format nil :tidy nil)))
