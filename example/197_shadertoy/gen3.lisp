;;;; =========================================================================
;;;; LISP TO GLSL SHADER GENERATOR (EXAMPLE 197 - DSL COMPILER VERSION)
;;;; =========================================================================
;;;;
;;;; DESCRIPTION:
;;;; This file implements a declarative DSL compiler on top of the
;;;; `cl-cpp-generator2` framework. It transpiles high-level S-expressions
;;;; representing 3D SDF geometry, transformation, and deformations
;;;; into optimized GLSL raymarching code.
;;;; =========================================================================

;; Load the C++ code generator framework
(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; =========================================================================
;; 1. DSL COMPILER IMPLEMENTATION
;; =========================================================================

(defun coerce-float (val)
  "Coerces a Lisp number to a single-float representation compatible with cl-cpp-generator2."
  (cond
    ((numberp val) (float val 0.0f0))
    (t val)))

(defun compile-vec (expr)
  "Recursively compile vec forms into single-float components."
  (if (and (listp expr) (member (first expr) '(vec2 vec3 vec4)))
      `(,(first expr) ,@(mapcar #'coerce-float (rest expr)))
      expr))

(defun compile-transform (transform p-var)
  "Transpiles a transformation form into the corresponding transformed coordinate expression."
  (let ((op (first transform))
        (args (rest transform)))
    (case op
      (translate
       (destructuring-bind (x y z) args
         `(- ,p-var (vec3 ,(coerce-float x) ,(coerce-float y) ,(coerce-float z)))))
      (rotate-x
       (destructuring-bind (angle) args
         `(rotateX ,p-var ,(coerce-float angle))))
      (rotate-y
       (destructuring-bind (angle) args
         `(rotateY ,p-var ,(coerce-float angle))))
      (rotate-z
       (destructuring-bind (angle) args
         `(rotateZ ,p-var ,(coerce-float angle))))
      (scale
       (destructuring-bind (factor) args
         `(/ ,p-var ,(coerce-float factor))))
      (t p-var))))

(defun compile-deform (deform base-dist p-var)
  "Transpiles surface deformation / noise displacement forms."
  (let ((op (first deform))
        (args (rest deform)))
    (case op
      (noise-displace
       (destructuring-bind (&key (amplitude 0.1f0) (frequency 1.0f0) &allow-other-keys) args
         `(+ ,base-dist (* ,(coerce-float amplitude) (simple_noise (* ,p-var ,(coerce-float frequency)))))))
      (t base-dist))))

(defun extract-positional-args (lst)
  "Helper to filter out keyword arguments and return only positional arguments in a property list."
  (let (res)
    (let ((i 0))
      (loop while (< i (length lst))
            do (let ((item (nth i lst)))
                 (if (keywordp item)
                     (incf i 2)
                     (progn
                       (push item res)
                       (incf i))))))
    (reverse res)))

(defun compile-sdf-form (form &optional (p-var 'p))
  "Recursively transpiles the declarative DSL representation of SDFs into cl-cpp-generator2 forms."
  (cond
    ((atom form) form)
    (t
     (let ((op (first form))
           (args (rest form)))
       (case op
         (sphere
          (destructuring-bind (&key (radius 1.0f0) transform deform) args
            (let* ((local-p (if transform (compile-transform transform p-var) p-var))
                   (base-dist `(sdSphere ,local-p ,(coerce-float radius))))
              (if deform
                  (compile-deform deform base-dist local-p)
                  base-dist))))
         (box
          (destructuring-bind (&key (size '(vec3 1.0f0)) transform deform) args
            (let* ((local-p (if transform (compile-transform transform p-var) p-var))
                   (base-dist `(sdBox ,local-p ,(compile-vec size))))
              (if deform
                  (compile-deform deform base-dist local-p)
                  base-dist))))
         (torus
          (destructuring-bind (&key (radius-major 1.0f0) (radius-minor 0.2f0) transform deform) args
            (let* ((local-p (if transform (compile-transform transform p-var) p-var))
                   (base-dist `(sdTorus ,local-p (vec2 ,(coerce-float radius-major) ,(coerce-float radius-minor)))))
              (if deform
                  (compile-deform deform base-dist local-p)
                  base-dist))))
         (cylinder
          (destructuring-bind (&key (radius 0.5f0) (height 1.0f0) transform deform) args
            (let* ((local-p (if transform (compile-transform transform p-var) p-var))
                   (base-dist `(sdCylinder ,local-p (vec2 ,(coerce-float radius) ,(coerce-float height)))))
              (if deform
                  (compile-deform deform base-dist local-p)
                  base-dist))))
         (infinite-plane
          (destructuring-bind (&key (height 0.0f0) transform deform) args
            (let* ((local-p (if transform (compile-transform transform p-var) p-var))
                   (base-dist `(+ (dot ,local-p y) ,(coerce-float height))))
              (if deform
                  (compile-deform deform base-dist local-p)
                  base-dist))))
         (union
          (let ((compiled-args (mapcar (lambda (arg) (compile-sdf-form arg p-var)) args)))
            (reduce (lambda (a b) `(min ,a ,b)) compiled-args)))
         (difference
          (destructuring-bind (a b) args
            `(max (- ,(compile-sdf-form b p-var)) ,(compile-sdf-form a p-var))))
         (smooth-blend
          (destructuring-bind (&key radius &allow-other-keys) args
            (let ((sub-args (extract-positional-args args)))
              (destructuring-bind (a b) sub-args
                `(smin ,(compile-sdf-form a p-var) ,(compile-sdf-form b p-var) ,(coerce-float radius))))))
         (t form))))))

;; =========================================================================
;; 2. UNIT TESTS DEFINITIONS & RUNNER
;; =========================================================================

(defparameter *test-cases*
  '((:input (sphere :radius 0.9)
     :expected "sdSphere(p, 0.90F)")
    (:input (union (sphere :radius 0.9) (box :size (vec3 0.5)))
     :expected "min(sdSphere(p, 0.90F), sdBox(p, vec3(0.50F)))")
    (:input (difference (sphere :radius 0.9) (cylinder :radius 0.3 :height 2.0))
     :expected "max( -(sdCylinder(p, vec2(0.30F, 2.0F))), sdSphere(p, 0.90F))")
    (:input (smooth-blend :radius 0.4 (sphere :radius 0.9) (box :size (vec3 0.5)))
     :expected "smin(sdSphere(p, 0.90F), sdBox(p, vec3(0.50F)), 0.40F)")
    (:input (sphere :radius 0.9 :transform (translate 0.0 1.0 0.0))
     :expected "sdSphere((p)-(vec3(0.F, 1.0F, 0.F)), 0.90F)")
    (:input (sphere :radius 0.9 :deform (noise-displace :amplitude 0.1 :frequency 2.0))
     :expected "(sdSphere(p, 0.90F))+((0.10F)*(simple_noise((p)*(2.0F))))")))

(defun run-unit-tests ()
  (format t "~%=== RUNNING DSL COMPILER UNIT TESTS ===~%")
  (let ((passed-count 0)
        (total-count 0))
    (dolist (tc *test-cases*)
      (incf total-count)
      (let* ((input (getf tc :input))
             (expected (getf tc :expected))
             (compiled-ast (compile-sdf-form input 'p))
             (generated-code (format nil "~A" (emit-c :code compiled-ast))))
        ;; Remove trailing newlines/whitespace for comparison
        (setf generated-code (string-trim '(#\Space #\Newline #\Tab #\Return) generated-code))
        (if (string= generated-code expected)
            (progn
              (incf passed-count)
              (format t "Test ~A: PASS~%  Input: ~S~%  Output: ~A~%" total-count input generated-code))
            (format t "Test ~A: FAIL~%  Input: ~S~%  Expected: ~A~%  Got:      ~A~%"
                    total-count input expected generated-code))))
    (format t "=== TEST RESULTS: ~A / ~A PASSED ===~%~%" passed-count total-count)
    (assert (= passed-count total-count))))

;; Run tests immediately on load to verify correctness
(run-unit-tests)

;; =========================================================================
;; 3. SHADER SCENE BUILDER
;; =========================================================================

;; Define the sliders (using underscores for variable names)
(defparameter *sliders-def*
  '((heat_intensity :default 0.5 :min 0.0 :max 2.0 :label "Lava Heat")
    (spin_speed     :default 1.0 :min 0.0 :max 4.0 :label "Ring Speed")
    (melt_factor    :default 0.4 :min 0.1 :max 1.5 :label "Melt Radius")))

;; Generate *widget-meta* list
(defparameter *widget-meta*
  (let ((components '(x y w))
        (y-ranges '((0.10f0 0.15f0) (0.18f0 0.23f0) (0.26f0 0.31f0)))
        (deltas '(0.05f0 0.10f0 0.05f0))
        (idx 0)
        (res nil))
    (loop for slider in *sliders-def*
          for comp in components
          for y-range in y-ranges
          for delta in deltas
          do (destructuring-bind (name &key (default 0.0f0) (min 0.0f0) (max 1.0f0) label) slider
               (declare (ignore default label))
               (push (list idx comp delta min max (first y-range) (second y-range)) res)
               (incf idx)))
    (reverse res)))

;; Generate initial state vector
(defparameter *initial-state*
  (let ((defaults (mapcar (lambda (s) (float (getf (rest s) :default) 0.0f0)) *sliders-def*)))
    `(vec4 ,(first defaults) ,(second defaults) 0.0f0 ,(third defaults))))

;; Define the complex lava-core scene geometry using the DSL (using underscores)
(defparameter *scene-geom*
  '(union
    (infinite-plane :height 1.5f0)
    (difference
     (smooth-blend :radius melt_factor
                   (sphere :radius 0.9f0 :deform (noise-displace :amplitude (* heat_intensity 0.15f0) :frequency 3.0f0))
                   (union
                    (torus :radius-major 1.4f0 :radius-minor 0.1f0 :transform (rotate-x (* iTime spin_speed)))
                    (torus :radius-major 1.6f0 :radius-minor 0.08f0 :transform (rotate-y (* iTime (* spin_speed 0.7f0))))))
     (cylinder :radius 0.35f0 :height 3.0f0))))

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

  ;; Slider overlay generation helper
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
  ;; 4. STATE BUFFER GENERATION (buf0.glsl)
  ;; =========================================================================
  (let* ((buf-code
          `(do0
            "// --- transpiled interactive state buffer ---"
            
            (defun is_key_down (key)
              (declare (type int key)
                       (values bool))
              (return (> (dot (texelFetch iKeyboard (ivec2 key 0) 0) x) 0.5f0)))

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
                      state ,*initial-state*)
                
                (when (== ipx (ivec2 0 0))
                  (when (> iFrame 0)
                    (setf state (texelFetch iChannel0 (ivec2 0 0) 0)))
                  
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
                  (setf fragColor (vec4 0.0f0))))))))
    (write-source *buf0-file* buf-code :format nil :tidy nil))

  ;; =========================================================================
  ;; 5. MAIN RENDERER GENERATION (main_image.glsl)
  ;; =========================================================================
  (let* ((main-code
          `(do0
            "// --- transpiled raymarching shader from declarative DSL ---"
            
            ;; Helper function: Smooth Minimum (smin)
            (defun smin (a b k)
              (declare (type float a b k)
                       (values float))
              (let (h)
                (declare (type float h))
                (setf h (clamp (+ 0.5f0 (* 0.5f0 (/ (- b a) k))) 0.0f0 1.0f0))
                (return (- (mix b a h) (* k h (- 1.0f0 h))))))

            ;; Helper functions: Transformations
            (defun rotateX (p a)
              (declare (type vec3 p) (type float a) (values vec3))
              (let (c s)
                (declare (type float c s))
                (setf c (cos a)
                      s (sin a))
                (return (vec3 p.x (- (* c p.y) (* s p.z)) (+ (* s p.y) (* c p.z))))))

            (defun rotateY (p a)
              (declare (type vec3 p) (type float a) (values vec3))
              (let (c s)
                (declare (type float c s))
                (setf c (cos a)
                      s (sin a))
                (return (vec3 (+ (* c p.x) (* s p.z)) p.y (- (* c p.z) (* s p.x))))))

            (defun rotateZ (p a)
              (declare (type vec3 p) (type float a) (values vec3))
              (let (c s)
                (declare (type float c s))
                (setf c (cos a)
                      s (sin a))
                (return (vec3 (- (* c p.x) (* s p.y)) (+ (* s p.x) (* c p.y)) p.z))))

            ;; Helper function: Simple Noise / Waves
            (defun simple_noise (p)
              (declare (type vec3 p) (values float))
              (return (* 0.333f0 (+ (sin p.x) (sin p.y) (sin p.z)))))

            ;; SDF Primitive: Sphere
            (defun sdSphere (p s)
              (declare (type vec3 p) (type float s) (values float))
              (return (- (length p) s)))

            ;; SDF Primitive: Box
            (defun sdBox (p b)
              (declare (type vec3 p) (type vec3 b) (values float))
              (let (q)
                (declare (type vec3 q))
                (setf q (- (abs p) b))
                (return (+ (length (max q 0.0f0)) (min (max q.x (max q.y q.z)) 0.0f0)))))

            ;; SDF Primitive: Torus
            (defun sdTorus (p tVal)
              (declare (type vec3 p) (type vec2 tVal) (values float))
              (let (q)
                (declare (type vec2 q))
                (setf q (vec2 (- (length (dot p xz)) (dot tVal x)) (dot p y)))
                (return (- (length q) (dot tVal y)))))

            ;; SDF Primitive: Cylinder
            (defun sdCylinder (p h)
              (declare (type vec3 p) (type vec2 h) (values float))
              (let (d)
                (declare (type vec2 d))
                (setf d (- (abs (vec2 (length (dot p xz)) (dot p y))) h))
                (return (+ (min (max d.x d.y) 0.0f0) (length (max d 0.0f0))))))

            ;; ---------------------------------------------------------------
            ;; Scene Evaluator / Distance Field (map)
            ;; ---------------------------------------------------------------
            (defun map (p heat_intensity spin_speed melt_factor)
              (declare (type vec3 p)
                       (type float heat_intensity spin_speed melt_factor)
                       (values float))
              (return ,(compile-sdf-form *scene-geom* 'p)))

            ;; Calculate Surface Normal
            (defun getNormal (p heat_intensity spin_speed melt_factor)
              (declare (type vec3 p)
                       (type float heat_intensity spin_speed melt_factor)
                       (values vec3))
              (let (e d n)
                (declare (type vec2 e)
                         (type float d)
                         (type vec3 n))
                (setf e (vec2 0.001f0 0.0f0)
                      d (map p heat_intensity spin_speed melt_factor)
                      n (- d (vec3 (map (- p e.xyy) heat_intensity spin_speed melt_factor)
                                   (map (- p e.yxy) heat_intensity spin_speed melt_factor)
                                   (map (- p e.yyx) heat_intensity spin_speed melt_factor))))
                (return (normalize n))))

            ;; Raymarched Soft Shadows
            (defun getShadow (ro rd mint maxt k heat_intensity spin_speed melt_factor)
              (declare (type vec3 ro rd)
                       (type float mint maxt k heat_intensity spin_speed melt_factor)
                       (values float))
              (let (res tVal)
                (declare (type float res tVal))
                (setf res 1.0f0
                      tVal mint)
                (for ("int i = 0" (< i 32) (incf i))
                  (let (h)
                    (declare (type float h))
                    (setf h (map (+ ro (* tVal rd)) heat_intensity spin_speed melt_factor))
                    (when (< h 0.001f0)
                      (return 0.0f0))
                    (setf res (min res (/ (* k h) tVal)))
                    (incf tVal (clamp h 0.01f0 0.2f0))
                    (when (> tVal maxt)
                      break)))
                (return (clamp res 0.0f0 1.0f0))))

            ;; Shading Main Image
            (defun mainImage (fragColor fragCoord)
              (declare (type "out vec4" fragColor)
                       (type "in vec2" fragCoord)
                       (values void))
              
              (let (state)
                (declare (type vec4 state))
                (setf state (texelFetch iChannel0 (ivec2 0 0) 0))
                
                (let (heat_intensity spin_speed focused_widget melt_factor)
                  (declare (type float heat_intensity spin_speed focused_widget melt_factor))
                  (setf heat_intensity (dot state x)
                        spin_speed (dot state y)
                        focused_widget (dot state z)
                        melt_factor (dot state w))
                  
                  (let (uv ro rd tVal hit p n lightPos l dif shadow objectColor col)
                    (declare (type vec2 uv)
                             (type vec3 ro rd p n lightPos l objectColor col)
                             (type float tVal dif shadow)
                             (type bool hit))
                    
                    (setf uv (/ (- fragCoord (* 0.5f0 iResolution.xy)) iResolution.y)
                          ro (vec3 0.0f0 1.0f0 -4.5f0)
                          rd (normalize (vec3 uv 1.0f0))
                          tVal 0.0f0
                          hit false)
                    
                    (for ("int i = 0" "i < 80" "i++")
                      (let (d)
                        (declare (type float d))
                        (setf d (map (+ ro (* tVal rd)) heat_intensity spin_speed melt_factor))
                        (when (< d 0.001f0)
                          (setf hit true)
                          break)
                        (incf tVal d)
                        (when (> tVal 12.0f0)
                          break)))
                    
                    (setf col (vec3 0.02f0 0.02f0 0.04f0))
                    (when hit
                      (setf p (+ ro (* tVal rd))
                            n (getNormal p heat_intensity spin_speed melt_factor)
                            lightPos (vec3 2.0f0 4.0f0 -3.0f0)
                            l (normalize (- lightPos p))
                            dif (clamp ("dot" n l) 0.0f0 1.0f0)
                            shadow (getShadow (+ p (* n 0.01f0)) l 0.01f0 5.0f0 16.0f0 heat_intensity spin_speed melt_factor))
                      
                      ;; Shading based on vertical position
                      (if (> p.y -1.49f0)
                          ;; Core/Rings have a warm fiery / metallic base
                          (let (centerDist)
                            (declare (type float centerDist))
                            (setf centerDist (length p)
                                  objectColor (mix (vec3 0.9f0 0.3f0 0.0f0) (vec3 0.7f0 0.7f0 0.8f0) (clamp centerDist 0.0f0 1.0f0))))
                          ;; Ground plane
                          (setf objectColor (vec3 0.15f0 0.15f0 0.15f0)))
                      
                      (setf col (* objectColor (+ (* dif shadow) 0.1f0))
                            col (pow col (vec3 0.4545f0))))
                    
                    ;; Sliders UI overlay
                    (let (scr_uv bar_color handle_color focus_color)
                      (declare (type vec2 scr_uv)
                               (type vec3 bar_color handle_color focus_color))
                      (setf scr_uv (/ fragCoord iResolution.xy)
                            bar_color (vec3 0.3f0)
                            handle_color (vec3 0.7f0)
                            focus_color (vec3 0.9f0 0.3f0 0.1f0))
                      
                      ,@(loop for (idx comp delta min-val max-val y-min y-max) in *widget-meta*
                              for val-expr = (case idx
                                               (0 'heat_intensity)
                                               (1 'spin_speed)
                                               (2 'melt_factor))
                              for y-center = (/ (+ y-min y-max) 2.0f0)
                              collect (make-slider-overlay idx val-expr y-center min-val max-val)))
                    
                    (setf fragColor (vec4 col 1.0f0)))))))))
    (write-source *main-file* main-code :format nil :tidy nil)))
