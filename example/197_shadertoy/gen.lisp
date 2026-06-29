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

;; Global metadata definition to avoid code repetition across keyboard and mouse interactions
(defparameter *widget-meta*
  '((0 x 0.005f0 0.0f0 2.0f0 0.10f0 0.15f0)      ; index, component, delta, min, max, y_min, y_max
    (1 y 0.2f0   1.0f0 100.0f0 0.18f0 0.23f0)
    (2 w 0.1f0   2.0f0 50.0f0 0.26f0 0.31f0)))

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
  ;; To avoid code repetition, we generate the widget Lisp forms dynamically.
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
  ;; 2. STATE BUFFER GENERATION (buf0.glsl)
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
              
              ;; Use uninitialized let bindings + setf to avoid generating C++11 initializer lists {}
              (let (ipx state)
                (declare (type ivec2 ipx)
                         (type vec4 state))
                (setf ipx (ivec2 fragCoord)
                      state (vec4 0.2f0 16.0f0 0.0f0 10.0f0))
                
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
                                                      (dot state ,comp) ,(if (zerop min-val)
                                                                             `(* val ,max-val)
                                                                             `(+ ,min-val (* val ,(- max-val min-val))))))))))))
                  
                  (setf fragColor state))
                
                (unless (== ipx (ivec2 0 0))
                  (setf fragColor (vec4 0.0f0)))))))))
    (write-source *buf0-file* buf-code :format nil :tidy nil))

  ;; =========================================================================
  ;; 3. MAIN RENDERER GENERATION (main_image.glsl)
  ;; =========================================================================
  (let* ((main-code
          `(do0
            "// --- transpiled raymarching shader with smin and shadows ---"
            
            ;; =========================================================================
            ;; FORWARD-MODE ALGORITHMIC DIFFERENTIATION (AD) LIBRARY
            ;; =========================================================================
            ;; Algorithmic differentiation computes exact analytical derivatives of
            ;; functions by tracing the operations performed on the inputs.
            ;;
            ;; In this library, the independent variable is the 3D coordinate p.
            ;; Any scalar variable tracks its value and its 3D gradient vector.
            ;; Any 3D vector tracks its 3D value and its 3x3 Jacobian matrix (the
            ;; derivative of each vector component with respect to the input p).
            ;;
            ;; This allows us to compute the exact normal vector of the SDF scene map
            ;; at any point in a single pass without using finite differences, which
            ;; reduces rendering calculations (1 map call vs 4) and avoids numerical
            ;; precision problems near corners and edges.

            ;; ---------------------------------------------------------------
            ;; DUAL NUMBER TYPES FOR ALGORITHMIC DIFFERENTIATION
            ;; ---------------------------------------------------------------
            ;; Dual represents a scalar with its gradient:
            ;;   v: The real scalar value.
            ;;   d: The gradient w.r.t input p (dp/dx, dp/dy, dp/dz).
            "struct Dual { float v; vec3 d; };"

            ;; Dual3 represents a 3D vector with its 3x3 Jacobian:
            ;;   v: The 3D vector value.
            ;;   d: The Jacobian w.r.t input p. In GLSL's column-major format,
            ;;      d[0] is the gradient of v.x,
            ;;      d[1] is the gradient of v.y,
            ;;      d[2] is the gradient of v.z.
            "struct Dual3 { vec3 v; mat3 d; };"

            ;; ---------------------------------------------------------------
            ;; DUAL NUMBER ARITHMETIC UTILITIES
            ;; ---------------------------------------------------------------

            ;; addDual: Adds two Dual scalars.
            ;;   Value: a.v + b.v
            ;;   Derivative: By linearity of derivatives, d(a + b) = da + db.
            (defun addDual (a b)
              (declare (type Dual a b)
                       (values Dual))
              (let (r)
                (declare (type Dual r))
                (setf (dot r v) (+ (dot a v) (dot b v))
                      (dot r d) (+ (dot a d) (dot b d)))
                (return r)))

            ;; subDual: Subtracts two Dual scalars.
            ;;   Value: a.v - b.v
            ;;   Derivative: By linearity of derivatives, d(a - b) = da - db.
            (defun subDual (a b)
              (declare (type Dual a b)
                       (values Dual))
              (let (r)
                (declare (type Dual r))
                (setf (dot r v) (- (dot a v) (dot b v))
                      (dot r d) (- (dot a d) (dot b d)))
                (return r)))

            ;; subDual3: Subtracts a constant 3D vector from a Dual3 vector.
            ;;   Value: a.v - b
            ;;   Derivative: Since b is a constant, its derivative is 0.
            ;;               Therefore, the Jacobian is unchanged: J(a - b) = J(a).
            (defun subDual3 (a b)
              (declare (type Dual3 a)
                       (type vec3 b)
                       (values Dual3))
              (let (r)
                (declare (type Dual3 r))
                (setf (dot r v) (- (dot a v) b)
                      (dot r d) (dot a d))
                (return r)))

            ;; mulMat3Dual3: Multiplies a constant 3x3 matrix by a Dual3 vector.
            ;;   Value: m * p.v
            ;;   Derivative: For a linear operator M, the derivative of M*p w.r.t input
            ;;               coordinates is the matrix product M * J_p.
            (defun mulMat3Dual3 (m p)
              (declare (type mat3 m)
                       (type Dual3 p)
                       (values Dual3))
              (let (r)
                (declare (type Dual3 r))
                (setf (dot r v) (* m (dot p v))
                      (dot r d) (* m (dot p d)))
                (return r)))

            ;; getX: Extracts the X-component of a Dual3 vector as a Dual scalar.
            ;;   Value: q.v.x
            ;;   Derivative: The gradient of the X component is the first column
            ;;               of the Jacobian matrix: q.d[0].
            (defun getX (q)
              (declare (type Dual3 q)
                       (values Dual))
              (return (Dual (dot q v x) (aref (dot q d) 0))))

            ;; getY: Extracts the Y-component of a Dual3 vector as a Dual scalar.
            ;;   Value: q.v.y
            ;;   Derivative: The gradient of the Y component is the second column
            ;;               of the Jacobian matrix: q.d[1].
            (defun getY (q)
              (declare (type Dual3 q)
                       (values Dual))
              (return (Dual (dot q v y) (aref (dot q d) 1))))

            ;; getZ: Extracts the Z-component of a Dual3 vector as a Dual scalar.
            ;;   Value: q.v.z
            ;;   Derivative: The gradient of the Z component is the third column
            ;;               of the Jacobian matrix: q.d[2].
            (defun getZ (q)
              (declare (type Dual3 q)
                       (values Dual))
              (return (Dual (dot q v z) (aref (dot q d) 2))))

            ;; maxDualDual: Computes the maximum of two Dual scalars.
            ;;   Value: max(a.v, b.v)
            ;;   Derivative: Propagates the derivative from whichever scalar is greater.
            (defun maxDualDual (a b)
              (declare (type Dual a b)
                       (values Dual))
              (if (> (dot a v) (dot b v))
                  (return a)
                  (return b)))

            ;; maxDualFloat: Computes the maximum of a Dual scalar and a constant float.
            ;;   Value: max(a.v, b)
            ;;   Derivative: Propagates a's gradient if a.v > b, else zero vector.
            (defun maxDualFloat (a b)
              (declare (type Dual a)
                       (type float b)
                       (values Dual))
              (if (> (dot a v) b)
                  (return a)
                  (return (Dual b (vec3 0.0f0)))))

            ;; minDualDual: Computes the minimum of two Dual scalars.
            ;;   Value: min(a.v, b.v)
            ;;   Derivative: Propagates the derivative from whichever scalar is smaller.
            (defun minDualDual (a b)
              (declare (type Dual a b)
                       (values Dual))
              (if (< (dot a v) (dot b v))
                  (return a)
                  (return b)))

            ;; minDualFloat: Computes the minimum of a Dual scalar and a constant float.
            ;;   Value: min(a.v, b)
            ;;   Derivative: Propagates a's gradient if a.v < b, else zero vector.
            (defun minDualFloat (a b)
              (declare (type Dual a)
                       (type float b)
                       (values Dual))
              (if (< (dot a v) b)
                  (return a)
                  (return (Dual b (vec3 0.0f0)))))

            ;; absDual3: Computes the element-wise absolute value of a Dual3 vector.
            ;;   Value: abs(p.v)
            ;;   Derivative: The derivative of |x| is sign(x) * dx. We scale each column
            ;;               of the Jacobian by 1.0 if the corresponding component of v
            ;;               is non-negative, and -1.0 otherwise.
            (defun absDual3 (p)
              (declare (type Dual3 p)
                       (values Dual3))
              (let (r)
                (declare (type Dual3 r))
                (setf (dot r v) (abs (dot p v))
                      (dot r d) (mat3 (* (aref (dot p d) 0) (? (>= (dot p v x) 0.0f0) 1.0f0 -1.0f0))
                                      (* (aref (dot p d) 1) (? (>= (dot p v y) 0.0f0) 1.0f0 -1.0f0))
                                      (* (aref (dot p d) 2) (? (>= (dot p v z) 0.0f0) 1.0f0 -1.0f0))))
                (return r)))

            ;; maxDual3Float: Computes the component-wise maximum w.r.t a constant float.
            ;;   Value: max(a.v, vec3(b))
            ;;   Derivative: Sets the Jacobian column of a component to 0 if that component
            ;;               is less than or equal to b.
            (defun maxDual3Float (a b)
              (declare (type Dual3 a)
                       (type float b)
                       (values Dual3))
              (let (r)
                (declare (type Dual3 r))
                (setf (dot r v) (max (dot a v) (vec3 b))
                      (dot r d) (mat3 (? (> (dot a v x) b) (aref (dot a d) 0) (vec3 0.0f0))
                                      (? (> (dot a v y) b) (aref (dot a d) 1) (vec3 0.0f0))
                                      (? (> (dot a v z) b) (aref (dot a d) 2) (vec3 0.0f0))))
                (return r)))

            ;; lengthDual3: Computes the length of a Dual3 vector as a Dual scalar.
            ;;   Value: length(a.v)
            ;;   Derivative: The derivative of length(a) is normalize(a) * J_a.
            ;;               This vector-matrix product maps the local coordinates
            ;;               to the global gradient vector.
            (defun lengthDual3 (a)
              (declare (type Dual3 a)
                       (values Dual))
              (let (r lenVal)
                (declare (type Dual r)
                         (type float lenVal))
                (setf lenVal (length (dot a v))
                      (dot r v) lenVal)
                (if (> lenVal 0.0f0)
                    (setf (dot r d) (* (normalize (dot a v)) (dot a d)))
                    (setf (dot r d) (vec3 0.0f0)))
                (return r)))

            ;; =========================================================================
            ;; ANALYTICAL DERIVATIVES FOR SHADER PRIMITIVES
            ;; =========================================================================

            ;; ---------------------------------------------------------------
            ;; FUNCTION: sdSphere (Sphere Signed Distance Function)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Calculates the shortest distance from point 'p' to the sphere boundary.
            ;;
            ;; PARAMETERS:
            ;; - p (vec3): 3D coordinate point.
            ;; - s (float): Radius of the sphere.
            ;;
            ;; RETURNS:
            ;; - float: Shortest distance to sphere surface.
            (defun sdSphere (p s)
              (declare (type vec3 p)
                       (type float s)
                       (values float))
              (return (- (length p) s)))

            ;; sdSphereDual: Sphere SDF for Algorithmic Differentiation.
            ;;   Computes length(p) - s. Derivation: Since s is constant, the
            ;;   derivative is exactly that of length(p).
            (defun sdSphereDual (p s)
              (declare (type Dual3 p)
                       (type float s)
                       (values Dual))
              (let (len)
                (declare (type Dual len))
                (setf len (lengthDual3 p))
                (return (Dual (- (dot len v) s) (dot len d)))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: sdBox (Box Signed Distance Function)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Calculates the shortest distance from point 'p' to the box boundary.
            ;;
            ;; PARAMETERS:
            ;; - p (vec3): 3D coordinate point.
            ;; - b (vec3): Dimensions (half-widths) of the box.
            ;;
            ;; RETURNS:
            ;; - float: Shortest distance to box surface.
            (defun sdBox (p b)
              (declare (type vec3 p)
                       (type vec3 b)
                       (values float))
              (let (q)
                (declare (type vec3 q))
                (setf q (- (abs p) b))
                (return (+ (length (max q 0.0f0)) (min (max q.x (max q.y q.z)) 0.0f0)))))

            ;; sdBoxDual: Box SDF for Algorithmic Differentiation.
            ;;   Traces the box distance computation through the absolute value,
            ;;   dimension subtraction, positive component length, and negative
            ;;   bounds comparison.
            (defun sdBoxDual (p b)
              (declare (type Dual3 p)
                       (type vec3 b)
                       (values Dual))
              (let (q len mx mn)
                (declare (type Dual3 q)
                         (type Dual len mx mn))
                (setf q (subDual3 (absDual3 p) b)
                      len (lengthDual3 (maxDual3Float q 0.0f0))
                      mx (maxDualDual (getX q) (maxDualDual (getY q) (getZ q)))
                      mn (minDualFloat mx 0.0f0))
                (return (Dual (+ (dot len v) (dot mn v))
                              (+ (dot len d) (dot mn d))))))

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

            ;; sminDual: Smooth Minimum for Algorithmic Differentiation.
            ;;   Calculates the smoothly blended distance value and derivative.
            ;;   By the envelope theorem, the derivative of smin(a, b, k) w.r.t the
            ;;   clamped interpolation factor h is exactly 0 at the optimum.
            ;;   Therefore, the total derivative propagates directly as:
            ;;     d(smin(a, b)) = h * d(a) + (1 - h) * d(b).
            (defun sminDual (a b k)
              (declare (type Dual a b)
                       (type float k)
                       (values Dual))
              (let (h r)
                (declare (type float h)
                         (type Dual r))
                (setf h (clamp (+ 0.5f0 (* 0.5f0 (/ (- (dot b v) (dot a v)) k))) 0.0f0 1.0f0)
                      (dot r v) (- (mix (dot b v) (dot a v) h) (* k h (- 1.0f0 h)))
                      (dot r d) (mix (dot b d) (dot a d) h))
                (return r)))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: mapDual (Scene Map / Distance Field & Gradient Evaluator)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Evaluates the 3D scene map using dual numbers to simultaneously
            ;; compute the signed distance and its exact analytical gradient w.r.t p.
            (defun mapDual (p_val smax_blend)
              (declare (type vec3 p_val)
                       (type float smax_blend)
                       (values Dual))
              (let (p plane c s rot pRot box sphere blendedObject)
                (declare (type Dual3 p pRot)
                         (type Dual plane box sphere blendedObject)
                         (type float c s)
                         (type mat3 rot))
                ;; 1. Initialize independent variable p with identity Jacobian
                (setf (dot p v) p_val
                      (dot p d) (mat3 1.0f0)
                      ;; 2. Evaluate ground plane: value is p.y + 1.0, gradient is d_y (p.d[1])
                      plane (Dual (+ (dot p v y) 1.0f0) (aref (dot p d) 1))
                      ;; 3. Set up rotating coordinate system for the shapes
                      c (cos iTime)
                      s (sin iTime)
                      rot (mat3 c 0.0f0 s 0.0f0 1.0f0 0.0f0 (- s) 0.0f0 c)
                      pRot (mulMat3Dual3 rot p)
                      ;; 4. Evaluate rotated box and sphere SDFs
                      box (sdBoxDual pRot (vec3 0.6f0))
                      sphere (sdSphereDual (subDual3 pRot (vec3 0.0f0 0.2f0 0.0f0)) 0.75f0)
                      ;; 5. Blend the shapes smoothly
                      blendedObject (sminDual box sphere smax_blend))
                ;; 6. Combine ground plane and shapes using the minimum operator
                (return (minDualDual plane blendedObject))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: map (Scene Map / Distance Field Evaluator)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Defines all shapes in the 3D scene.
            ;;
            ;; PARAMETERS:
            ;; - p (vec3): 3D coordinate point.
            ;; - smax_blend (float): Dynamic smax parameter for smin blending.
            ;;
            ;; RETURNS:
            ;; - float: Combined distance field value.
            (defun map (p smax_blend)
              (declare (type vec3 p)
                       (type float smax_blend)
                       (values float))
              ;; Simply returns the scalar value part (.v) of the Dual computation
              (return (dot (mapDual p smax_blend) v)))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: getNormal (Calculate Surface Normal)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Computes the surface normal analytically using Algorithmic Differentiation.
            ;; This replaces the finite-difference approach which used multiple map calls.
            ;;
            ;; PARAMETERS:
            ;; - p (vec3): 3D point on or near the surface.
            ;; - smax_blend (float): Dynamic smax parameter for smin blending.
            ;;
            ;; RETURNS:
            ;; - vec3: Normalized direction vector pointing outwards from the surface.
            (defun getNormal (p smax_blend)
              (declare (type vec3 p)
                       (type float smax_blend)
                       (values vec3))
              ;; Simply returns the normalized gradient vector part (.d) of the Dual computation
              (return (normalize (dot (mapDual p smax_blend) d))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: getShadow (Raymarched Soft Shadow Calculator)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Casts a secondary ray from the surface point 'ro' to the light.
            ;; Calculates how much the light is blocked by checking proximity
            ;; to other scene shapes along the path.
            ;;
            ;; PARAMETERS:
            ;; - ro (vec3): Ray origin (surface coordinate).
            ;; - rd (vec3): Ray direction (pointing towards the light source).
            ;; - mint (float): Minimum ray step limit (avoids self-shadowing).
            ;; - maxt (float): Maximum ray range limit (distance to the light).
            ;; - k (float): Penumbra control factor (lower value = softer shadows).
            ;; - smax_blend (float): Dynamic smax parameter for smin blending.
            ;;
            ;; RETURNS:
            ;; - float: Shadow factor between 0.0 (fully shadowed) and 1.0 (unshadowed).
            (defun getShadow (ro rd mint maxt k smax_blend)
              (declare (type vec3 ro rd)
                       (type float mint maxt k smax_blend)
                       (values float))
              ;; Declare:
              ;; - res: Running shadow strength (starts fully bright / 1.0).
              ;; - tVal: Current travel distance along the shadow ray.
              (let (res tVal)
                (declare (type float res tVal))
                (setf res 1.0f0
                      tVal mint)
                ;; Step along the ray towards the light.
                (for ("int i = 0" (< i 32) (incf i))
                  (let (h)
                    (declare (type float h))
                    ;; Measure distance to closest shape.
                    (setf h (map (+ ro (* tVal rd)) smax_blend))
                    ;; If we hit something directly (distance ~ 0), we are in full shadow.
                    (when (< h 0.001f0)
                      (return 0.0f0))
                    ;; Estimate penumbra soft edge factor based on how close we got to the shape.
                    (setf res (min res (/ (* k h) tVal)))
                    ;; Step forward along the ray, clamping the step size for safety.
                    (incf tVal (clamp h 0.01f0 0.2f0))
                    ;; Break if we exceeded the distance to the light source.
                    (when (> tVal maxt)
                      break)))
                ;; Clamp and return shadow visibility value.
                (return (clamp res 0.0f0 1.0f0))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: mainImage (Main Viewport Render Entrypoint)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Setup camera, cast primary ray per pixel, evaluate hits, calculate
            ;; lighting, soft shadows, and colorize the pixel.
            ;;
            ;; PARAMETERS:
            ;; - fragColor (out vec4): Output pixel color (RGBA).
            ;; - fragCoord (in vec2): Input pixel screen coordinate.
            (defun mainImage (fragColor fragCoord)
              (declare (type "out vec4" fragColor)
                       (type "in vec2" fragCoord)
                       (values void))
              
              ;; Fetch interactive states from buf0 (iChannel0)
              (let (state)
                (declare (type vec4 state))
                (setf state (texelFetch iChannel0 (ivec2 0 0) 0))
                
                (let (smax_blend shadow_k focused_widget maxDist)
                  (declare (type float smax_blend shadow_k focused_widget maxDist))
                  (setf smax_blend (dot state x)
                        shadow_k (dot state y)
                        focused_widget (dot state z)
                        maxDist (dot state w))
                  
                  ;; Local variables for raymarching, hitting, and shading logic.
                  (let (uv ro rd tVal hit p n lightPos l dif shadow objectColor col)
                    (declare (type vec2 uv)
                             (type vec3 ro rd p n lightPos l objectColor col)
                             (type float tVal dif shadow)
                             (type bool hit))
                    
                    ;; 1. Normalize viewport coordinates (aspect ratio corrected, centered).
                    (setf uv (/ (- fragCoord (* 0.5f0 iResolution.xy)) iResolution.y)
                          ;; 2. Setup Camera Origin (ro) and Camera Vector / Ray Direction (rd).
                          ro (vec3 0.0f0 1.0f0 -3.0f0)
                          rd (normalize (vec3 uv 1.0f0))
                          ;; 3. Initialize raymarching limits and hit state.
                          tVal 0.0f0
                          hit false)
                    
                    ;; 4. Raymarching Loop: step along the primary camera ray.
                    (for ("int i = 0" "i < 80" "i++")
                      (let (d)
                        (declare (type float d))
                        ;; Measure distance to closest shape.
                        (setf d (map (+ ro (* tVal rd)) smax_blend))
                        ;; If distance is very small, we have hit a surface!
                        (when (< d 0.001f0)
                          (setf hit true)
                          break)
                        ;; Step forward along the ray by the safe distance.
                        (incf tVal d)
                        ;; Stop searching if the ray travelled past the maximum distance limit.
                        (when (> tVal maxDist)
                          break)))
                    
                    ;; 5. Background color defaults to dark slate blue.
                    (setf col (vec3 0.1f0 0.15f0 0.2f0))
                    ;; 6. Shade the surface if a hit occurred.
                    (when hit
                      ;; Calculate 3D intersection coordinate (p) and surface normal vector (n).
                      (setf p (+ ro (* tVal rd))
                            n (getNormal p smax_blend)
                            ;; Position the point light source in 3D space.
                            lightPos (vec3 2.0f0 4.0f0 -1.0f0)
                            ;; Direction pointing from hit point to the light.
                            l (normalize (- lightPos p))
                            ;; Diffuse lambertian lighting calculation (dot product).
                            dif (clamp ("dot" n l) 0.0f0 1.0f0)
                            ;; Calculate soft shadow factor pointing towards the light.
                            shadow (getShadow (+ p (* n 0.01f0)) l 0.01f0 5.0f0 shadow_k smax_blend))
                      ;; Assign colors: Ground plane (grey) vs. Blended shape (orange).
                      (if (> p.y -0.99f0)
                          (setf objectColor (vec3 0.9f0 0.4f0 0.1f0))
                          (setf objectColor (vec3 0.5f0)))
                      ;; Combine diffuse lighting, ambient lighting component (0.1), and shadow occlusion.
                      (setf col (* objectColor (+ (* dif shadow) 0.1f0))
                            ;; Apply Gamma Correction (factor 2.2 -> 1/2.2 ~ 0.4545) to convert linear colors to sRGB.
                            col (pow col (vec3 0.4545f0))))
                    
                    ;; 7. Overlay 2D GUI widgets (Sliders)
                    (let (scr_uv bar_color handle_color focus_color)
                      (declare (type vec2 scr_uv)
                               (type vec3 bar_color handle_color focus_color))
                      (setf scr_uv (/ fragCoord iResolution.xy)
                            bar_color (vec3 0.4f0)
                            handle_color (vec3 0.8f0)
                            focus_color (vec3 0.2f0 0.9f0 0.2f0))
                      
                      ;; Dynamically generate the slider forms using make-slider-overlay to avoid code duplication
                      ,@(loop for (idx comp delta min-val max-val y-min y-max) in *widget-meta*
                              for val-expr = (case idx
                                               (0 'smax_blend)
                                               (1 'shadow_k)
                                               (2 'maxDist))
                              for y-center = (/ (+ y-min y-max) 2.0f0)
                              collect (make-slider-overlay idx val-expr y-center min-val max-val)))
                    
                    (setf fragColor (vec4 col 1.0f0)))))))))
    (write-source *main-file* main-code :format nil :tidy nil)))
