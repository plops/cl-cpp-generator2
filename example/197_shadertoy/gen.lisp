;;;; =========================================================================
;;;; LISP TO GLSL SHADER GENERATOR (EXAMPLE 197)
;;;; =========================================================================
;;;;
;;;; DESCRIPTION:
;;;; This file contains Common Lisp generator code that uses the
;;;; `cl-cpp-generator2` transpiler to emit a high-performance GLSL (OpenGL
;;;; Shading Language) fragment shader. The target shader is compatible with
;;;; Shadertoy's fragment shader input/output signature.
;;;;
;;;; The transpiled shader renders a 3D scene using Raymarching and Signed
;;;; Distance Fields (SDFs). The scene features:
;;;;  1. An analytical sphere and box shape merged dynamically using a
;;;;     smooth minimum (smin) function (inspired by Inigo Quilez's math).
;;;;  2. Real-time rotation of the blended shapes based on time (`iTime`).
;;;;  3. Analytical infinite ground plane.
;;;;  4. Custom normal calculations using central differences.
;;;;  5. Raymarched soft shadows pointing towards a point light source.
;;;;  6. Phong diffuse lighting, ambient lighting, and Gamma Correction.
;;;;
;;;; TRANSPILATION WORKFLOW:
;;;; 1. The script loads the `:cl-cpp-generator2` package.
;;;; 2. S-Expressions describing the GLSL abstract syntax tree (AST) are
;;;;    defined in Lisp format.
;;;; 3. The `write-source` function compiles these S-expressions and outputs
;;;;    the corresponding GLSL code string to `main_image.glsl`.
;;;;
;;;; =========================================================================

;; Ensure the transpiler framework is loaded and ready during compilation,
;; execution, and top-level load times.
(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

;; Switch the compiler scope to the transpiler package to resolve all DSL symbols.
(in-package :cl-cpp-generator2)

(progn
  ;; Define the target file where the transpiled GLSL shader should be written.
  (defparameter *code-file* 
    (asdf:system-relative-pathname 'cl-cpp-generator2 
                                   "example/197_shadertoy/vulkan-shadertoy-x11/launcher/shaders/shadertoy/main_image.glsl"))
  
  ;; Create parent directories for the target file if they do not exist yet.
  (ensure-directories-exist *code-file*)
  
  ;; Construct the GLSL abstract syntax tree using Common Lisp S-Expressions.
  (let* ((code
          `(do0
            "// --- transpiled raymarching shader with smin and shadows ---"
            
            ;; ---------------------------------------------------------------
            ;; FUNCTION: smin (Smooth Minimum)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Blends two distance fields (SDF values) together smoothly.
            ;; This creates a "melting" or organic transition between shapes.
            ;;
            ;; PARAMETERS:
            ;; - a (float): Distance value to the first object.
            ;; - b (float): Distance value to the second object.
            ;; - k (float): Smoothing factor (higher value = wider blend area).
            ;;
            ;; RETURNS:
            ;; - float: The smoothly blended distance value.
            (defun smin (a b k)
              (declare (type float a b k)
                       (values float))
              ;; Declare local variable 'h' which measures the interpolation factor.
              (let (h)
                (declare (type float h))
                ;; Calculate linear interpolation factor clamped between 0.0 and 1.0.
                (setf h (clamp (+ 0.5f0 (* 0.5f0 (/ (- b a) k))) 0.0f0 1.0f0))
                ;; Perform mix (linear interpolation) and subtract a quadratic factor
                ;; to round off the intersection corner smoothly.
                (return (- (mix b a h) (* k h (- 1.0f0 h))))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: sdSphere (Sphere Signed Distance Function)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Calculates the shortest distance from point 'p' to the sphere boundary.
            ;;
            ;; PARAMETERS:
            ;; - p (vec3): The 3D coordinate of the point to sample.
            ;; - s (float): The radius of the sphere.
            ;;
            ;; RETURNS:
            ;; - float: Distance (negative if inside the sphere, positive if outside).
            (defun sdSphere (p s)
              (declare (type vec3 p)
                       (type float s)
                       (values float))
              ;; Length of point vector from center minus the radius.
              (return (- (length p) s)))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: sdBox (Box Signed Distance Function)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Calculates the shortest distance from point 'p' to the box boundary
            ;; centered at the origin.
            ;;
            ;; PARAMETERS:
            ;; - p (vec3): The 3D coordinate of the point to sample.
            ;; - b (vec3): Half-dimensions (extents) of the box along X, Y, and Z.
            ;;
            ;; RETURNS:
            ;; - float: Shortest distance to the box surface.
            (defun sdBox (p b)
              (declare (type vec3 p)
                       (type vec3 b)
                       (values float))
              ;; Declare local vector 'q' representing the absolute distance vector
              ;; from the center minus the box half-extents.
              (let (q)
                (declare (type vec3 q))
                (setf q (- (abs p) b))
                ;; Distance formula combining exterior distance (length of positive components of q)
                ;; and interior distance (maximum negative component of q).
                (return (+ (length (max q 0.0f0)) (min (max q.x (max q.y q.z)) 0.0f0)))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: map (Scene Map / Distance Field Evaluator)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Defines and coordinates the positions of all shapes in the 3D scene.
            ;; Returns the overall closest distance from point 'p' to any shape.
            ;;
            ;; PARAMETERS:
            ;; - p (vec3): The 3D coordinate of the ray position.
            ;;
            ;; RETURNS:
            ;; - float: Minimum distance to the closest object in the scene.
            (defun map (p)
              (declare (type vec3 p)
                       (values float))
              ;; Define local variables:
              ;; - plane: Distance to the infinite ground floor plane at y = -1.0.
              ;; - c, s: Cosine and sine of the time variable (drives rotation).
              ;; - rot: 3x3 rotation matrix around the Y axis.
              ;; - pRot: Rotated coordinates of the object.
              ;; - box, sphere: SDF distance calculations for the individual primitives.
              ;; - blendedObject: Merged distance value of box and sphere.
              (let (plane c s rot pRot box sphere blendedObject)
                (declare (type float plane c s blendedObject box sphere)
                         (type mat3 rot)
                         (type vec3 pRot))
                ;; Position the infinite plane at Y = -1.0.
                (setf plane (+ p.y 1.0f0)
                      ;; Slow rotation calculations over time.
                      c (cos iTime)
                      s (sin iTime)
                      ;; Build rotation matrix around Y axis.
                      rot (mat3 c 0.0f0 s 0.0f0 1.0f0 0.0f0 (- s) 0.0f0 c)
                      ;; Apply rotation to the incoming coordinate vector.
                      pRot (* rot p)
                      ;; Evaluate Box SDF.
                      box (sdBox pRot (vec3 0.6f0))
                      ;; Translate the sphere coordinate slightly upwards and evaluate Sphere SDF.
                      sphere (sdSphere (- pRot (vec3 0.0f0 0.2f0 0.0f0)) 0.75f0)
                      ;; Melt the box and sphere together using the smooth minimum.
                      blendedObject (smin box sphere 0.2f0))
                ;; Return the minimum of the ground plane and the blended object.
                (return (min plane blendedObject))))

            ;; ---------------------------------------------------------------
            ;; FUNCTION: getNormal (Calculate Surface Normal)
            ;; ---------------------------------------------------------------
            ;; TASK:
            ;; Estimates the surface normal (perpendicular vector) at point 'p'
            ;; by evaluating numerical derivatives (central differences) of the SDF.
            ;;
            ;; PARAMETERS:
            ;; - p (vec3): 3D surface coordinate where the normal is needed.
            ;;
            ;; RETURNS:
            ;; - vec3: Normalized direction vector pointing outwards from the surface.
            (defun getNormal (p)
              (declare (type vec3 p)
                       (values vec3))
              ;; Declare:
              ;; - e: Small offset vector used for finite differences.
              ;; - d: Distance at the sample point.
              ;; - n: Gradient vector.
              (let (e d n)
                (declare (type vec2 e)
                         (type float d)
                         (type vec3 n))
                ;; Offset size (0.001) for taking derivatives.
                (setf e (vec2 0.001f0 0.0f0)
                      ;; Sample the distance field.
                      d (map p)
                      ;; Approximate partial derivatives along X, Y, Z.
                      n (- d (vec3 (map (- p e.xyy))
                                   (map (- p e.yxy))
                                   (map (- p e.yyx)))))
                ;; Normalize the resulting gradient vector to make it a unit vector.
                (return (normalize n))))

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
            ;;
            ;; RETURNS:
            ;; - float: Shadow factor between 0.0 (fully shadowed) and 1.0 (unshadowed).
            (defun getShadow (ro rd mint maxt k)
              (declare (type vec3 ro rd)
                       (type float mint maxt k)
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
                    (setf h (map (+ ro (* tVal rd))))
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
              ;; Local variables for raymarching, hitting, and shading logic.
              (let (uv ro rd tVal maxDist hit p n lightPos l dif shadow objectColor col)
                (declare (type vec2 uv)
                         (type vec3 ro rd p n lightPos l objectColor col)
                         (type float tVal maxDist dif shadow)
                         (type bool hit))
                ;; 1. Normalize viewport coordinates (aspect ratio corrected, centered).
                (setf uv (/ (- fragCoord (* 0.5f0 iResolution.xy)) iResolution.y)
                      ;; 2. Setup Camera Origin (ro) and Camera Vector / Ray Direction (rd).
                      ro (vec3 0.0f0 1.0f0 -3.0f0)
                      rd (normalize (vec3 uv 1.0f0))
                      ;; 3. Initialize raymarching limits and hit state.
                      tVal 0.0f0
                      maxDist 10.0f0
                      hit false)
                ;; 4. Raymarching Loop: step along the primary camera ray.
                (for ("int i = 0" "i < 80" "i++")
                  (let (d)
                    (declare (type float d))
                    ;; Measure distance to closest shape.
                    (setf d (map (+ ro (* tVal rd))))
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
                        n (getNormal p)
                        ;; Position the point light source in 3D space.
                        lightPos (vec3 2.0f0 4.0f0 -1.0f0)
                        ;; Direction pointing from hit point to the light.
                        l (normalize (- lightPos p))
                        ;; Diffuse lambertian lighting calculation (dot product).
                        dif (clamp ("dot" n l) 0.0f0 1.0f0)
                        ;; Calculate soft shadow factor pointing towards the light.
                        shadow (getShadow (+ p (* n 0.01f0)) l 0.01f0 5.0f0 16.0f0))
                  ;; Assign colors: Ground plane (grey) vs. Blended shape (orange).
                  (if (> p.y -0.99f0)
                      (setf objectColor (vec3 0.9f0 0.4f0 0.1f0))
                      (setf objectColor (vec3 0.5f0)))
                  ;; Combine diffuse lighting, ambient lighting component (0.1), and shadow occlusion.
                  (setf col (* objectColor (+ (* dif shadow) 0.1f0))
                        ;; Apply Gamma Correction (factor 2.2 -> 1/2.2 ~ 0.4545) to convert linear colors to sRGB.
                        col (pow col (vec3 0.4545f0))))
                ;; 7. Output final pixel color with alpha = 1.0.
                (setf fragColor (vec4 col 1.0f0)))))))
    
    ;; Transpile the S-Expression AST to C/GLSL syntax and write the result.
    ;; Disabling format and tidy since this is GLSL, not standard C++.
    (write-source *code-file* code :format nil :tidy nil)))
