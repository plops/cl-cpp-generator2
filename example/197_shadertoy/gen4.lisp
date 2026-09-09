;;;; =========================================================================
;;;; LISP TO GLSL SHADER GENERATOR (EXAMPLE 197 - ANIMATED EARTH GLOBE)
;;;; =========================================================================
;;;;
;;;; DESCRIPTION:
;;;; Generates a Shadertoy-compatible "planet Earth" shader:
;;;;
;;;;   * analytic ray/sphere planet with real coastlines, albedo and city
;;;;     lights baked from public datasets into GLSL constant tables
;;;;     (see tools/bake_earth_data.py -> earth_data.lisp),
;;;;   * procedural terrain bump mapping, ocean shelf / sea ice / snow,
;;;;     sun glint, soft terminator, night-side city lights,
;;;;   * an independently drifting animated cloud shell that also casts
;;;;     shadows onto the ground,
;;;;   * Rayleigh-ish atmosphere rim + crescent halo, star field background,
;;;;   * city markers (glowing surface rings + vertical light beams),
;;;;   * a camera "tour": the globe orientation is a SLERP interpolation
;;;;     between per-marker quaternions, computed at generation time in Lisp.
;;;;
;;;; OUTPUT (headless runner: --shader-dir <dir>):
;;;;   vulkan-shadertoy-x11/launcher/shaders/earth/common.glsl
;;;;   vulkan-shadertoy-x11/launcher/shaders/earth/buf0.glsl
;;;;   vulkan-shadertoy-x11/launcher/shaders/earth/main_image.glsl
;;;; =========================================================================

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; -------------------------------------------------------------------------
;; 0. Baked real-world Earth data (land mask / albedo / night lights)
;; -------------------------------------------------------------------------
(defparameter *earth-data-file*
  (asdf:system-relative-pathname 'cl-cpp-generator2
                                 "example/197_shadertoy/earth_data.lisp"))
(load *earth-data-file*)

;; =========================================================================
;; 1. LISP-SIDE VECTOR / QUATERNION MATH
;;    (the camera choreography is solved here, the shader only interpolates)
;; =========================================================================

(defun d->r (deg) (* (coerce pi 'double-float) (/ (float deg 1d0) 180d0)))

(defun v3 (x y z) (list (float x 1d0) (float y 1d0) (float z 1d0)))
(defun v3+ (a b) (mapcar #'+ a b))
(defun v3* (a s) (mapcar (lambda (x) (* x s)) a))
(defun v3dot (a b) (reduce #'+ (mapcar #'* a b)))
(defun v3len (a) (sqrt (v3dot a a)))
(defun v3norm (a) (let ((l (v3len a))) (if (< l 1d-12) a (v3* a (/ 1d0 l)))))
(defun v3cross (a b)
  (destructuring-bind (ax ay az) a
    (destructuring-bind (bx by bz) b
      (list (- (* ay bz) (* az by))
            (- (* az bx) (* ax bz))
            (- (* ax by) (* ay bx))))))

;; Quaternions are (x y z w), i.e. vector part first, scalar last -- exactly
;; the layout used by the generated GLSL helpers qMul / qRot / qSlerp.
(defun q-vec (q) (subseq q 0 3))
(defun q-w (q) (fourth q))
(defun q-mul (a b)
  (let ((av (q-vec a)) (bv (q-vec b)) (aw (q-w a)) (bw (q-w b)))
    (append (v3+ (v3+ (v3* bv aw) (v3* av bw)) (v3cross av bv))
            (list (- (* aw bw) (v3dot av bv))))))
(defun q-norm (q)
  (let ((l (sqrt (reduce #'+ (mapcar (lambda (x) (* x x)) q)))))
    (mapcar (lambda (x) (/ x l)) q)))
(defun q-rot (q v)
  "Rotate vector V by quaternion Q (same formula as the GLSL qRot)."
  (let ((qv (q-vec q)) (qw (q-w q)))
    (v3+ v (v3* (v3cross qv (v3+ (v3cross qv v) (v3* v qw))) 2d0))))
(defun q-axis-angle (axis angle)
  (let ((a (v3norm axis)))
    (append (v3* a (sin (* 0.5d0 angle))) (list (cos (* 0.5d0 angle))))))

(defun q-align (from to)
  "Shortest-arc quaternion rotating unit vector FROM onto unit vector TO."
  (let* ((f (v3norm from)) (g (v3norm to))
         (c (v3dot f g))
         (axis (v3cross f g))
         (s (v3len axis)))
    (cond
      ((< s 1d-9)
       (if (> c 0d0)
           (list 0d0 0d0 0d0 1d0)                       ; already aligned
           (q-axis-angle (v3 0 1 0) (coerce pi 'double-float)))) ; antipodal
      (t (q-axis-angle axis (atan s c))))))

(defun latlon->dir (lat lon)
  "Geographic coordinates -> earth-fixed unit vector.
Matches the shader's dirToUV(): lon = atan(x, z), lat = asin(y)."
  (let ((la (d->r lat)) (lo (d->r lon)))
    (v3 (* (cos la) (sin lo)) (sin la) (* (cos la) (cos lo)))))

(defun dir->latlon (d)
  (list (/ (* 180d0 (asin (max -1d0 (min 1d0 (second d))))) (coerce pi 'double-float))
        (/ (* 180d0 (atan (first d) (third d))) (coerce pi 'double-float))))

(defun view-quat (dir)
  "Globe orientation that brings earth-fixed DIR to the screen centre (+z,
towards the camera) while keeping the north pole pointing up on screen."
  (let* ((qa (q-align dir (v3 0 0 1)))
         (n (q-rot qa (v3 0 1 0)))            ; where the earth axis ends up
         (r (sqrt (+ (* (first n) (first n)) (* (second n) (second n))))))
    (if (< r 1d-6)
        (q-norm qa)                            ; marker sits on a pole
        (q-norm (q-mul (q-axis-angle (v3 0 0 1) (atan (first n) (second n))) qa)))))

;; =========================================================================
;; 2. THE MARKER TOUR (cities visited by the camera)
;; =========================================================================

(defparameter *markers*
  '((:name "BERLIN"     :lat  52.520d0 :lon  13.405d0 :color (1.00 0.86 0.30))
    (:name "NEW YORK"   :lat  40.713d0 :lon -74.006d0 :color (0.35 0.90 1.00))
    (:name "RIO"        :lat -22.907d0 :lon -43.173d0 :color (0.55 1.00 0.55))
    (:name "CAPE TOWN"  :lat -33.925d0 :lon  18.424d0 :color (1.00 0.55 0.75))
    (:name "TOKYO"      :lat  35.676d0 :lon 139.650d0 :color (1.00 0.45 0.35))
    (:name "SYDNEY"     :lat -33.869d0 :lon 151.209d0 :color (0.70 0.70 1.00))))

(defparameter *marker-count* (length *markers*))
(defparameter *tour-seg-seconds* 4.0)

(defun marker-dirs ()
  (mapcar (lambda (m) (latlon->dir (getf m :lat) (getf m :lon))) *markers*))

(defun marker-quats ()
  (mapcar #'view-quat (marker-dirs)))

;; -------------------------------------------------------------------------
;; 2b. Self-tests for the generation-time math
;; -------------------------------------------------------------------------
(defun approx= (a b &optional (eps 1d-6)) (< (abs (- a b)) eps))

(defun run-tour-tests ()
  (format t "~%=== EARTH TOUR MATH SELF-TESTS ===~%")
  (let ((n 0) (ok 0))
    ;; (1) lat/lon <-> direction round trip
    (dolist (m *markers*)
      (incf n)
      (let* ((d (latlon->dir (getf m :lat) (getf m :lon)))
             (ll (dir->latlon d)))
        (if (and (approx= (first ll) (getf m :lat) 1d-9)
                 (approx= (second ll) (getf m :lon) 1d-9)
                 (approx= (v3len d) 1d0 1d-12))
            (progn (incf ok)
                   (format t "  [PASS] ~12a dir=(~,4f ~,4f ~,4f) |d|=1~%"
                           (getf m :name) (first d) (second d) (third d)))
            (format t "  [FAIL] ~12a round trip ~a -> ~a~%"
                    (getf m :name) (list (getf m :lat) (getf m :lon)) ll))))
    ;; (2) view quaternion puts the marker in the screen centre, north up
    (loop for m in *markers*
          for d in (marker-dirs)
          for q in (marker-quats)
          do (incf n)
             (let* ((c (q-rot q d))              ; must be (0 0 1)
                    (up (q-rot q (v3 0 1 0))))   ; must have x=0, y>0
               (if (and (approx= (first c) 0d0 1d-9)
                        (approx= (second c) 0d0 1d-9)
                        (approx= (third c) 1d0 1d-9)
                        (approx= (first up) 0d0 1d-9)
                        (> (second up) 0d0))
                   (progn (incf ok)
                          (format t "  [PASS] ~12a centred, north up (up.y=~,4f)~%"
                                  (getf m :name) (second up)))
                   (format t "  [FAIL] ~12a centre=~a up=~a~%" (getf m :name) c up))))
    ;; (3) quaternions are unit length (slerp in the shader assumes that)
    (dolist (q (marker-quats))
      (incf n)
      (if (approx= (sqrt (reduce #'+ (mapcar (lambda (x) (* x x)) q))) 1d0 1d-9)
          (incf ok)
          (format t "  [FAIL] non-unit quaternion ~a~%" q)))
    ;; (4) q-mul / q-rot consistency: rotating twice == rotating by the product
    (let* ((qa (q-axis-angle (v3 1 2 3) 0.7d0))
           (qb (q-axis-angle (v3 -2 1 0.5) -1.3d0))
           (v (v3norm (v3 0.3 -0.7 0.2)))
           (r1 (q-rot qa (q-rot qb v)))
           (r2 (q-rot (q-mul qa qb) v)))
      (incf n)
      (if (every (lambda (a b) (approx= a b 1d-9)) r1 r2)
          (incf ok)
          (format t "  [FAIL] q-mul inconsistent: ~a vs ~a~%" r1 r2)))
    (format t "=== ~a / ~a TESTS PASSED ===~%~%" ok n)
    (assert (= ok n))))

(run-tour-tests)

;; =========================================================================
;; 3. GLSL EMISSION HELPERS
;; =========================================================================

(defun f32 (x) (coerce x 'single-float))

(defun glsl-uint-array (name words &optional (per-line 8))
  "Emit a GLSL const uint table (raw source string)."
  (let ((n (length words)))
    (with-output-to-string (s)
      (format s "const uint ~a[~a] = uint[~a](" name n n)
      (loop for w in words
            for i from 0
            do (when (zerop (mod i per-line)) (format s "~%    "))
               (format s "0x~8,'0xu" w)
               (when (< i (1- n)) (format s ",")))
      (format s ");~%"))))

(defun vec3-lit (v) `(vec3 ,(f32 (first v)) ,(f32 (second v)) ,(f32 (third v))))
(defun vec4-lit (v) `(vec4 ,(f32 (first v)) ,(f32 (second v))
                           ,(f32 (third v)) ,(f32 (fourth v))))

(defun index-dispatch (var entries default)
  "(when (== var 0) (return e0)) ... plus a trailing default return."
  (append (loop for e in entries
                for i from 0
                collect `(when (== ,var ,i) (return ,e)))
          (list `(return ,default))))

;; =========================================================================
;; 4. OUTPUT FILES
;; =========================================================================

(defparameter *earth-shader-dir*
  (asdf:system-relative-pathname 'cl-cpp-generator2
                                 "example/197_shadertoy/vulkan-shadertoy-x11/launcher/shaders/earth/"))
(defparameter *common-file* (merge-pathnames "common.glsl" *earth-shader-dir*))
(defparameter *buf0-file* (merge-pathnames "buf0.glsl" *earth-shader-dir*))
(defparameter *main-file* (merge-pathnames "main_image.glsl" *earth-shader-dir*))
(ensure-directories-exist *earth-shader-dir*)

;; -------------------------------------------------------------------------
;; 4a. common.glsl -- quaternion algebra, marker table, camera tour
;; -------------------------------------------------------------------------
(write-source
 *common-file*
 `(do0
   "// ================================================================="
   "// EARTH GLOBE -- shared math (generated by gen4.lisp, do not edit)"
   "// ================================================================="
   ,(format nil "const float PI = 3.14159265359;")
   ,(format nil "const int   MARKER_N = ~a;" *marker-count*)
   ,(format nil "const float MARKER_NF = ~,1f;" *marker-count*)
   ,(format nil "const float TOUR_SEG = ~,2f;   // seconds spent per marker" *tour-seg-seconds*)
   "const float EARTH_R = 1.0;"
   "const float CLOUD_R = 1.018;"

   "// --- layout of the baked Earth data inside the state buffer -----------"
   "// The GL driver turns big dynamically indexed const uint[] tables into"
   "// uncached memory traffic (measured 19.5 ms vs 2.3 ms per 1080p frame),"
   "// so buf0 unpacks them once per frame into its RGBA32F texture and the"
   "// render pass reads them with texelFetch instead."
   ,(format nil "const int LAND_W = ~a;   // 2 bit land coverage, 1 cell per texel" *land-mask-w*)
   ,(format nil "const int LAND_H = ~a;" *land-mask-h*)
   ,(format nil "const int SMALL_W = ~a;  // coast / albedo / lights grids" *coast-w*)
   ,(format nil "const int SMALL_H = ~a;" *coast-h*)
   "const int LAND_Y  = 4;             // land block occupies rows 4..131"
   "const int SMALL_Y = 136;           // small blocks occupy rows 136..167"
   "const int COAST_X = 0;"
   "const int ALB_X   = 70;"
   "const int LIT_X   = 140;"
   "// -> the state buffer must be at least 256 x 168 texels"

   "// --- quaternion algebra (x,y,z = vector part, w = scalar) ---"
   (defun qMul (a b)
     (declare (type vec4 a b) (values vec4))
     (return (vec4 (+ (* a.w b.xyz) (* b.w a.xyz) (cross a.xyz b.xyz))
                   (- (* a.w b.w) ("dot" a.xyz b.xyz)))))

   (defun qConj (q)
     (declare (type vec4 q) (values vec4))
     (return (vec4 (- q.xyz) q.w)))

   (defun qRot (q v)
     (declare (type vec4 q) (type vec3 v) (values vec3))
     (return (+ v (* 2.0f0 (cross q.xyz (+ (cross q.xyz v) (* q.w v)))))))

   (defun qAxisAngle (ax ang)
     (declare (type vec3 ax) (type float ang) (values vec4))
     (return (vec4 (* (normalize ax) (sin (* 0.5f0 ang))) (cos (* 0.5f0 ang)))))

   (defun qSlerp (a b s)
     (declare (type vec4 a b) (type float s) (values vec4))
     (let ((d ("dot" a b)))
       (declare (type float d))
       (when (< d 0.0f0)
         (setf b (- b)
               d (- d)))
       (when (> d 0.9995f0)
         (return (normalize (mix a b s))))
       (let ((th0 (acos (clamp d -1.0f0 1.0f0)))
             (th 0.0f0))
         (declare (type float th0 th))
         (setf th (* th0 s))
         (return (normalize (+ (* a (/ (sin (- th0 th)) (sin th0)))
                               (* b (/ (sin th) (sin th0)))))))))

   "// --- marker table (cities, generated from the Lisp *markers* list) ---"
   (defun markerDir (i)
     (declare (type int i) (values vec3))
     ,@(index-dispatch 'i (mapcar #'vec3-lit (marker-dirs)) '(vec3 0.0f0 0.0f0 1.0f0)))

   (defun markerCol (i)
     (declare (type int i) (values vec3))
     ,@(index-dispatch 'i (mapcar (lambda (m) (vec3-lit (getf m :color))) *markers*)
                       '(vec3 1.0f0 1.0f0 1.0f0)))

   "// Camera orientations solved in Lisp: qRot(q, markerDir(i)) == (0,0,1)"
   (defun markerQuat (i)
     (declare (type int i) (values vec4))
     ,@(index-dispatch 'i (mapcar #'vec4-lit (marker-quats)) '(vec4 0.0f0 0.0f0 0.0f0 1.0f0)))

   "// --- camera tour: dwell on a marker, then slerp to the next one ---"
   (defun tourPhase (tt)
     (declare (type float tt) (values vec2))
     (let ((g (/ tt TOUR_SEG)))
       (declare (type float g))
       (return (vec2 (floor g) (fract g)))))

   (defun tourEase (ph)
     (declare (type float ph) (values float))
     (return (smoothstep 0.42f0 1.0f0 ph)))

   (defun tourQuat (tt)
     (declare (type float tt) (values vec4))
     (let ((seg (tourPhase tt)))
       (declare (type vec2 seg))
       (let ((i0 (int (mod seg.x MARKER_NF)))
             (i1 (int (mod (+ seg.x 1.0f0) MARKER_NF)))
             (e (tourEase seg.y)))
         (declare (type int i0 i1) (type float e))
         (let ((q (qSlerp (markerQuat i0) (markerQuat i1) e))
               ;; a slow idle drift keeps the globe alive while dwelling
               (dq (qMul (qAxisAngle (vec3 0.0f0 1.0f0 0.0f0) (* 0.035f0 (sin (* tt 0.55f0))))
                         (qAxisAngle (vec3 1.0f0 0.0f0 0.0f0) (* 0.030f0 (sin (+ (* tt 0.37f0) 1.1f0)))))))
           (declare (type vec4 q dq))
           (return (normalize (qMul dq q)))))))

   "// (activeMarker marker, phase, camera distance, travel ease)"
   (defun tourAux (tt)
     (declare (type float tt) (values vec4))
     (let ((seg (tourPhase tt)))
       (declare (type vec2 seg))
       (let ((e (tourEase seg.y)))
         (declare (type float e))
         (return (vec4 (mod seg.x MARKER_NF)
                       seg.y
                       (+ 3.50f0 (* 0.60f0 (sin (* PI e))))
                       e)))))
   )
 :format t :tidy nil :omit-parens t)

;; -------------------------------------------------------------------------
;; 4b. buf0.glsl -- state pass: camera choreography + baked data unpacking
;;
;; Runs once per frame over the whole viewport, but every texel touches at most
;; one word of the packed tables, so the expensive const-array indexing happens
;; O(1) times per texel instead of 16 times per screen pixel in the render pass.
;; -------------------------------------------------------------------------
(write-source
 *buf0-file*
 `(do0
   "// ================================================================="
   "// EARTH GLOBE -- state pass (generated by gen4.lisp)"
   "//   texel (0,0)          : globe orientation quaternion"
   "//   texel (1,0)          : (activeMarker marker, phase, camera distance, ease)"
   "//   rows LAND_Y..+LAND_H : land coverage in .r"
   "//   rows SMALL_Y..+32    : coast (.r), albedo (.rgb), night lights (.r)"
   "// Baked datasets: Natural Earth 110m land, NASA Blue Marble albedo,"
   "// NASA Earth's City Lights.  See tools/bake_earth_data.py"
   "// ================================================================="
   ,(glsl-uint-array "EARTH_LAND" *land-mask-words*)
   ,(glsl-uint-array "EARTH_COAST" *coast-words*)
   ,(glsl-uint-array "EARTH_ALB" *albedo-words* 6)
   ,(glsl-uint-array "EARTH_LIT" *lights-words*)

   "// 2 bits per cell: sub-cell land coverage 0, 1/3, 2/3, 1"
   (defun landCovRaw (ix iy)
     (declare (type int ix iy) (values float))
     (let ((idx (+ (* iy LAND_W) ix))
           (w (uint 0))
           (sh (uint 0)))
       (declare (type int idx) (type uint w sh))
       (setf w (aref EARTH_LAND (>> idx 4))
             sh (uint (* (& idx 15) 2)))
       (return (* 0.33333334f0 (float (& (>> w sh) (uint 3)))))))

   "// 4 bits per cell: gaussian blurred land coverage (continental shelf)"
   (defun coastRaw (ix iy)
     (declare (type int ix iy) (values float))
     (let ((idx (+ (* iy SMALL_W) ix))
           (v (uint 0))
           (sh (uint 0)))
       (declare (type int idx) (type uint v sh))
       (setf v (aref EARTH_COAST (>> idx 3))
             sh (uint (* (& idx 7) 4)))
       (return (/ (float (& (>> v sh) (uint 15))) 15.0f0))))

   "// RGB565, two cells per word"
   (defun albRaw (ix iy)
     (declare (type int ix iy) (values vec3))
     (let ((idx (+ (* iy SMALL_W) ix))
           (w (uint 0))
           (v (uint 0)))
       (declare (type int idx) (type uint w v))
       (setf w (aref EARTH_ALB (>> idx 1))
             v (& (>> w (uint (* (& idx 1) 16))) (uint 65535)))
       (return (vec3 (/ (float (& (>> v (uint 11)) (uint 31))) 31.0f0)
                     (/ (float (& (>> v (uint 5)) (uint 63))) 63.0f0)
                     (/ (float (& v (uint 31))) 31.0f0)))))

   "// 4 bits per cell: night time light emission"
   (defun litRaw (ix iy)
     (declare (type int ix iy) (values float))
     (let ((idx (+ (* iy SMALL_W) ix))
           (v (uint 0))
           (sh (uint 0)))
       (declare (type int idx) (type uint v sh))
       (setf v (aref EARTH_LIT (>> idx 3))
             sh (uint (* (& idx 7) 4)))
       (return (/ (float (& (>> v sh) (uint 15))) 15.0f0))))

   (defun mainImage (fragColor fragCoord)
     (declare (type "out vec4" fragColor)
              (type "in vec2" fragCoord)
              (values void))
     (let ((ipx (ivec2 fragCoord))
           (outv (vec4 0.0f0))
           (ly 0) (sy 0) (sx 0))
       (declare (type ivec2 ipx) (type vec4 outv) (type int ly sy sx))
       (setf ly (- ipx.y LAND_Y)
             sy (- ipx.y SMALL_Y))

       "// --- camera choreography ---"
       (when (== ipx (ivec2 0 0))
         (setf outv (tourQuat iTime)))
       (when (== ipx (ivec2 1 0))
         (setf outv (tourAux iTime)))

       "// --- land coverage block ---"
       (when (logand (>= ly 0) (< ly LAND_H) (< ipx.x LAND_W))
         (setf outv (vec4 (landCovRaw ipx.x ly))))

       "// --- coast / albedo / night light blocks ---"
       (when (logand (>= sy 0) (< sy SMALL_H))
         (setf sx (- ipx.x COAST_X))
         (when (logand (>= sx 0) (< sx SMALL_W))
           (setf outv (vec4 (coastRaw sx sy))))
         (setf sx (- ipx.x ALB_X))
         (when (logand (>= sx 0) (< sx SMALL_W))
           (setf outv (vec4 (albRaw sx sy) 1.0f0)))
         (setf sx (- ipx.x LIT_X))
         (when (logand (>= sx 0) (< sx SMALL_W))
           (setf outv (vec4 (litRaw sx sy)))))

       (setf fragColor outv)))
   )
 :format t :tidy nil :omit-parens t)

;; -------------------------------------------------------------------------
;; 4c. main_image.glsl -- the planet renderer
;;
;; Style note: the generated GLSL functions keep a *flat* single-scope layout
;; (one `let` with all locals, then a sequence of assignments).  This mirrors
;; how a GPU compiler sees the code and keeps the S-expressions readable.
;; -------------------------------------------------------------------------
(write-source
 *main-file*
 `(do0
   "// ================================================================="
   "// EARTH GLOBE -- render pass (generated by gen4.lisp)"
   "// Baked datasets: Natural Earth 110m land, NASA Blue Marble albedo,"
   "// NASA Earth's City Lights.  See tools/bake_earth_data.py"
   "// ================================================================="
   "// ------------- baked Earth data, read back from the state pass -------"
   "// buf0 unpacked the tables into iChannel0 this frame; a texture read is"
   "// an order of magnitude cheaper here than indexing a const uint array."
   (defun landCov (ix iy)
     (declare (type int ix iy) (values float))
     (return (dot (texelFetch iChannel0
                              (ivec2 (& ix (- LAND_W 1))
                                     (+ LAND_Y (clamp iy 0 (- LAND_H 1))))
                              0)
                  x)))

   (defun coastTexel (ix iy)
     (declare (type int ix iy) (values float))
     (return (dot (texelFetch iChannel0
                              (ivec2 (+ COAST_X (& ix (- SMALL_W 1)))
                                     (+ SMALL_Y (clamp iy 0 (- SMALL_H 1))))
                              0)
                  x)))

   (defun albTexel (ix iy)
     (declare (type int ix iy) (values vec3))
     (return (dot (texelFetch iChannel0
                              (ivec2 (+ ALB_X (& ix (- SMALL_W 1)))
                                     (+ SMALL_Y (clamp iy 0 (- SMALL_H 1))))
                              0)
                  xyz)))

   (defun litTexel (ix iy)
     (declare (type int ix iy) (values float))
     (return (dot (texelFetch iChannel0
                              (ivec2 (+ LIT_X (& ix (- SMALL_W 1)))
                                     (+ SMALL_Y (clamp iy 0 (- SMALL_H 1))))
                              0)
                  x)))

   "// bilinear land coverage in [0,1]; u wraps around the globe"
   (defun landField (uv)
     (declare (type vec2 uv) (values float))
     (let ((x (- (* (fract uv.x) (float LAND_W)) 0.5f0))
           (y (- (* (clamp uv.y 0.0f0 1.0f0) (float LAND_H)) 0.5f0))
           (x0 0) (y0 0) (fx 0.0f0) (fy 0.0f0))
       (declare (type float x y fx fy) (type int x0 y0))
       (setf x0 (int (floor x))
             y0 (int (floor y))
             fx (- x (float x0))
             fy (- y (float y0)))
       (return (mix (mix (landCov x0 y0) (landCov (+ x0 1) y0) fx)
                    (mix (landCov x0 (+ y0 1)) (landCov (+ x0 1) (+ y0 1)) fx)
                    fy))))

   "// gaussian-blurred land coverage: 1 = deep inland, ~0.5 = coastline,"
   "// 0 = open ocean.  Drives the continental shelf colour."
   (defun coastField (uv)
     (declare (type vec2 uv) (values float))
     (let ((x (- (* (fract uv.x) (float SMALL_W)) 0.5f0))
           (y (- (* (clamp uv.y 0.0f0 1.0f0) (float SMALL_H)) 0.5f0))
           (x0 0) (y0 0) (fx 0.0f0) (fy 0.0f0))
       (declare (type float x y fx fy) (type int x0 y0))
       (setf x0 (int (floor x))
             y0 (int (floor y))
             fx (- x (float x0))
             fy (- y (float y0))
             fx (* fx fx (- 3.0f0 (* 2.0f0 fx)))
             fy (* fy fy (- 3.0f0 (* 2.0f0 fy))))
       (return (mix (mix (coastTexel x0 y0) (coastTexel (+ x0 1) y0) fx)
                    (mix (coastTexel x0 (+ y0 1)) (coastTexel (+ x0 1) (+ y0 1)) fx)
                    fy))))

   (defun albField (uv)
     (declare (type vec2 uv) (values vec3))
     (let ((x (- (* (fract uv.x) (float SMALL_W)) 0.5f0))
           (y (- (* (clamp uv.y 0.0f0 1.0f0) (float SMALL_H)) 0.5f0))
           (x0 0) (y0 0) (fx 0.0f0) (fy 0.0f0))
       (declare (type float x y fx fy) (type int x0 y0))
       (setf x0 (int (floor x))
             y0 (int (floor y))
             fx (- x (float x0))
             fy (- y (float y0)))
       (return (mix (mix (albTexel x0 y0) (albTexel (+ x0 1) y0) fx)
                    (mix (albTexel x0 (+ y0 1)) (albTexel (+ x0 1) (+ y0 1)) fx)
                    fy))))

   (defun litField (uv)
     (declare (type vec2 uv) (values float))
     (let ((x (- (* (fract uv.x) (float SMALL_W)) 0.5f0))
           (y (- (* (clamp uv.y 0.0f0 1.0f0) (float SMALL_H)) 0.5f0))
           (x0 0) (y0 0) (fx 0.0f0) (fy 0.0f0))
       (declare (type float x y fx fy) (type int x0 y0))
       (setf x0 (int (floor x))
             y0 (int (floor y))
             fx (- x (float x0))
             fy (- y (float y0)))
       (return (mix (mix (litTexel x0 y0) (litTexel (+ x0 1) y0) fx)
                    (mix (litTexel x0 (+ y0 1)) (litTexel (+ x0 1) (+ y0 1)) fx)
                    fy))))

   "// --------------------------- noise --------------------------------"
   (defun hash31 (p)
     (declare (type vec3 p) (values float))
     (let ((q (fract (+ (* p 0.3183099f0) (vec3 0.11f0 0.17f0 0.13f0)))))
       (declare (type vec3 q))
       (setf q (* q 17.0f0))
       (return (fract (* q.x q.y q.z (+ q.x q.y q.z))))))

   (defun vnoise (x)
     (declare (type vec3 x) (values float))
     (let ((i (floor x))
           (f (fract x)))
       (declare (type vec3 i f))
       (setf f (* f f (- 3.0f0 (* 2.0f0 f))))
       (return (mix (mix (mix (hash31 (+ i (vec3 0.0f0 0.0f0 0.0f0)))
                              (hash31 (+ i (vec3 1.0f0 0.0f0 0.0f0))) f.x)
                         (mix (hash31 (+ i (vec3 0.0f0 1.0f0 0.0f0)))
                              (hash31 (+ i (vec3 1.0f0 1.0f0 0.0f0))) f.x) f.y)
                    (mix (mix (hash31 (+ i (vec3 0.0f0 0.0f0 1.0f0)))
                              (hash31 (+ i (vec3 1.0f0 0.0f0 1.0f0))) f.x)
                         (mix (hash31 (+ i (vec3 0.0f0 1.0f0 1.0f0)))
                              (hash31 (+ i (vec3 1.0f0 1.0f0 1.0f0))) f.x) f.y)
                    f.z))))

   "// Rotating every octave breaks the axis aligned value noise lattice"
   "// (otherwise the deserts show a corduroy pattern at high resolution)."
   "const mat3 NROT = mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);"
   "// fractal brownian motion, result roughly in [0,1]"
   (defun fbmN (p oct)
     (declare (type vec3 p) (type int oct) (values float))
     (let ((a 0.5f0) (s 0.0f0) (q p))
       (declare (type float a s) (type vec3 q))
       (for ("int i = 0" (< i 8) (incf i))
            (when (>= i oct) break)
            (incf s (* a (vnoise q)))
            (setf q (+ (* NROT q 2.03f0) (vec3 1.7f0 9.2f0 3.3f0))
                  a (* a 0.5f0)))
       (return s)))

   "// ridged multifractal -> mountain chains"
   (defun ridgedN (p oct)
     (declare (type vec3 p) (type int oct) (values float))
     (let ((a 0.5f0) (s 0.0f0) (q p) (n 0.0f0))
       (declare (type float a s n) (type vec3 q))
       (for ("int i = 0" (< i 8) (incf i))
            (when (>= i oct) break)
            (setf n (- (* 2.0f0 (vnoise q)) 1.0f0))
            (incf s (* a (- 1.0f0 (abs n))))
            (setf q (+ (* NROT q 2.11f0) (vec3 4.4f0 1.9f0 7.1f0))
                  a (* a 0.5f0)))
       (return s)))

   "// ---------------------- geography helpers -------------------------"
   "// earth-fixed direction -> equirectangular texture coordinate"
   (defun dirToUV (d)
     (declare (type vec3 d) (values vec2))
     (let ((lat (asin (clamp d.y -1.0f0 1.0f0)))
           (lon (atan d.x (+ d.z 1.0f-7))))
       (declare (type float lat lon))
       (return (vec2 (+ (/ lon (* 2.0f0 PI)) 0.5f0) (- 0.5f0 (/ lat PI))))))

   "// terrain elevation proxy used for bump mapping and snow lines"
   (defun terrainH (ld)
     (declare (type vec3 ld) (values float))
     (return (+ (* 0.55f0 (fbmN (* ld 3.4f0) 4))
                (* 0.45f0 (ridgedN (+ (* ld 11.0f0) (vec3 3.1f0 0.7f0 5.3f0)) 5)))))

   "// animated cloud cover in [0,1]: large weather systems eroded by fine"
   "// detail, drifting on latitude dependent jet streams."
   (defun cloudAt (ld)
     (declare (type vec3 ld) (values float))
     (let ((lat (asin (clamp ld.y -1.0f0 1.0f0)))
           (lon (atan ld.x (+ ld.z 1.0f-7)))
           (cd (vec3 0.0f0))
           (w (vec3 0.0f0))
           (base 0.0f0) (fine 0.0f0) (wisp 0.0f0) (cov 0.0f0) (dens 0.0f0))
       (declare (type float lat lon base fine wisp cov dens) (type vec3 cd w))
       (setf lon (+ lon (* iTime (+ 0.012f0 (* 0.020f0 (sin (* lat 5.0f0))))))
             cd (vec3 (* (cos lat) (sin lon)) (sin lat) (* (cos lat) (cos lon)))
             ;; domain warp -> swirling fronts and cyclones
             w (- (vec3 (fbmN (+ (* cd 2.2f0) (vec3 0.0f0 0.0f0 (* iTime 0.021f0))) 3)
                        (fbmN (+ (* cd 2.2f0) (vec3 5.2f0 1.3f0 (* iTime 0.019f0))) 3)
                        (fbmN (+ (* cd 2.2f0) (vec3 9.1f0 7.7f0 (* iTime 0.023f0))) 3))
                  0.5f0)
             base (fbmN (+ (* cd 3.6f0) (* w 1.5f0) (vec3 0.0f0 (* iTime 0.012f0) 0.0f0)) 5)
             fine (fbmN (+ (* cd 14.0f0) (* w 3.0f0) (vec3 (* iTime 0.03f0) 0.0f0 0.0f0)) 4)
             wisp (fbmN (+ (* cd 32.0f0) (vec3 0.0f0 0.0f0 (* iTime 0.05f0))) 3)
             ;; climate bands: humid ITCZ and storm belts, dry subtropics
             cov (- 0.568f0
                    (* 0.050f0 (cos (* lat 6.2f0)))
                    (* 0.030f0 (sin (abs lat))))
             dens (+ base (* 0.20f0 (- fine 0.5f0)))
             dens (smoothstep cov (+ cov 0.155f0) dens)
             ;; erode the blobs so edges stay wispy instead of milky
             dens (* dens (+ 0.45f0 (* 0.55f0 fine)) (+ 0.70f0 (* 0.30f0 wisp))))
       (return (clamp (* dens 1.45f0) 0.0f0 1.0f0))))

   "// ---------------------- background star field ---------------------"
   (defun starField (rd)
     (declare (type vec3 rd) (values vec3))
     (let ((col (vec3 0.002f0 0.0035f0 0.007f0))
           (bandAxis (normalize (vec3 0.35f0 0.55f0 -0.75f0)))
           (bd 0.0f0)
           (sc 0.0f0)
           (g (vec3 0.0f0))
           (id (vec3 0.0f0))
           (fp (vec3 0.0f0))
           (off (vec3 0.0f0))
           (h 0.0f0) (dd 0.0f0) (tw 0.0f0) (tint 0.0f0))
       (declare (type vec3 col bandAxis g id fp off)
                (type float bd sc h dd tw tint))
       ;; faint galactic band
       (setf bd (/ ("dot" rd bandAxis) 0.33f0))
       (incf col (* (vec3 0.045f0 0.040f0 0.075f0)
                    (exp (- (* bd bd)))
                    (+ 0.35f0 (* 0.65f0 (fbmN (* rd 5.0f0) 4)))))
       ;; three layers of stars
       (for ("int k = 0" (< k 3) (incf k))
            (setf sc (* 70.0f0 (+ 1.0f0 (* 1.35f0 (float k))))
                  g (* rd sc)
                  id (floor g)
                  fp (- (fract g) 0.5f0)
                  h (hash31 (+ id (* 13.7f0 (float k)))))
            (when (> h 0.945f0)
              (setf off (- (vec3 (hash31 (+ id (vec3 1.7f0 0.3f0 5.1f0)))
                                 (hash31 (+ id (vec3 3.3f0 8.2f0 1.9f0)))
                                 (hash31 (+ id (vec3 5.9f0 2.4f0 7.3f0))))
                           0.5f0)
                    dd (length (- fp (* off 0.65f0)))
                    tw (+ 0.55f0 (* 0.45f0 (sin (+ (* iTime 2.2f0) (* h 90.0f0)))))
                    tint (hash31 (+ id (vec3 0.5f0 0.5f0 0.5f0))))
              (incf col (* (mix (vec3 0.65f0 0.78f0 1.0f0) (vec3 1.0f0 0.85f0 0.62f0) tint)
                           (exp (* -900.0f0 dd dd))
                           tw
                           (* 26.0f0 (- h 0.945f0))))))
       (return col)))

   "// ------------------- marker geometry helpers ----------------------"
   "// closest distance between the view ray and the line segment [a,b]"
   (defun raySegDist (ro rd a b)
     (declare (type vec3 ro rd a b) (values float))
     (let ((v (- b a))
           (w0 (- ro a))
           (a12 0.0f0) (a22 0.0f0) (b1 0.0f0) (b2 0.0f0)
           (det 0.0f0) (s 0.0f0) (tt 0.0f0)
           (pt (vec3 0.0f0)))
       (declare (type vec3 v w0 pt) (type float a12 a22 b1 b2 det s tt))
       (setf a12 (- ("dot" rd v))
             a22 ("dot" v v)
             b1 (- ("dot" rd w0))
             b2 ("dot" v w0)
             det (- a22 (* a12 a12)))
       (when (> det 1.0f-6)
         (setf s (/ (- b2 (* a12 b1)) det)))
       (setf s (clamp s 0.0f0 1.0f0)
             pt (+ a (* v s))
             tt (max ("dot" rd (- pt ro)) 0.0f0))
       (return (length (- (+ ro (* rd tt)) pt)))))

   "// glowing rings painted onto the planet surface at every marker"
   (defun markerSurface (ld activeMarker)
     (declare (type vec3 ld) (type float activeMarker) (values vec3))
     (let ((acc (vec3 0.0f0))
           (mi (vec3 0.0f0))
           (ci (vec3 0.0f0))
           (hi 0.0f0) (ang 0.0f0) (core 0.0f0) (ph 0.0f0)
           (halo 0.0f0) (rr 0.0f0) (e1 0.0f0) (e2 0.0f0))
       (declare (type vec3 acc mi ci)
                (type float hi ang core ph halo rr e1 e2))
       ,@(loop for i from 0 below *marker-count*
               append
               `((setf mi ,(vec3-lit (nth i (marker-dirs)))
                       ci ,(vec3-lit (getf (nth i *markers*) :color))
                       hi (? (< (abs (- activeMarker ,(f32 i))) 0.5f0) 1.0f0 0.35f0)
                       ang (acos (clamp ("dot" ld mi) -1.0f0 1.0f0)))
                 (when (< ang 0.30f0)
                   (setf core (smoothstep 0.016f0 0.007f0 ang)
                         ph (fract (- (* iTime 0.55f0) ,(f32 (* 0.17 i))))
                         halo (* 0.55f0 (exp (* -900.0f0 ang ang)))
                         rr (+ 0.022f0 (* ph 0.155f0))
                         e1 (/ (- ang rr) 0.0075f0)
                         e2 (/ (- ang 0.030f0) 0.0045f0))
                   (incf acc (* ci hi
                                (+ (* core 4.0f0)
                                   halo
                                   (* 2.0f0 (exp (- (* e1 e1))) (- 1.0f0 ph))
                                   (* 1.1f0 (exp (- (* e2 e2)))
                                      (+ 0.55f0 (* 0.45f0 (sin (* iTime 3.0f0)))))))))))
       (return acc)))

   "// vertical light beams standing on the markers"
   (defun markerBeams (ro rd q activeMarker)
     (declare (type vec3 ro rd) (type vec4 q) (type float activeMarker) (values vec3))
     (let ((acc (vec3 0.0f0))
           (wp (vec3 0.0f0))
           (ci (vec3 0.0f0))
           (hi 0.0f0) (facing 0.0f0) (vis 0.0f0) (top 0.0f0) (dseg 0.0f0))
       (declare (type vec3 acc wp ci) (type float hi facing vis top dseg))
       ,@(loop for i from 0 below *marker-count*
               append
               `((setf wp (qRot q ,(vec3-lit (nth i (marker-dirs))))
                       ci ,(vec3-lit (getf (nth i *markers*) :color))
                       hi (? (< (abs (- activeMarker ,(f32 i))) 0.5f0) 1.0f0 0.4f0)
                       facing ("dot" (normalize (- ro wp)) wp))
                 (when (> facing 0.02f0)
                   (setf vis (smoothstep 0.02f0 0.25f0 facing)
                         top (+ 1.0f0 (* 0.26f0 hi))
                         dseg (raySegDist ro rd (* wp 1.001f0) (* wp top)))
                   (incf acc (* ci hi vis
                                (+ (* 0.85f0 (exp (* -1800.0f0 dseg dseg)))
                                   (* 0.18f0 (exp (* -160.0f0 dseg dseg)))))))))
       (return acc)))

   "// ---------------------------- HUD ---------------------------------"
   (defun hudOverlay (suv activeMarker phase)
     (declare (type vec2 suv) (type float activeMarker phase) (values vec3))
     (let ((acc (vec3 0.0f0))
           (p (vec2 (* suv.x (/ iResolution.x iResolution.y)) suv.y))
           (c (vec3 0.0f0))
           (pc (vec2 0.0f0))
           (on 0.0f0) (dd 0.0f0) (play 0.0f0))
       (declare (type vec3 acc c) (type vec2 p pc) (type float on dd play))
       ,@(loop for i from 0 below *marker-count*
               append
               `((setf c ,(vec3-lit (getf (nth i *markers*) :color))
                       pc (vec2 ,(f32 (+ 0.055 (* 0.045 i))) 0.055f0)
                       on (? (< (abs (- activeMarker ,(f32 i))) 0.5f0) 1.0f0 0.0f0)
                       dd (length (- p pc)))
                 (incf acc (* c (+ (* (mix 0.20f0 1.0f0 on) (smoothstep 0.008f0 0.005f0 dd))
                                   (* on 0.30f0 (smoothstep 0.030f0 0.011f0 dd)
                                      (+ 0.45f0 (* 0.55f0 (sin (* iTime 4.0f0))))))))))
       ;; thin tour progress bar underneath the pips
       (setf play (+ 0.055f0 (* 0.045f0 (+ activeMarker phase))))
       (when (logand (> p.x 0.050f0)
                     (< p.x ,(f32 (+ 0.055 (* 0.045 (1- *marker-count*)) 0.005)))
                     (< (abs (- p.y 0.030f0)) 0.0018f0))
         (incf acc (* (vec3 0.55f0 0.72f0 0.95f0) (? (< p.x play) 0.80f0 0.13f0))))
       (return acc)))

   "// ---------------------- planet surface shading --------------------"
   (defun shadeSurface (ro rd tHit q qi sunDir activeMarker)
     (declare (type vec3 ro rd sunDir) (type float tHit activeMarker)
              (type vec4 q qi) (values vec3))
     (let ((pw (+ ro (* rd tHit)))
           (nw (vec3 0.0f0)) (ld (vec3 0.0f0)) (tuv (vec2 0.0f0))
           (wuv (vec2 0.0f0)) (lf 0.0f0) (lb 0.0f0) (cn 0.0f0)
           (isLand 0.0f0) (shelf 0.0f0) (h0 0.0f0) (latAbs 0.0f0)
           (tu (vec3 0.0f0)) (tv (vec3 0.0f0))
           (eps 0.0035f0) (hu 0.0f0) (hv 0.0f0) (bump 0.0f0)
           (nl (vec3 0.0f0)) (nrm (vec3 0.0f0)) (nsea (vec3 0.0f0))
           (wob 0.0f0) (wob2 0.0f0)
           (lcol (vec3 0.0f0)) (ocol (vec3 0.0f0)) (alb (vec3 0.0f0))
           (snow 0.0f0) (ice 0.0f0)
           (sunLocal (vec3 0.0f0)) (cshadow 1.0f0)
           (ndl 0.0f0) (gdl 0.0f0) (wrap 0.0f0) (day 0.0f0)
           (sunCol (vec3 1.0f0 0.955f0 0.90f0))
           (hvec (vec3 0.0f0)) (spec 0.0f0) (lights 0.0f0) (fres 0.0f0)
           (lit (vec3 0.0f0)))
       (declare (type vec3 pw nw ld tu tv nl nrm nsea lcol ocol alb sunLocal
                      sunCol hvec lit)
                (type vec2 tuv wuv)
                (type float lf lb cn isLand shelf h0 latAbs eps hu hv bump
                      wob wob2 snow ice cshadow ndl gdl wrap day spec lights fres))
       (setf nw (normalize pw)
             ld (qRot qi nw)
             tuv (dirToUV ld)
             ;; warp the lookup so the 1.4 deg mask grid never shows as stairs
             wuv (+ tuv
                    (vec2 (* 0.0070f0 (- (fbmN (* ld 6.5f0) 3) 0.5f0))
                          (* 0.0035f0 (- (fbmN (+ (* ld 6.5f0) (vec3 5.0f0 1.0f0 3.0f0)) 3) 0.5f0)))
                    (vec2 (* 0.0022f0 (- (fbmN (* ld 27.0f0) 2) 0.5f0))
                          (* 0.0011f0 (- (fbmN (+ (* ld 27.0f0) (vec3 2.0f0 7.0f0 4.0f0)) 2) 0.5f0))))
             lf (landField wuv)
             lb (coastField wuv)
             cn (- (fbmN (* ld 26.0f0) 4) 0.5f0)
             isLand (smoothstep 0.44f0 0.56f0 (+ lf (* 0.11f0 cn)))
             shelf (* 0.90f0 (smoothstep 0.015f0 0.42f0 lb))
             h0 0.5f0
             latAbs (abs ld.y))

       "// tangent frame (robust at the poles) + wind ruffled water normal"
       (setf tu (normalize (cross (? (< (abs ld.y) 0.90f0)
                                     (vec3 0.0f0 1.0f0 0.0f0)
                                     (vec3 1.0f0 0.0f0 0.0f0))
                                  ld))
             tv (cross ld tu)
             wob (- (vnoise (+ (* ld 260.0f0) (vec3 0.0f0 (* iTime 0.45f0) 0.0f0))) 0.5f0)
             wob2 (- (vnoise (+ (* ld 430.0f0) (vec3 (* iTime 0.35f0) 0.0f0 0.0f0))) 0.5f0)
             nsea (normalize (+ nw (* 0.022f0 (qRot q (+ (* tu wob) (* tv wob2))))))
             nl ld)

       "// terrain bump: differentiate the elevation proxy in the tangent"
       "// frame -- skipped over open water, which is most of the planet"
       (when (> isLand 0.02f0)
         (setf h0 (terrainH ld)
               hu (terrainH (normalize (+ ld (* tu eps))))
               hv (terrainH (normalize (+ ld (* tv eps))))
               bump (* 0.020f0 isLand)
               nl (normalize (- ld (* (+ (* tu (- hu h0)) (* tv (- hv h0)))
                                      (/ bump eps))))))
       (setf nrm (normalize (mix nsea (normalize (qRot q nl)) isLand)))

       "// albedo: baked Blue Marble land colour, analytic ocean, snow & ice"
       (setf lcol (albField tuv)
             lcol (* lcol lcol)
             lcol (* lcol (+ 0.78f0 (* 0.44f0 (fbmN (* ld 34.0f0) 3))))
             snow (smoothstep 0.930f0 0.990f0
                              (+ latAbs (* 0.10f0 (- h0 0.50f0))
                                 (* 0.03f0 (- (fbmN (* ld 11.0f0) 3) 0.5f0))))
             ice (smoothstep 0.935f0 0.985f0
                             (+ latAbs (* 0.05f0 (- (fbmN (* ld 17.0f0) 3) 0.5f0))))
             lcol (mix lcol (vec3 0.86f0 0.90f0 0.95f0) snow)
             ocol (mix (vec3 0.0060f0 0.0265f0 0.0730f0)
                       (vec3 0.030f0 0.160f0 0.215f0)
                       (* shelf shelf))
             ocol (mix ocol (vec3 0.72f0 0.80f0 0.86f0) ice)
             alb (mix ocol lcol isLand))

       "// direct light, soft terminator, cloud shadow"
       (setf sunLocal (qRot qi sunDir)
             ndl ("dot" nrm sunDir)
             gdl ("dot" nw sunDir))
       (when (> gdl -0.10f0)
         (setf cshadow (- 1.0f0 (* 0.30f0 (cloudAt (normalize (+ ld (* sunLocal 0.030f0))))))))
       (setf wrap (clamp (/ (+ ndl 0.18f0) 1.18f0) 0.0f0 1.0f0)
             day (smoothstep -0.20f0 0.16f0 gdl)
             ;; reddened sunlight in the low-sun band around the terminator
             sunCol (mix (vec3 1.0f0 0.62f0 0.34f0) sunCol (smoothstep 0.0f0 0.30f0 gdl))
             lit (* alb sunCol wrap 1.05f0 cshadow))
       (incf lit (* alb (vec3 0.030f0 0.048f0 0.085f0)))

       "// specular sun glint on open water"
       (setf hvec (normalize (- sunDir rd))
             spec (pow (clamp ("dot" nsea hvec) 0.0f0 1.0f0) 900.0f0)
             ;; break the highlight into wave sparkle
             spec (* spec (+ 0.55f0 (* 0.9f0 (vnoise (+ (* ld 900.0f0)
                                                        (vec3 0.0f0 (* iTime 0.8f0) 0.0f0)))))))
       (incf lit (* sunCol spec 1.25f0 (- 1.0f0 isLand)
                    (smoothstep 0.0f0 0.15f0 gdl) cshadow))

       "// night side city lights: gated to the dark side, broken into clusters"
       (setf lights (* (smoothstep 0.22f0 0.80f0 (litField tuv))
                       isLand
                       (pow (- 1.0f0 day) 2.5f0)
                       (+ 0.12f0 (* 1.15f0 (smoothstep 0.42f0 0.80f0 (fbmN (* ld 150.0f0) 2))))))
       (incf lit (* (vec3 1.0f0 0.70f0 0.36f0) lights 1.6f0
                    (+ 0.88f0 (* 0.12f0 (sin (+ (* iTime 6.0f0) (* 40.0f0 tuv.x)))))))

       "// atmospheric limb brightening on the disc"
       (setf fres (pow (- 1.0f0 (clamp ("dot" nw (- rd)) 0.0f0 1.0f0)) 4.2f0))
       (incf lit (* (vec3 0.22f0 0.44f0 1.0f0) fres (+ 0.05f0 (* 0.95f0 day)) 1.10f0))

       (incf lit (markerSurface ld activeMarker))
       (return lit)))

   "// ------------------------- cloud shell ----------------------------"
   "// returns rgb in .xyz and coverage alpha in .w"
   (defun shadeClouds (ro rd bb qi sunDir)
     (declare (type vec3 ro rd sunDir) (type float bb) (type vec4 qi) (values vec4))
     (let ((cc (- ("dot" ro ro) (* CLOUD_R CLOUD_R)))
           (d2 0.0f0) (ts 0.0f0)
           (ps (vec3 0.0f0)) (ns (vec3 0.0f0))
           (cld 0.0f0) (cldSun 0.0f0) (grz 0.0f0) (path 0.0f0) (gdl 0.0f0)
           (alpha 0.0f0) (dayc 0.0f0)
           (ccol (vec3 0.0f0)))
       (declare (type float cc d2 ts cld cldSun grz path gdl alpha dayc)
                (type vec3 ps ns ccol))
       (setf d2 (- (* bb bb) cc))
       (when (< d2 0.0f0)
         (return (vec4 0.0f0)))
       (setf ts (- (- bb) (sqrt d2)))
       (when (< ts 0.0f0)
         (return (vec4 0.0f0)))
       (setf ps (+ ro (* rd ts))
             ns (normalize ps)
             cld (cloudAt (qRot qi ns)))
       (when (< cld 0.004f0)
         (return (vec4 0.0f0)))
       (setf grz (clamp (- ("dot" ns (- rd))) 0.0f0 1.0f0)
             path (min (/ 1.0f0 (max grz 0.38f0)) 1.7f0)
             gdl ("dot" ns sunDir)
             alpha (min (* cld path 0.80f0) 0.94f0)
             cldSun (cloudAt (qRot qi (normalize (+ ns (* sunDir 0.035f0)))))
             dayc (smoothstep -0.22f0 0.16f0 gdl)
             ccol (* (vec3 1.0f0 0.985f0 0.96f0)
                     (+ (* 1.22f0 (clamp gdl 0.0f0 1.0f0)) 0.055f0)
                     (- 1.0f0 (* 0.40f0 cldSun))))
       (incf ccol (* (vec3 0.045f0 0.075f0 0.14f0) (- 1.0f0 dayc)))
       "// grazing rays look through much more cloud and air: brighten towards"
       "// the limb so the shell silhouette blends into the atmosphere ring"
       "// instead of cutting a dark notch out of it"
       (incf ccol (* (vec3 0.42f0 0.60f0 1.0f0)
                     (pow (- 1.0f0 grz) 3.0f0)
                     (+ 0.12f0 (* 0.88f0 dayc))
                     1.15f0))
       (return (vec4 ccol alpha))))

   "// ------------------- atmosphere halo outside the disc -------------"
   (defun atmoHalo (ro rd bb sunDir)
     (declare (type vec3 ro rd sunDir) (type float bb) (values vec3))
     (let ((perp 0.0f0) (hgt 0.0f0) (g 0.0f0) (sunSide 0.0f0)
           (nearDir (vec3 0.0f0)))
       (declare (type float perp hgt g sunSide) (type vec3 nearDir))
       (when (> bb 0.0f0)
         (return (vec3 0.0f0)))
       (setf perp (sqrt (max (- ("dot" ro ro) (* bb bb)) 0.0f0)))
       (when (< perp EARTH_R)
         (return (vec3 0.0f0)))
       (setf nearDir (normalize (+ ro (* rd (- bb))))
             hgt (- perp EARTH_R)
             g (exp (* -21.5f0 hgt))
             sunSide (smoothstep -0.35f0 0.60f0 ("dot" nearDir sunDir)))
       (return (* (vec3 0.24f0 0.46f0 1.0f0) g (+ 0.06f0 (* 1.25f0 sunSide)) 0.95f0))))

   "// ========================== main image ==========================="
   (defun mainImage (fragColor fragCoord)
     (declare (type "out vec4" fragColor)
              (type "in vec2" fragCoord)
              (values void))
     (let ((q (texelFetch iChannel0 (ivec2 0 0) 0))
           (aux (texelFetch iChannel0 (ivec2 1 0) 0))
           (qi (vec4 0.0f0))
           (camDist 3.0f0) (activeMarker 0.0f0) (sunA 0.0f0)
           (uv (/ (- fragCoord (* 0.5f0 iResolution.xy)) iResolution.y))
           (ro (vec3 0.0f0)) (rd (vec3 0.0f0)) (sunDir (vec3 0.0f0))
           (col (vec3 0.0f0))
           (bb 0.0f0) (cc 0.0f0) (disc 0.0f0) (tHit -1.0f0) (vig 0.0f0)
           (clouds (vec4 0.0f0)))
       (declare (type vec4 q aux qi clouds) (type vec2 uv)
                (type vec3 ro rd sunDir col)
                (type float camDist activeMarker sunA bb cc disc tHit vig))
       (setf qi (qConj q)
             camDist (max aux.z 2.2f0)
             activeMarker aux.x
             ro (vec3 0.0f0 0.0f0 camDist)
             rd (normalize (vec3 uv (- 1.62f0)))
             sunA (+ 1.18f0 (* 0.20f0 (sin (* iTime 0.05f0))))
             sunDir (normalize (vec3 (sin sunA) 0.26f0 (cos sunA)))
             bb ("dot" ro rd)
             cc (- ("dot" ro ro) (* EARTH_R EARTH_R))
             disc (- (* bb bb) cc))

       (when (> disc 0.0f0)
         (setf tHit (- (- bb) (sqrt disc))))

       (if (> tHit 0.0f0)
           (setf col (shadeSurface ro rd tHit q qi sunDir activeMarker))
           (setf col (starField rd)))

       (setf clouds (shadeClouds ro rd bb qi sunDir)
             col (mix col clouds.xyz clouds.w))

       "// the halo goes on top of the cloud shell, whose silhouette would"
       "// otherwise paint a dark ring across the blue limb glow"
       (when (< tHit 0.0f0)
         (incf col (atmoHalo ro rd bb sunDir)))

       (incf col (markerBeams ro rd q activeMarker))

       "// tone mapping, vignette, HUD"
       (setf col (max col (vec3 0.0f0))
             col (* col 1.02f0)
             col (/ col (+ (vec3 1.0f0) (* col 0.62f0)))
             col (pow col (vec3 0.4545f0))
             col (mix (vec3 ("dot" col (vec3 0.299f0 0.587f0 0.114f0))) col 1.18f0)
             col (max col (vec3 0.0f0))
             vig (clamp (- 1.0f0 (* 0.28f0 ("dot" uv uv))) 0.0f0 1.0f0)
             col (* col vig))
       (incf col (hudOverlay (/ fragCoord iResolution.xy) activeMarker aux.y))
       "// tiny dither so the smooth ramps do not band in 8 bit"
       (incf col (/ (- (hash31 (vec3 fragCoord (* iTime 60.0f0))) 0.5f0) 255.0f0))
       (setf fragColor (vec4 col 1.0f0))))
   )
 :format t :tidy nil :omit-parens t)

(format t "~%Generated earth shader:~%  ~a~%  ~a~%  ~a~%"
        *common-file* *buf0-file* *main-file*)
