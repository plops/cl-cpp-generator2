;;;; =========================================================================
;;;; DSL COMPILER EXAMPLE C: CSG DIFFERENCE AND SMOOTH BLEND
;;;; =========================================================================
(eval-when (:compile-toplevel :execute :load-toplevel)
  (load "gen3.lisp"))

(in-package :cl-cpp-generator2)

(let* ((input '(difference
                (smooth-blend :radius melt_factor
                              (sphere :radius 0.9f0)
                              (torus :radius-major 1.10f0 :radius-minor 0.12f0))
                (cylinder :radius 0.35f0 :height 3.0f0)))
       (compiled (compile-sdf-form input 'p))
       (glsl (emit-c :code compiled)))
  (format t "~%--- Example C Input S-Expression ---~%")
  (format t "~S~%" input)
  (format t "~%--- Example C Compiled GLSL Output ---~%")
  (format t "~A~%" glsl))
