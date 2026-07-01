;;;; =========================================================================
;;;; DSL COMPILER EXAMPLE B: ROTATING TORUS IN WORKSPACE
;;;; =========================================================================
(eval-when (:compile-toplevel :execute :load-toplevel)
  (load "gen3.lisp"))

(in-package :cl-cpp-generator2)

(let* ((input '(torus :radius-major 1.25f0 :radius-minor 0.10f0
                      :transform (rotate-z (* iTime (* spin_speed 0.80f0)))))
       (compiled (compile-sdf-form input 'p))
       (glsl (emit-c :code compiled)))
  (format t "~%--- Example B Input S-Expression ---~%")
  (format t "~S~%" input)
  (format t "~%--- Example B Compiled GLSL Output ---~%")
  (format t "~A~%" glsl))
