;;;; =========================================================================
;;;; DSL COMPILER EXAMPLE A: DEFORMED PULSING LAVA SPHERE
;;;; =========================================================================
(eval-when (:compile-toplevel :execute :load-toplevel)
  (load "gen3.lisp"))

(in-package :cl-cpp-generator2)

(let* ((input '(sphere :radius (+ 0.85f0 (* 0.10f0 (sin (* iTime 3.0f0))))
                       :deform (noise-displace :amplitude (* heat_intensity 0.15f0) :frequency 3.0f0)))
       (compiled (compile-sdf-form input 'p))
       (glsl (emit-c :code compiled :omit-redundant-parentheses t)))
  (format t "~%--- Example A Input S-Expression ---~%")
  (format t "~S~%" input)
  (format t "~%--- Example A Compiled GLSL Output ---~%")
  (format t "~A~%" glsl))
