(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; https://www.youtube.com/watch?v=Cfe5UQ-1L9Q
;; LIVE Shader Deconstruction :: happy jumping
;; Inigo Quilez

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/02_shader/source/shader.c"))
  (let* ((code
	  `(do0
	    
	    (defun mainImage (fragColor fragCoord)
	      (declare (type "out vec4" fragColor)
		       (type "in vec2" fragCoord)
		       (values void))
	      ))))
    (write-source *code-file* code)))
