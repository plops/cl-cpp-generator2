(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; i'm watching this live stream
;; https://www.youtube.com/watch?v=Cfe5UQ-1L9Q
;; LIVE Shader Deconstruction :: happy jumping
;; Inigo Quilez

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/02_shader/source/shader.c"))
  (let* ((code
	  `(do0
	    (defun map (pos)
	      (declare (type "in vec3" pos)
		       (values float))
	      (let ((d (- (length pos) .25)))
		(return d)))
	    (defun calcNormal (pos)
	      (declare (type "in vec3" pos)
		       (values vec3)))
	    (defun mainImage (fragColor fragCoord)
	      (declare (type "out vec4" fragColor)
		       (type "in vec2" fragCoord)
		       (values void))
	      (let ((p (/ (* 2.0
			     (- fragCoord
				iResolution.xy))
			  iResolution.y))
		    (ro (vec3 0.0 0.0 2.0))
		    (rd (normalize (vec3 p -1.5)))
		    (col (vec3 0.0))
		    (tt 0.0))
		(declare (type vec2 p)
			 (type vec3 ro rd col)
			 (type float tt))
		(dotimes (i 100)
		  (let ((pos (+ ro (* tt rd)))
			(h (map pos)))
		    (declare (type vec3 pos)
			     (type float h))
		    (when (< h .001)
		      break)
		    (incf tt h)
		    (when (< 20.0 t)
		      break)))
		(when (< t 20.0)
		  break)
		(setf fragColor (vec4 col 1.0)))
	      ))))
    (write-source *code-file* code)))
