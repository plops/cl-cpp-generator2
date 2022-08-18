(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/02_shader/source01/")
    (ensure-directories-exist (asdf:system-relative-pathname
			       'cl-cpp-generator2
			       *source-dir*))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"main.cpp"
				    *source-dir*))
		  `(do0
		    (defun Truchet (p)
		      (declare (type vec2 p)
			       (values vec4))
		      (let ((d 0d0)
			    (col (vec3 0d0)))
			(declare (type float d)
				 (type vec3 col)))
		      (setf col.rg p)
		      (return (vec4 col d)))
		    (defun mainImage (fragColor fragCoord)
		      (declare (type "out vec4" fragColor)
			       (type "in vec2" fragCoord))
		      (let ((uv (/ (- fragCoord
				      (* .5d0 iResolution.xy))
				   iResolution.y))
			    (col (vec3 0d0)))
			(declare (type vec2 uv)
				 (type vec3 col))
			(*= uv 5d0)
			(let ((t1 (Truchet uv)))
			  (declare (type vec4 t1))
			  (setf col t1.rgb))
			(setf fragColor (vec4 col 1d0))))))))



