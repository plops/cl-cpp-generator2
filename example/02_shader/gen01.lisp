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
		    "// try to learn how to write shaders based on https://www.youtube.com/watch?v=pmS-F6RJhAk"
		    (defun Truchet (p)
		      (declare (type vec2 p)
			       (values vec4))
		      (setf p (- (fract p)
				 .5d0))
		      (let ((d 0d0)
			    ;;distance to center
			    (cd (length p))
			    (col (vec3 0d0)))
			(declare (type float d cd w)
				 (type vec3 col)))
					;(setf col.rg p)
		      ,(let* ((edge-blur .1)
			      (circle-thickness .05))
			 `(incf col (smoothstep ,edge-blur
						,(* -1 edge-blur)
						(- (abs (- cd .5d0))
						   ,circle-thickness)
						)))

		      ,(let ((tile-border .01))
			 `(do0
			 ;; DEBUG: visualize edge of tile
			   (when (or ,@(loop for e in `(x y)
					     appending
					     `((< ,(- .5 tile-border) (dot p ,e))
					       (< (dot p ,e) ,(- tile-border .5)))))
			   (incf col 1d0))
			 ))
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
			(*= uv 3d0)
			(let ((t1 (Truchet uv)))
			  (declare (type vec4 t1))
			  (setf col t1.rgb))
			(setf fragColor (vec4 col 1d0))))))))



