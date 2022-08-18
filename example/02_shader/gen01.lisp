(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/02_shader/source01/")
  (ensure-directories-exist (asdf:system-relative-pathname
			     'cl-cpp-generator2
			     *source-dir*))

  ;; i want all the floats to be double float so that they don't print as 1.0f in the c source
  ;; this doesn't seem to work, though. i have to define all constants as 1d0
  (let ((common-lisp::*read-default-float-format* 'double-float))



    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"main.cpp"
				    *source-dir*))
		  `(do0
		    "// try to learn how to write shaders based on https://www.youtube.com/watch?v=pmS-F6RJhAk"
		    (defun Hash21 (p)
		      (declare (values float)
			       (type vec2 p))
		      (setf p (fract (* p (vec2 123.22222234d0 132.323d0)))
			    )
		      (incf p ("dot" p (+ p 223.12d0)))
		      (return (fract (* p.x p.y))))
		    (defun Truchet (p)
		      (declare (type vec2 p)
			       (values vec4))
		      

		      (let ((id (floor p))
			    (n (Hash21 id))
			    (col (vec3 0d0))
			    (d 0d0))
			(declare (type vec3 col)
				 (type vec2 id)
				 (type float d n)))
		      (setf p (- (fract p)
				 .5d0))
		      (when (< n .5d0)
			(setf p.xy (vec2 p.x -p.y)))
		      ,@(loop for circle-center-def in `((:name top-right :scale -1)
							 (:name btm-left :scale 1))
			      collect
			      (destructuring-bind (&key name scale) circle-center-def
				(let ((corner-distance `(length (+ p ,(* scale .5d0)))))
				 `(do0
				    
				    (comments ,(format nil "circle around ~a" name))
					;(setf col.rg p)
				    ,(let* ((edge-blur .01d0)
					    (circle-thickness .05d0))
				       `(incf col (smoothstep ,edge-blur
							      ,(* -1 edge-blur)
							      (- (abs (- ,corner-distance .5d0))
								 ,circle-thickness)
							      )))))))

		      ,(let ((tile-border .01d0))
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



