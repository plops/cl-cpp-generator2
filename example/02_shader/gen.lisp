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
	      (let ((d (- (length pos) .25))
		    (d2 (- pos.y (-.25))) ;; plane at vertical offset
		    )
		(declare (type float d d2))
		(return (min d d2))))
	    (defun calcNormal (pos)
	      (declare (type "in vec3" pos)
		       (values vec3))
	      (return (normalize (vec3 (- (map (+ pos e.xyy))
					  (map (- pos e.xyy)))
				       (- (map (+ pos e.xyy))
					  (map (- pos e.xyy)))
				       (- (map (+ pos e.xyy))
					  (map (- pos e.xyy)))
				       ))))
	    (defun castRay (ro rd)
	      (declare (type vec3 ro rd)
		       (values float))
	      (let ((tt 0.0))
		  (declare (type float tt))
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
		 (return tt)))
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
		    (tt (castRay ro rd)))
		(declare (type vec2 p)
			 (type vec3 ro rd col)
			 (type float tt))
		
		(when (< t 20.0)
		  (let ((pos (+ ro (* tt rd)))
			(nor (calcNormal pos))
			;; albedo of grass
			(mate (vec3 .2 .2 .2))
			(sun_dir (normalize (vec3 .8 .4 .2)))
			(sun_sha (step (castRay (+ pos
						   (* .001) nor) sun_dir)))
			(sun_dif (clamp (dot nor sim_dir)
					0.0 1.0))
			(sky_dif (clamp (+ .5
					   (* .5 (dot nor
						      (vec 0.0 1.0 0.0))))
					0.0 1.0)))
		    (declare (type vec3  pos nor sun_dir mate)
			     (type float sun_dif sky_dif))
		    (setf col (* mate
				 ;; sun has around 10
				 (vec3 9.0 8.0 5.0)
				 sun_dif
				 sun_sha))
		    (incf col (* mate
				 ;; fill light around 1
				 (vec3 0.5 .8 0.9)
				 sky_dif))))
		(setf col (pow col (vec3 .4545)))
		(setf fragColor (vec4 col 1.0)))
	      ))))
    (write-source *code-file* code)))
