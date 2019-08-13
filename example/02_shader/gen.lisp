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
	    
	    (defun sdSphere (pos)
	      (declare (type "in vec3" pos)
		       (values float))
	      
	      (let (
		    (k0 (length (/ pos rad)))
		    (k1 (length (/ pos (* rad rad))))
		    )
		(declare (type float k0 k1)
			 )
		(return (/ (* k0 (- k0 1.0))
			   k1))))
	    (defun sdEllipsoid (pos)
	      (declare (type "in vec3" pos)
		       (values float))
	      
	      (let (
		    (k0 (length (/ pos rad)))
		    (k1 (length (/ pos (* rad rad))))
		    )
		(declare (type float k0 k1)
			 )
		(return (/ (* k0 (- k0 1.0))
			   k1))))
	    (defun smin (a b k)
	      (declare (type "in float" a b)
		       (type float k)
		       (values float))
	      (let ((h (max (- k (abs (- a b)))
			    k)))
		(declare (type float h))
		(return (- (min a b)
			   (/ (* h h)
			      (* k 4.0))))))
	    (defun sdGuy (pos)
	      (declare (type "in vec3" pos)
		       (values float))
	      
	      (let ((tt (fract iTime))
		    (y (* (* 4.0 tt)
			  (- 1.0 tt)))
		    (dy (* 4.0 (- 1.0 (* 2.0 tt))))
		    (u (normalize (vec3 1.0 dy)))
		    (v (vec3 -dy 1.0))
		    (cen (vec3 .0 y .0))
		    (sy (+ .5 (* .5 y)))
		    (sz (/ 1.0 sy)) ;; keep volume constant
		    (rad (vec3 .25 (* .25 sy)  (* .25 sz)))
		    (q (- pos cen))
		    (d (sdEllipsoid q rad))
		    (h q ;(- pos (vec3 .0 ))
		      )
		    ;; head
		    (d2 (sdEllipsoid (- h (vec3 .0 .28 .0)
					)
				     (vec3 .2)))
		    (d3 (sdEllipsoid (- h (vec3 .0 .28 .1)
					)
				     (vec3 .2))))
		(declare (type float y tt sy sz dy d d2)
			 (type vec3 cen rad u v q)
			 
			 (values float))
		#+nil (setf q.yz (vec2 (dot u q.yz)
				       (dot v q.yz)))
		(setf d2 (smin d2 d3 .03)
		      d (smin d d2 .1))
		;; eye
		
		(return d)))
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
		    (ww (normalize (- ta ro))) ;; moving camera
		    (uu (normalize (cross ww
					  (vec3 0 1 0))))
		    (vv (normalize (cross uu ww)))
		    (rd (normalize (vec3 (+ (* p.x uu)
					    (* p.y vv)
					    (* 1.5 ww)))))
		    
		    (col (- (vec3 .65 .75 .9) ;; sky gradient
			    (* .5 rd.y)))
		    
		    (tt (castRay ro rd)))
		(declare (type vec2 p)
			 (type vec3 ro rd col)
			 (type float tt))
		(setf col (mix col
			       (vec3 .7 .75 .8)q
			       (exp (* -10 rd.y))))
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
					0.0 1.0))
			;; yellow bounce from floor
			(bou_dif (clamp (+ .5
					   (* .5 (dot nor
						      (vec 0.0 -1.0 0.0))))
					0.0 1.0)))
		    (declare (type vec3  pos nor sun_dir mate)
			     (type float sun_dif sky_dif bou_dif))
		    (setf col (* mate
				 ;; sun has around 10
				 (vec3 7.0 5.0 3.0)
				 sun_dif
				 sun_sha))
		    (incf col (* mate
				 ;; fill light around 1
				 (vec3 0.5 .8 0.9)
				 sky_dif))
		    (incf col (* mate
				 ;; bounce light
				 (vec3 0.7 .3 0.2)
				 bou_dif))))
		(setf col (pow col (vec3 .4545)))
		(setf fragColor (vec4 col 1.0)))
	      ))))
    (write-source *code-file* code)))
