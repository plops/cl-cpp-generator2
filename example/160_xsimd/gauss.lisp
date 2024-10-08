(let ((gen (std--mt19937		;42
	    "std::random_device{}()"))
	     
      (gauss (lambda (			;mu
		      sig)
	       (declare (type Scalar mu sig)
			(values Scalar))
	       (comments "Leva Gaussian Noise with ratio of uniforms")
	       "Scalar u,v,x,y,q; "
	       (space
		do
		(progn
		  (setf u (* 2 (/ (gen) RAND_MAX))
			v (* 1.7156s0 (- (* 2 (/ (gen) RAND_MAX)) .5s0))
			x (- u .449871s0)
			y (+ (std--abs v)
			     .386595s0)
			q (+ (* x x)
			     (* y (- (* .196 y)
				     (* .25472 x))))))
		while (paren (logand (< .27597 q)
				     (paren
				      (logior (< .27846 q)
					      (< (* -4s0
						    (std--log u)
						    u u)
						 (* v v)))))))
	       (return (+		;mu
			(* sig (/ v u))))
	       ))))
