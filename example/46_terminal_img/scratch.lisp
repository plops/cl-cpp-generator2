		 (defun emit_color (r g b bg)
		   (declare (type int r g b)
			    (type bool bg)
			    )
		   (if bg
		       (<< std--cout (string
				      "\\x1b[48;2;")
			   r (string ";")
			   g (string ";")
			   b (string "m"))
		       (<< std--cout (string
				      "\\x1b[38;2;")
			   r (string ";")
			   g (string ";")
			   b (string "m")))
		   #+nil ,(flet ((iter (&key
				    (prefixes `(r g b))
				    fun
				    (extra (string "")))
			     (loop for c in prefixes
				   collect
				   (let ((cnew (format nil "~a~a" c extra)
					       ))
				     (funcall fun c cnew)))))
		      `(do0
			
			,@(iter :fun #'(lambda (c cnew)
					 `(setf ,c (clamp_byte ,c))))
			(let (,@(iter :extra "i"
				      :fun #'(lambda (c cnew)
					       `(,cnew (best_index ,c COLOR_STEPS COLOR_STEP_COUNT))))
			      ,@(iter :extra "q"
				      :fun #'(lambda (c cnew)
					       `(,cnew (aref COLOR_STEPS ,(format nil "~ai" c)))))
			      (gray (static_cast<int> (std--round (+ (* .2989s0 r)
								     (* .587s0 g)
								     (* .114s0 b)))))
			      (gri (best_index gray GRAYSCALE_STEPS GRAYSCALE_STEP_COUNT))
			      (grq (aref GRAYSCALE_STEPS gri))
			      (color_index 0))
			  (if (< (+ (* .2989s0 (sqr (- rq r)))
				    (* .587s0 (sqr (- gq g)))
				    (* .114s0 (sqr (- bq b))))
				 (+ (* .2989s0 (sqr (- grq r)))
				    (* .587s0 (sqr (- grq g)))
				    (* .114s0 (sqr (- grq b)))))
			      (setf color_index (+ 16
						   (* 36 ri)
						   (* 6 gi)
						   bi))
			      (setf color_index (+ 232 gri)))
			  )
			(if bg
			    (<< std--cout (string "\\x1B[48;5;")
				color_index (string "m"))
			    (<< std--cout (string "\\x001B[38;5;")
				color_index (string "m")))
			
			)))


	 #+nil (defun best_index (value data[] count)
		   (declare (type int value count)
			    (type "const int" data[])
			    (values int))
		   (let ((result 0)
			 (best_diff (std--abs (- (aref data 0)
						 value))))
		     (for ((= "int i" 1)
			   (< i count)
			   (incf i))
			  (let ((diff (std--abs (- (aref data i)
						   value))))
			    (when (< diff best_diff)
			      (setf result i
				    best_diff diff))))
		     
		     (return result)))
