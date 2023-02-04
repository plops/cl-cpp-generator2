(defun lprint (&key (msg "")
		    (vars nil)

		    )
  `(fmt--print
    (string ,(format nil "~a~{ ~a~}\\n"
		     msg
		     (loop for e in vars
			   collect
			   (format nil " ~a='{}'" (emit-c :code e)))))
    ,@vars
    ))

