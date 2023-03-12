(defun lprint (&key (msg "")
		 (vars nil)
		 )
  `(<< std--cout
       (string ,(format nil "~a"
			msg
			
			))
       ,@(loop for e in vars
	       appending
	       `((string ,(format nil " ~a='" (emit-c :code e)))
		 ,e
		 (string "' ")))   
       std--endl))

