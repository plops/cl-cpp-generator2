(defun lprint (&key (msg "") (vars nil))
  #+nil `(comments ,msg)
  #-nil`(progn				;do0
	  " "
	  (do0				;let
	   #+nil ((lock (std--unique_lock<std--mutex> ,(g `_stdout_mutex)))
		  )

	   (do0
					;("std::setprecision" 3)
	    (<< "std::cout"
		;;"std::endl"
		("std::setw" 10)
		(dot ("std::chrono::high_resolution_clock::now")
		     (time_since_epoch)
		     (count))
					;,(g `_start_time)

		(string " ")
		("std::this_thread::get_id")
		(string " ")
		__FILE__
		(string ":")
		__LINE__
		(string " ")
		__func__
		(string " ")
		(string ,msg)
		(string " ")
		,@(loop for e in vars appending
			`(("std::setw" 8)
					;("std::width" 8)
			  (string ,(format nil " ~a='" (emit-c :code e)))
			  ,e
			  (string "'")))
		"std::endl"
		"std::flush")))))
