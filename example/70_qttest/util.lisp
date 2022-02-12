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

(defun write-class (&key name dir code headers header-preamble implementation-preamble preamble moc)
  "split class definition in .h file and implementation in .cpp file. use defclass in code. headers will only be included into the .cpp file. the .h file will get forward class declarations. additional headers can be added to the .h file with header-preamble and to the .cpp file with implementation preamble. if moc is true create moc_<name>.h file from <name>.h"
  (let ((fn-h (format nil "~a/~a.h" dir name))
	(fn-h-nodir (format nil "~a.h" name))
	(fn-moc-h (format nil "~a/moc_~a.cpp" dir name))
	(fn-moc-h-nodir (format nil "moc_~a.cpp" name))
	(fn-cpp (format nil "~a/~a.cpp" dir name)))
    (with-open-file (sh fn-h
			:direction :output
			:if-exists :supersede
			:if-does-not-exist :create)
      (loop for e in `((pragma once)
		       ,@(loop for h in headers
			       collect
			       ;; write forward declaration for classes
			       (format nil "class ~a;" h))
		       ,preamble
		       ,header-preamble
		       )
	    do
	    (when e
	      (format sh "~a~%"
		      (emit-c :code e))))

      (when code
	(emit-c :code
		`(do0
		  ,code)
		:hook-defun #'(lambda (str)
				(format sh "~a~%" str))
		:hook-defclass #'(lambda (str)
                                   (format sh "~a;~%" str))
		:header-only t)))
    (sb-ext:run-program "/usr/bin/clang-format"
                        (list "-i"  (namestring fn-h)
			      "-o"))
    (when moc
      (sb-ext:run-program "/usr/bin/moc-qt5"
                          (list (namestring fn-h)
				"-o" (namestring fn-moc-h))))
    (write-source fn-cpp
		  `(do0
		    ,(if preamble
			 preamble
			 `(comments "no preamble"))
		    ,(if implementation-preamble
			 implementation-preamble
			 `(comments "no implementation preamble"))
		    ,@(loop for h in headers
			    collect
			    `(include ,(format nil "<~a>" h)))
		    #+nil ,(if moc
			 `(include ,(format nil "~a" fn-moc-h-nodir))
			 `(include ,(format nil "~a" fn-h-nodir)))
		    ,(if code
			 code
			 `(comments "no code"))))))
