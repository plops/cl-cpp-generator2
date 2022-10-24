(defun lprint (&key (msg "") (vars nil))
  `(lprint (curly  __FILE__
		   (string ":")
		   ("std::to_string" __LINE__)
		   (string " ")
		   (ref (aref __PRETTY_FUNCTION__ 0)) ;__func__
		   (string " ")
		   (string ,msg)
		   (string " ")
		   ,@(loop for e in vars appending
			   `(	;("std::setw" 8)
					;("std::width" 8)
			     (string ,(format nil " ~a='" (emit-c :code e)))
			     ("std::to_string" ,e)
			     (string "'"))))))
(defun init-lprint ()
  `(defun lprint (il)
     (declare (type "std::initializer_list<std::string>" il))

     "std::chrono::duration<double>  timestamp(0);"
     (setf timestamp (- ("std::chrono::high_resolution_clock::now")
			g_start_time
			))
     (let ((defaultWidth 10))
       (declare (type "const auto" defaultWidth)))
     (<< "std::cout"
	 ("std::setw" defaultWidth)
	 (dot timestamp
	      (count))
	 (string " ")
	 ("std::this_thread::get_id")
	 (string " ")
	 )
     (for-range ((elem :type "const auto&")
		 il)
		(<< "std::cout"

		    elem) )
     (<< "std::cout"
	 "std::endl"
	 "std::flush")))

(defmacro only-write-when-hash-changed (fn str &key (formatter `(sb-ext:run-program "/usr/bin/clang-format"
										    (list "-i"  (namestring ,fn)
											  "-o"))))
  (let ((hash-db (gensym "file-hash")))
    `(progn
       (defparameter ,hash-db (make-hash-table))
       (let ((fn-hash (sxhash ,fn))
	     (code-hash (sxhash ,str)))
	 (multiple-value-bind (old-code-hash exists) (gethash fn-hash ,hash-db)
	   (when (or (not exists)
		     (/= fn-code-hash old-code-hash)
		     (not (probe-file ,fn)))
					;,@body
	     (progn
	       (with-open-file (sh ,fn
				   :direction :output
				   :if-exists :supersede
				   :if-does-not-exist :create)
       		 (format sh "~a" ,str))
	       ,formatter)
	     (setf (gethash fn-hash ,hash-db) code-hash)
	     ))))))

(defun write-class (&key name dir code headers header-preamble implementation-preamble preamble )
  "split class definition in .h file and implementation in .cpp file. use defclass in code. headers will only be included into the .cpp file. the .h file will get forward class declarations. additional headers can be added to the .h file with header-preamble and to the .cpp file with implementation preamble."
  (let* ((fn-h (format nil "~a/~a.h" dir name))
	 (once-guard (string-upcase (format nil "~a_H" name)))
	 (fn-h-nodir (format nil "~a.h" name))
	 (fn-cpp (format nil "~a/~a.cpp" dir name))
	 )
    (let* ((fn-h-str
	    (with-output-to-string (sh)
	      (loop for e in `(,(format nil "#ifndef ~a" once-guard)
				,(format nil "#define ~a~%" once-guard)

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
			:header-only t))
	      (format sh "~%#endif /* !~a */" once-guard))))
      (only-write-when-hash-changed
       fn-h
       fn-h-str))
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
	       	    (include ,(format nil "~a" fn-h-nodir))
		    ,(if code
			 code
			 `(comments "no code"))))))
