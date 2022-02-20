(defun lprint (&key (msg "") (vars nil))
  #+nil `(comments ,msg)
  #-nil`(progn				;do0
	  " "
	  (let
	      ((lock (std--unique_lock<std--mutex> g_stdout_mutex))
	       )

	    (do0
					;("std::setprecision" 3)
	     ;; https://stackoverflow.com/questions/7889136/stdchrono-and-cout

	     "std::chrono::duration<double>  timestamp = std::chrono::high_resolution_clock::now() - g_start_time;"
	     (<< "std::cout"
		 ;;"std::endl"
		 ("std::setw" 10)
		 #+nil (dot ("std::chrono::high_resolution_clock::now")
			    (time_since_epoch)
			    (count)
			    )
		 (dot timestamp
		      (count))
					;,(g `_start_time)

					;(string " ")
					;(dot now (period))
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

(defun write-class (&key name dir code headers header-preamble implementation-preamble preamble )
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
			       )
		    (include ,(format nil "~a" fn-h-nodir))
		    ,(if code
			 code
			 `(comments "no code"))))))




(defun write-impl-class
    ( &key name dir public-members private-members (public-headers
                                                    '(comments
                                                      "placeholder public-headers")) (public-header-preamble
                                                    '(comments
                                                      "placeholder public-header-preamble")) (public-implementation-preamble
                                                    '(comments
                                                      "placeholder public-implementation-preamble")) (private-headers
                                                    '(comments
                                                      "placeholder private-headers")) (private-header-preamble
                                                    '(comments
                                                      "placeholder private-header-preamble")) (private-implementation-preamble
                                                    '(comments
                                                      "placeholder private-implementation-preamble")) (public-constructor-code
                                                    '(comments
                                                      "placeholder public-constructor-code")) (public-destructor-code
                                                    '(comments
                                                      "placeholder public-destructor-code")) (public-code-inside-class
                                                    '(comments
                                                      "placeholder public-code-inside-class")) (public-code-outside-class
                                                    '(comments
                                                      "placeholder public-code-outside-class")) (private-constructor-code
                                                    '(comments
                                                      "placeholder private-constructor-code")) (private-destructor-code
                                                    '(comments
                                                      "placeholder private-destructor-code")) (private-code-inside-class
                                                    '(comments
                                                      "placeholder private-code-inside-class")) (private-code-outside-class
                                                    '(comments
                                                      "placeholder private-code-outside-class")))
  #+nil(&key name dir

	     (public-headers `(comments "public ")) (public-header-preamble "") (public-implementation-preamble "")
	     (private-headers "") (private-header-preamble "") ( private-implementation-preamble "")
	     public-members (public-constructor-code "") (public-destructor-code "") (public-code-inside-class "") (public-code-outside-class "")
	     private-members (private-constructor-code "") (private-destructor-code "") (private-code-inside-class "") (private-code-outside-class ""))

  "split class definition in two .h file and two implementations in .cpp file. members ::= (name type [default|init-form] [no-construct])*"
  (let ((public-name (format nil "~a" name))
	(private-name (format nil "~aImpl" name)))
    (write-class
     :dir dir
     :name private-name
					;:headers
     :header-preamble `(do0

			,private-header-preamble)
     :implementation-preamble private-implementation-preamble
     :code `(do0
	     (comments "code")
	     (defclass ,private-name ()
	       "public:"
	       ,@(loop for e in private-members
		       collect
		       (destructuring-bind (&key name type init-form default no-construct) e
			 (format nil "~a ~a;" type name)))

	       (defmethod ,private-name (&key
					   ,@(remove-if
					      #'null
					      (loop for e in private-members
						    collect
						    (destructuring-bind (&key name type init-form default no-construct) e
						      (when default
							`(,(intern (string-upcase (format nil "~a_" name)))
							   ,default))))))
		 (declare
		  ,@(remove-if
		     #'null
		     (loop for e in private-members
			   collect
			   (destructuring-bind (&key name type init-form default no-construct) e
			     (when default
			       `(type ,type ,(intern (string-upcase (format nil "~a_" name))))))))

		  (construct
		   ,@(remove-if
		      #'null
		      (loop for e in private-members
			    collect
			    (destructuring-bind (&key name type init-form default no-construct) e
			      (if init-form
				  `(,name ,init-form)
				  (unless no-construct
				    `(,name ,(intern (string-upcase (format nil "~a_" name)))))))))
		   )
		  (values :constructor))
		 (do0
		  ,private-constructor-code))

	       (defmethod ,(format nil "~~~a" private-name) ()
		 (declare
		  (values :constructor))
		 (do0
		  ,private-destructor-code))
	       ,private-code-inside-class)
	     ,private-code-outside-class))
    (let ((public-members (let ((type (format nil "std::unique_ptr<~a>" private-name)))
			    (append `((:name pimpl :type ,type
					     :init-form (new (,private-name))
					     #+nil ((lambda ()
						      (declare (values ,type))
						      (let ((s (new (,private-name))))
							(declare (type ,(format nil "~a*" private-name) s))
							(return (,(format nil "std::make_unique<~a>" private-name)
								  s
								  (lambda (o)
								    (declare (type ,(format nil "~a**" private-name)
										   o))
								    (delete *o)
								    ))))))
					     )
				      )
				    public-members))))
      (write-class
       :dir dir
       :name public-name
					;:headers public-headers
       :header-preamble `(do0
			  (include<> memory)
			  ,(format nil "class ~a;" private-name)
			  ,public-header-preamble)
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" private-name))
				  ,public-implementation-preamble)
       :code `(do0
	       (defclass ,public-name ()
		 "public:"
		 (do0
					;,(format nil "std::unique_ptr<~a> pimpl;" private-name)
		  )
		 ,@(loop for e in public-members
			 collect
			 (destructuring-bind (&key name type init-form default no-construct) e
			   (format nil "~a ~a;" type name)))

		 (defmethod ,public-name (&key
					    ,@(remove-if
					       #'null
					       (loop for e in public-members
						     collect
						     (destructuring-bind (&key name type init-form default no-construct) e
						       (when default
							 `(,(intern (string-upcase (format nil "~a_" name)))
							    ,default))))))
		   (declare
		    ,@(remove-if
		       #'null
		       (loop for e in public-members
			     collect
			     (destructuring-bind (&key name type init-form default no-construct) e
			       (when default
				 `(type ,type ,(intern (string-upcase (format nil "~a_" name))))))))

		    (construct
		     ,@(remove-if
			#'null
			(loop for e in public-members
			      collect
			      (destructuring-bind (&key name type init-form default no-construct) e
				(if init-form
				    `(,name ,init-form)
				    (unless no-construct
				      `(,name ,(intern (string-upcase (format nil "~a_" name)))))))))
		     )
		    (values :constructor))
		   (do0
		    ,public-constructor-code))

		 (defmethod ,(format nil "~~~a" public-name) ()
		   (declare
		    (values :constructor))
		   (do0
		    ,public-destructor-code))
		 ,public-code-inside-class)
	       ,public-code-outside-class)))
    ))


(defmacro write-cmake (fn &key code)
  `(let ((fn ,fn))
     (with-open-file (s fn
			:direction :output
			:if-exists :supersede
			:if-does-not-exist :create)
       ,code)
     ;; pip install cmakelang
     (sb-ext:run-program "/home/martin/.local/bin/cmake-format"
			 (list "-i"  (namestring fn)))))
