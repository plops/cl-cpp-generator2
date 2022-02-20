(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-commonlisp-generator")
  (ql:quickload "alexandria"))
(in-package :cl-commonlisp-generator)



(let ((fn "/home/martin/stage/cl-cpp-generator2/example/72_emsdk/util-pimpl"))
  (write-source
   fn
   `(toplevel
     (defun write-impl-class
	 (&key name dir
	    public-members
	    private-members
	    ,@(loop for e in `(public-headers public-header-preamble public-implementation-preamble
					      private-headers private-header-preamble
					      private-implementation-preamble
					      public-constructor-code public-destructor-code
					      public-code-inside-class  public-code-outside-class
					      private-constructor-code  private-destructor-code
					      private-code-inside-class private-code-outside-class)
		    collect
		    (list e `(quote (comments ,(emit-cl :code `(string ,(format nil "placeholder ~a" e))))))
		    )
	    )

       (string "split class definition in two .h file and two implementations in .cpp file. members ::= (name type [default|init-form] [no-construct])*")
       (let ((public-name (format nil (string "~a") name))
	     (private-name (format nil (string "~aImpl") name)))
	 (write-class
	  :dir dir
	  :name private-name
	  :headers `(do0
		     (include<> memory)
		     ,private-headers)
	  :header-preamble private-header-preamble
	  :implementation-preamble private-implementation-preamble
	  :code `(do0
		  (defclass ,private-name ()
		    ,(string "public:")
		    (format nil (string "class ~a;") private-name)
		    (format nil (string "std::unique_ptr<~a> pimpl;") private-name)

		    ,@(loop for e in private-members
			    collect
			    (destructuring-bind (&key name type init-form default no-construct) e
			      (format nil "~a ~a;" type name)))

		    (defmethod ,private-name (&key
						,@(remove-if
						   #'null
						   (loop for e in def-members
							 collect
							 (destructuring-bind (&key name type init-form default no-construct) e
							   (when default
							     `(,(intern (string-upcase (format nil "~a_" name)))
								,default))))))
		      (declare
		       ,@(remove-if
			  #'null
			  (loop for e in def-members
				collect
				(destructuring-bind (&key name type init-form default no-construct) e
				  (when default
				    `(type ,type ,(intern (string-upcase (format nil (string "~a_") name))))))))

		       (construct
			,@(remove-if
			   #'null
			   (loop for e in def-members
				 collect
				 (destructuring-bind (&key name type init-form default no-construct) e
				   (if init-form
				       `(,name ,init-form)
				       (unless no-construct
					 `(,name ,(intern (string-upcase (format nil (string "~a_") name)))))))))
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
	 ))		  ))
  (sb-ext:run-program "/usr/local/bin/lisp-format"
		      (list "-i"  (format nil "~a.lisp" fn))))

