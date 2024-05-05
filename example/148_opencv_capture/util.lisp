(defun lprint (&key (msg "")
		 (vars nil)
		 )
  `(<< std--cout
       (std--format
	(string ,(format nil "~a~{ ~a='{}'~}\\n"
			 msg
			 (loop for e in vars collect (emit-c :code e  :omit-redundant-parentheses t)) ))
	,@vars))
  #+nil
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


(defmacro only-write-when-hash-changed (fn str &key (formatter `(sb-ext:run-program "/usr/bin/clang-format"
										    (list "-i"  (namestring ,fn)
											  "-o"))))
  (let ((hash-db 'file-hash
					;(gensym "file-hash")
		 ))
    `(progn
       (defvar
					;  parameter
	   ,hash-db (make-hash-table))
       (let ((fn-hash (sxhash ,fn))
	     (code-hash (sxhash ,str)))
	 (multiple-value-bind (old-code-hash exists) (gethash fn-hash ,hash-db)
	   (when (or (not exists)
		     (/= code-hash old-code-hash)
		     (not (probe-file ,fn)))
					;,@body
	     (progn
	       (format t "hash has changed in ~a exists=~a old=~a new=~a~%" ,fn exists old-code-hash code-hash)
	       (with-open-file (sh ,fn
				   :direction :output
				   :if-exists :supersede
				   :if-does-not-exist :create)
       		 (format sh "~a" ,str))
	       ,formatter)
	     (setf (gethash fn-hash ,hash-db) code-hash)
	     ))))))
(defun share (name)
    (format nil "std::shared_ptr<~a>" name))
(defun uniq (name)
  (format nil "std::unique_ptr<~a>" name))

(defun write-class (&key name dir code headers header-preamble implementation-preamble preamble format)
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
			 :header-only t
			 ))
	       (format sh "~%#endif /* !~a */" once-guard))))
      (if format
	  (only-write-when-hash-changed
	   fn-h
	   fn-h-str
	   )
	  (only-write-when-hash-changed
	   fn-h
	   fn-h-str
	   :formatter nil)))
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
			 `(comments "no code")))
		  :format t
		  :tidy nil
		  :omit-parens t)))


(defun make-class (class-name
		     &key members0 header-preamble
		       implementation-preamble
		       constructor-code
		       class-code)
    (let* ((members (loop for e in members0
			  collect
			  (destructuring-bind (&key name type param (initform 0) setter getter) e
			    `(:name ,name
			      :type ,type
			      :param ,param
			      :initform ,initform
			      :setter ,(if setter setter "")
			      :getter ,(if getter getter "")
			      :member-name ,(intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))
			      :param-name ,(when param
					     (intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))))))))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name class-name
       :headers `()
       :header-preamble header-preamble
       :implementation-preamble implementation-preamble
       :code `(do0
	       (defclass ,class-name ()
		 "public:"
	       	 (defmethod ,class-name ( ,@(remove-if
					     #'null
					     (loop for e in members
						   collect
						   (destructuring-bind (&key name type param initform param-name member-name setter getter) e
						     (unless (and param initform)
						       param-name))))
					 &key
					   ,@(remove-if
					      #'null
					      (loop for e in members
						    collect
						    (destructuring-bind (&key name type param initform param-name member-name setter getter) e
						      (when (and param initform)
							`(,param-name ,initform))))))
		   (declare
		    ,@(remove-if #'null
				 (loop for e in members
				       collect
				       (destructuring-bind (&key name type param initform param-name member-name setter getter) e
					 (when param
					   `(type ,type ,param-name)))))
		    (construct
		     ,@(remove-if #'null
				  (loop for e in members
					collect
					(destructuring-bind (&key name type param initform param-name member-name setter getter) e
					  (cond
					    (param
					     `(,member-name ,param-name)) 
					    (initform
					     `(,member-name ,initform)))))))
					;(explicit)	    
		    (values :constructor))
		   ,constructor-code
		   
		   )
		 ,@class-code
		 
		 ,@(remove-if
		    #'null
	            (loop for e in members
			  appending
			  (destructuring-bind (&key name type param initform param-name member-name setter getter) e
			    (let ((get (cl-change-case:pascal-case (format nil "get-~a" name)))
				  (set (cl-change-case:pascal-case (format nil "set-~a" name)))
				  (const-p (let* ((s  (format nil "~a" type))
						  (last-char (aref s (- (length s)
									1))))
					     (not (eq #\* last-char)))))
			      `((defmethod ,get ()
				  (declare ,@(if const-p
						 `((const)
						   (values ,(format nil "const ~a&" type)))
						 `((values ,type))))
				  ,getter
				  (return ,member-name))
				(defmethod ,set (,member-name)
				  (declare (type ,type ,member-name))
				  (setf (-> this ,member-name)
					,member-name)
				  ,setter))))))
		 "public:"
		 ,@(remove-if #'null
			      (loop for e in members
				    collect
				    (destructuring-bind (&key name type param initform param-name member-name setter getter) e
				      `(space ,type ,member-name)))))))))
