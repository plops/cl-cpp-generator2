(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more ;; command line parsing
						    
						    )))
  (setf *features* (set-exclusive-or *features* (list :more
						      ;:invert
						      ))))

(let ()
  (defparameter *source-dir* #P"example/163_value_based_oop/source04/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (defun begin (arr)
    `(ref (aref ,arr 0)) )
  (defun end (arr)
    `(+ (ref (aref ,arr 0)) (dot ,arr (size))))
  
  (defun lprint (&key (msg "")
		   (vars nil)
		   )
    `(<< std--cout
	 (std--format
	  (string ,(format nil "(~a~{:~a '{}'~^ ~})\\n"
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
		 `((string ,(format nil " ~a='" (emit-c :code e :omit-redundant-parentheses t)))
		   ,e
		   (string "' ")))   
	 std--endl))
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
      iostream
      format
      vector
      memory
      )

     (defclass+ UniversalTE ()
       "private:"
       (space std--unique_ptr (angle Interface) pimpl)
       (defclass+ Interface ()
	 "public:"
	 (space virtual (~Interface) = default)
	 (space virtual void (getTreat) = 0)
	 (space virtual void (getPetted) = 0))
       (space template (angle "typename Type")
	      struct is_shared_ptr ":" std--false_type (progn))
       (space template (angle "typename Type")
	      struct is_shared_ptr (angle (space std--shared_ptr
						 (angle Type)))
	      (progn))
       (space (angle "typename Object"
		     "typename Strategy")
	      (defclass+ Implementation ("public Interface")
		"private:"
		(space Object object_)
		(space Strategy strategy_)
		"public:"
		(defmethod object ()
		  (if (space constexpr (space is_shared_ptr
					(angle (space
						std--remove_cvref_t
						(angle Object)))
					--value))
		      (return *object_)
		      (return object)))
		(defmethod strategy ()
		  (if (space constexpr (space is_shared_ptr
					(angle (space
						std--remove_cvref_t
						(angle Strategy)))
					--value))
		      (return *strategy_)
		      (return strategy)))))
       "public:")


     )
   :omit-parens t
   :format t
   :tidy nil))
