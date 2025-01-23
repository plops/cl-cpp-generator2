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
	 (string ,(format nil "(~a" msg))
	 ,(if (< 0 (length vars))
	   `(std--format
	    (string ,(format nil "~{:~a '{}'~^ ~})\\n"
			     (loop for e in vars
				   collect
				   (emit-c :code e
					   :omit-redundant-parentheses t))))
	    ,@vars)
	   `(string ")\\n"))
	 )
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

  #+nil
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
		  (if-constexpr (space is_shared_ptr
				       (angle (space
					       std--remove_cvref_t
					       (angle Object)))
				       --value)
				(return *object_)
				(return object_)))
		(defmethod strategy ()
		  (if-constexpr
		   (space is_shared_ptr
			  (angle (space
				  std--remove_cvref_t
				  (angle Strategy)))
			  --value)
		   (return *strategy_)
		   (return strategy_)))))
       (space template (angle "typename Object2"
			      "typename Strategy2")
	      (defmethod Implementation (o s)
		(declare (type Object2&& o)
			 (type Strategy2&& s)
			 (construct (object_ (space std--forward
						    ((angle Object2) o)))
				    (strategy_ (space std--forward
						      ((angle Strategy2) s))))))
	      )
       
       "public:"
       (defmethod getTreat ()
	 (declare (override))
	 (dot (strategy)
	      (getTreat (object))))
       (defmethod getPetted ()
	 (declare (override))
	 (dot (strategy)
	      (getPetted (object)))))
  
  (defun create-type-erasure (&key name
				functions
				typenames
				)
    "functions .. ((:name getTreat :return void :params ()) ... )
where params .. ((:pname alpha :type int) ...)"
    (let* ((template `(space-n template
			     (angle ,@(loop for ty in typenames
				collect
				(format nil "typename ~a" ty)))))
	   (make-unique `(space std--make_unique
				(angle
				 (space
				  Implementation
				  (angle
				   ,@(loop for type in typenames
					   collect
					   `(space std--__remove_cvref_t
						   (angle ,type))))))))
	   (type-objects (loop for e in typenames
			       collect
			       (intern (string-upcase (format nil "~a_" e)))))
	   (type-object-accessors (loop for e in typenames
					collect
					(string-downcase (format nil "~a" e))))
	   (typenames2 (loop for e in typenames
			     collect
			     (gensym (format nil "~a" e))))
	   (type-objects2 (loop for e in typenames2
				collect
				(intern (string-upcase (format nil "~a" e)))))
	   )
      `(defclass+ ,name ()
	 (defclass+ Interface ()
	   "public:"
	   (space virtual (~Interface) = default)
	   ,@(loop for e in functions
		   collect
		   (destructuring-bind (&key name return params code) e
		     `(space virtual ,return
			     (,name
			      ,@(loop for p in params
				      collect
				      (destructuring-bind (&key pname type)
					  p
					`(space ,type ,pname)))) = 0))))

	 (space std--unique_ptr (angle Interface) const pimpl)
	 (space template (angle "typename Type")
		struct is_shared_ptr ":" std--false_type (progn))
	 (space template (angle "typename Type")
		struct is_shared_ptr (angle (space std--shared_ptr
						   (angle Type)))
		(progn))
	 (space template
		(angle ,@(loop for ty in typenames
			       collect
			       (format nil "typename ~a" ty)))
		(defclass+ "Implementation final" "public Interface"
		  ;"private:"
		  ,@(loop for type in typenames
			  and object in type-objects
			  collect
			  `(space ,type ,object))
		  "public:"
		  ,@(loop for type in typenames
			  and access in type-object-accessors
			  and object in type-objects
			  collect
			  `(defmethod ,access ()
			     (declare (values auto&))
			     (if-constexpr (space is_shared_ptr
						  (angle (space
							  std--__remove_cvref_t
							  (angle ,type)))
						  --value)
					   (return (deref ,object))
					   (return ,object))))
		  (space template
		(angle ,@(loop for e in typenames2
			       collect
			       (format nil "typename ~a" e)))
		(defmethod Implementation (,@type-objects2)
		  (declare ,@(loop for e in typenames2
				   and f in type-objects2
				   collect `(type ,(format nil "~a&&" e) ,f))
			   (values :constructor)
			   (construct
			    ,@(loop for e in typenames2
				    and f in type-objects2
				    and g in type-objects
				    collect
				    `(,g (space std--forward
						(angle ,e)
						(paren ,f)))))))
		,@(loop for e in functions
		
		 collect
		 (destructuring-bind (&key name return params code) e
		   `(defmethod ,name ()
		      #+nil ,(loop for p in params
				   collect
				   (destructuring-bind (&key pname type)
				       p
				     `(space ,type ,pname)))
		      (declare (override)
			       ;(virtual)
			       (values ,return))
		      ,code)))
		)))
	 
       
	 "public:"
	 
	 (space
	  ,template
		(defmethod ,name (,@type-objects)
		  (declare (values :constructor)
			   ,@(loop for type in typenames
				   and object in type-objects
				   collect
				   `(type ,type ,object))
			   (construct
			    (pimpl
			     (space
			      ,make-unique
			      
			      (paren
			       ,@(loop for type in typenames
				       and object in type-objects
				       collect
				       `(space std--forward
					       (angle ,type)
					       (paren ,object))))))))))
	 #+nil
	 (do0
	  (comments "copy and move constructors")
	  (comments "copy constructor")
	  ,template
	  (defmethod ,name (other)
	    (declare (values :constructor)
		     (type ,(format nil "~a const&" name) other)
		     (construct (pimpl (space
					,make-unique
					#+nil (space std--make_unique
						     (angle Interface))
					
					(paren (deref other.pimpl)))))))

	  (comments "copy assignment operator")
	  ,template
	  (defmethod operator= (other)
	    (declare (values ,(format nil "~a&" name))
		     (type ,(format nil "~a const&" name) other))
	    (when (== this &other)
	      (return *this))
	    (setf *pimpl *other.pimpl)
	    (return *this))
	  (comments "move constructor")
       	  ,template
	  (defmethod ,name (other)
	    (declare (values :constructor)
		     (type ,(format nil "~a&&" name) other)
					;(noexcept)
		     (construct (pimpl (space ,make-unique
					      (paren (std--move *other.pimpl)))))))
	  
	  (comments "move assignment operator")
	  ,template
	  (defmethod operator= (other)
	    (declare (values ,(format nil "~a&" name))
					;(noexcept)
		     (type ,(format nil "~a&&" name) other))
	    (when (== this &other)
	      (return *this))
	    (setf *pimpl (std--move *other.pimpl))
	    (return *this)))
	 
	 ,@(loop for e in functions
		 collect
		 (destructuring-bind (&key name return params code) e
		   `(defmethod ,name ()
		      (declare (const)
			       (values ,return))
		      (-> pimpl (,name)))))
	 ))
    )
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
      iostream
      format
      string
      string_view
      vector
      memory
      type_traits
      )

     ,(create-type-erasure
       :name `UniversalTE
       :functions `((:name getTreat :return void :params () :code (dot (strategy)
								       (getTreat (object))))
		    (:name getPetted :return void :params () :code (dot (strategy)
								       (getPetted (object)))))
       :typenames `(Object Strategy))

     (defclass+ Cat ()
       "public:"
       (space std--string name)
       (defmethod Cat (name)
	 (declare (values :constructor)
		  (type "std::string_view" name)
		  (construct (name name))))
       (defmethod meow ()
	 (declare (const))
	 ,(lprint :msg "meow"))
       (defmethod scratch ()
	 (declare (const))
	 ,(lprint :msg "scratch")))

     (defclass+ PetStrategy1 ()
       "public:"
       (defmethod getTreat (cat)
	 (declare (type "const Cat&" cat))
	 (cat.meow)
	 (cat.scratch))
       (defmethod getPetted (cat)
	 (declare (type "const Cat&" cat))
	 (cat.meow)
	 ))
     
     (defun main ()
       (declare (values int))
       (let ((lazy (Cat (string "lazy")))
	     (s1 (PetStrategy1))
	     #+nil (v (std--vector<UniversalTE>
		 )))
	 "UniversalTE e{lazy,s1};"
	 "std::vector<UniversalTE> v;"
	 (v.emplace_back e)
	 ))

     )
   :omit-parens t
   :format t
   :tidy nil))
