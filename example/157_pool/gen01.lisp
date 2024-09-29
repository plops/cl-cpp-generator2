(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ()
  (defparameter *source-dir* #P"example/157_pool/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  ;(load "util.lisp")
  (defun lprint (&key (msg "")
		 (vars nil)
		 )
  #-nil `(<< std--cout
       (std--format
	(string ,(format nil "(~a~{ :~a '{}'~})\\n"
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
      memory_resource
      string
      array vector
      iostream
      iomanip
      algorithm
      format
      )

     "using namespace std;"

     (defclass+ test_resource
       "public pmr::memory_resource"

       "public:"
       (defmethod test_resource (&key (upstream (pmr--get_default_resource)))
	 (declare (type pmr--memory_resource* upstream)
		  (construct (_upstream upstream))
		  (values :constructor))
	 (let ((upstream_ptr (reinterpret_cast<uint64_t> _upstream)))
	  ,(lprint :msg "make_test_resource"
		   :vars `(upstream_ptr))))
       
       (defmethod ~test_resource ()
	 (declare (values :constructor)))

       (defmethod upstream ()
	 (declare (const)
		  (values "pmr::memory_resource*"))
	 (return _upstream))

       (defmethod bytes_allocated ()
	 (declare (const)
		  (values size_t))
	 (return _bytes_allocated))

       (defmethod bytes_deallocated ()
	 (declare (const)
		  (values size_t))
	 (return 0))

       (defmethod bytes_outstanding ()
	 (declare (const)
		  (values size_t))
	 (return _bytes_outstanding))

       (defmethod bytes_highwater ()
	 (declare (const)
		  (values size_t))
	 (return _bytes_highwater))

       (defmethod blocks_outstanding ()
	 (declare (const)
		  (values size_t))
	 (return 0))

       (comments "We can't throw in the destructor that is why we need the following three functions")

       (defmethod leaked_bytes ()
	 (declare (static)
		  (values size_t))
	 (return _s_leaked_bytes))

       (defmethod leaked_blocks ()
	 (declare (static)
		  (values size_t))
	 (return _s_leaked_blocks))

       (defmethod clear_leaked ()
	 (declare (static)
		  (values void))
	 (setf _s_leaked_bytes 0
	       _s_leaked_blocks 0))

       "protected:"

       (defmethod do_allocate (bytes alignment)
	 (declare (override)
		  (type size_t bytes alignment)
		  (values void*))
	 ,(lprint :msg "do_allocate" :vars `(bytes alignment))
	 (let ((ret (_upstream->allocate bytes alignment)))
	   (_blocks.push_back (space allocation_rec(curly  ret bytes alignment)))
	   (incf _bytes_allocated bytes)
	   (incf _bytes_outstanding bytes)
	   (when (< _bytes_highwater _bytes_outstanding)
	     (setf _bytes_highwater _bytes_outstanding))
	   (return ret))
	 )

       (defmethod do_deallocate (p bytes alignment)
	 (declare (override)
		  (type void* p)
		  (type size_t bytes alignment)
		  (values void))
	 ,(lprint :msg "do_deallocate" :vars `(p bytes alignment))
	 (let ((i (std--find_if (_blocks.begin)
				(_blocks.end)
				(lambda (r)
				  (declare (capture p)
					   (type allocation_rec& r))
				  (return (== r._ptr p))))))
	   (cond ((== i (_blocks.end))
		  (throw (std--invalid_argument (string "deallocate: Invalid pointer"))))
		 ((!= bytes i->_bytes)
		  (throw (std--invalid_argument (string "deallocate: Size mismatch")))
		  )
		 ((!= alignment i->_alignment)
		  (throw (std--invalid_argument (string "deallocate: Alignment mismatch")))
		  ))
	   (_upstream->deallocate p i->_bytes i->_alignment)
	   (_blocks.erase i)
	   (decf _bytes_outstanding bytes)))

       (defmethod do_is_equal (other)
	 (declare (override)
		  (const)
		  (noexcept)
		  (type "const pmr::memory_resource&" other)
		  (values bool))
	 (comments "convert for printing")
	 (let ((this_ptr (reinterpret_cast<uint64_t> this))
	       (other_ptr (reinterpret_cast<uint64_t> &other)))
	   ,(lprint :msg "do_is_equal" :vars `(this_ptr other_ptr)))
	 (return (== this &other)))

       "private:"

       (defstruct0 allocation_rec
	   (_ptr void*)
	 (_bytes size_t)
	 (_alignment size_t))


       ,(let ((members `((:name upstream :type "pmr::memory_resource*")
			 (:name bytes_allocated :type size_t)
			 (:name bytes_outstanding :type size_t)
			 (:name bytes_highwater :type size_t)
			 (:name blocks :type "pmr::vector<allocation_rec>")
			 (:name s_leaked_bytes :type "static size_t" :default nil)
			 (:name s_leaked_blocks :type "static size_t" :default nil))))
	  `(do0
	    ,@(loop for e in members
		    collect
		    (destructuring-bind (&key name type (default t)) e
		      (format nil "~a _~a~a;" type name (if default
							    "{}"
							    "")))))))

     (defstruct0 point_2d
	 (x double)
       (y double))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))

       "constexpr int rawN{1'600};"
       
       (let ((raw ("std::array<std::byte,rawN>"))
	     (buf0 (pmr--monotonic_buffer_resource (raw.data) (raw.size) (pmr--null_memory_resource)))
	     (buf (test_resource &buf0))
	     )
	 "constexpr int nPoints{100};"
	 (let ((sizeof_point_2d (sizeof point_2d)))
	  ,(lprint :msg "main" :vars `(rawN nPoints sizeof_point_2d)))
	 (let ((points (pmr--vector<point_2d> nPoints &buf)))
	   (setf (aref points 0)
		 (curly .1 .2)))
	 )
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
