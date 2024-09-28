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
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      memory_resource
      string
      iostream
      iomanip
      )

     "using namespace std;"

     (defclass+ test_resource
       "public pmr::memory_resource"

       "public:"
       (defmethod test_resource (&key (parent (pmr--get_default_resource)))
	 (declare (type pmr--memory_resource* parent)
		  (values :constructor)))
       
       (defmethod ~test_resource ()
	 (declare (values :constructor)))

       (defmethod upstream ()
	 (declare (const)
		  (values "pmr::memory_resource*")))

       
       
       (defmethod bytes_allocated ()
	 (declare (const)
		  (values size_t)))

       (defmethod bytes_deallocated ()
	 (declare (const)
		  (values size_t)))

       (defmethod bytes_outstanding ()
	 (declare (const)
		  (values size_t)))

       (defmethod bytes_highwater ()
	 (declare (const)
		  (values size_t)))

       (defmethod blocks_outstanding ()
	 (declare (const)
		  (values size_t)))

       (comments "We can't throw in the destructor that is why we need the following three functions")

       (defmethod leaked_bytes ()
	 (declare (static)
		  (values size_t)))

       (defmethod leaked_blocks ()
	 (declare (static)
		  (values size_t)))

       (defmethod clear_leaked ()
	 (declare (static)
		  (values void)))

       "protected:"

       (defmethod do_allocate (bytes alignment)
	 (declare (override)
		   (type size_t bytes alignment)
		  (values void*)))

       (defmethod do_deallocate (p bytes alignment)
	 (declare (override)
		  (type void* p)
		  (type size_t bytes alignment)
		  (values void)))

       (defmethod do_is_equal (other)
	 (declare (override)
		  (const)
		  (noexcept)
		  (type "const pmr::memory_resource&" other)
		  (values bool))))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
     
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))
