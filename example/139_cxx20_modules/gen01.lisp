(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(let ()
  (defparameter *source-dir* #P"example/139_cxx20_modules/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
    
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "user.cpp"
		     *source-dir*))
   `(do0
     (include<>
      iostream)

     (commens "Stroustrup Tour of C++ (2022) page 35")
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
       		(type char** argv))

       ,(lprint :msg "main entry point" :vars `(argc (aref argv 0)))

       ))
   :omit-parens t
   :format t
   :tidy nil)
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "Vector.cpp"
		     *source-dir*))
   `(do0
     (space export module Vector)
     (space export
	    (defclass+ Vector ()
	      "public:"
	      (defmethod Vector (s)
		(declare (type int s)
			 (construct (elem (new (aref double s)))
				    (sz s))
			 (values :constructor))
		)
	      (defmethod "operator[]" (i)
		(declare (type int i)
			 (values double&))
		(return (aref elem i)))
	      
	      (defmethod size ()
		(declare (values int))
		(return sz))
	      "private:"
	      "double* elem;"
	      "int sz;"))
     (space export
	       (defun "operator==" (v1 v2)
		 (declare (type "const Vector&" v1 v2)
			  
			  (values bool))
		 (unless (== (v1.size)
			     (v2.size))
		   (return false))
		 (dotimes (i (v1.size))
		   (unless (== (aref v1 i)
			       (aref v2 i))
		     (return false))
		   )
		 (return true))))
   :omit-parens t
   :format t
   :tidy nil))

