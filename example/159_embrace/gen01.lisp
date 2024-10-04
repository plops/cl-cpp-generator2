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
  (defparameter *source-dir* #P"example/159_embrace/source01/src/")
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
      format
      iostream
      cstddef
      cstdint
      )
     (defun calculatePadding (address alignment)
       (declare (type "const char*" address)
		(type "std::size_t" alignment)
		(values "std::size_t"))
       (return (& (- alignment
		     (reinterpret_cast<std--uintptr_t> address))
		  (- alignment 1))))
     
     (space "template<std::size_t N>"
	    
	    (defclass+ MonotonicBuffer ()
	      "public:"
	      "char d_buffer[N]; // fixed-size buffer"
	      "char* d_top_p;    // next available address"
	      (defmethod MonotonicBuffer ()
		(declare (values :constructor)
			 (construct (d_top_p d_buffer))))
	      (space
	       template "<typename T>"
	       (defmethod allocate ()
		 (declare (values void*)
			  )
		 (return 0)))))
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (dotimes (i 100s0)
	 (<< std--cout (std--format (string "{}")
				    i)
	     std--endl))
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
