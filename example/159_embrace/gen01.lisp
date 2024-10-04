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
		 (let ((padding (calculatePadding d_top_p (alignof T)))
		       (delta (+ padding (sizeof T))))
		   (when (< (- (+ d_buffer
				  N)
			       d_top_p)
			    delta)
		     (comments "not enough properly aligned unused space remaining")
		     (return 0))
		    (let ((alignedAddres (+ d_top_p
						padding)))
			  (incf d_top_p
				delta)
			  (return alignedAddres)))))))
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))

       ,(let ((l-vars `(char double short int bool)))
	    `(let ((mb (MonotonicBuffer<20>))
		   ,@(loop for e in l-vars
			   collect
			   `(,(format nil "~ap" e)
			     (,(format nil "static_cast<~a *>"
				       e)
			      (dot mb
					    (,(format nil "allocate<~a>" e)))))))
	       ,@ (loop for e in l-vars
			collect
			`(<< std--cout
			     (string ,(format nil "~a:" e))
			     (- (reinterpret_cast<char*> ,(format nil "~ap" e))
				(ref (aref mb.d_buffer 0)))
			     std--endl))))
       
       
       
       (dotimes (i 3s0)
	 (<< std--cout (std--format (string "{}")
				    i)
	     std--endl))
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
