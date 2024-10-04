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
    `(<< std--cout
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
     (comments "Some experiments while reading Lakos: Embracing Modern C++ Safely (2021)")
     (include<>
      format
      iostream
      cstddef
      cstdint
      vector
      array
      atomic
      )
     "std::atomic<int64_t> g_allocCount{0};"

     (defun "operator new" (size)
       (declare (values void*)
		(type "std::size_t" size))
       (g_allocCount.fetch_add size std--memory_order_relaxed)
       (return (malloc size)))
     (defun "operator delete" (p size)
       (declare (type "std::size_t" size)
		(type void* p))
       (g_allocCount.fetch_sub size std--memory_order_relaxed)
       (free p))
     
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

     (comments "emulate named parameters")
     (defclass+ ComputeArgs () 
	      "public:"
       "float sigma{1.2};"
       "int maxIterations{100};"
       "float tol{.001};")
     (defun compute (args)
       (declare (values float)
		(type "const ComputeArgs&" args))
       ,(lprint :vars `(args.sigma args.maxIterations args.tol))
       (return (* args.sigma args.tol args.maxIterations)))
     
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
       
       (let ((vec (std--vector<int> 12))
	     (count 100))
	 (for-range (v vec)
		    (setf v count++)))
       
       (dotimes (i 3s0)
	 (<< std--cout (std--format (string "{}")
				    i)
	     std--endl))

       (dotimes (i 3)
	 (<< std--cout (std--format (string "{}")
				    (aref vec i))
	     std--endl))

       ,(lprint :vars `((g_allocCount.load)))
       
       "constexpr int nVb=  1'000'000;"
       (let ((vb (std--vector<bool> nVb))
	     (sizeofVb (sizeof vb))
	     (sizeVb (vb.size))
	     (bytesVb 0 #+nil (- (ref (aref (vb.data) (- nVb 1)))
				 (ref (aref (vb.data) 0)))))
	 ,(lprint :msg "vector<bool>" :vars `(sizeVb sizeofVb bytesVb)))


       ,(lprint :vars `((g_allocCount.load)))
       "constexpr int nAb=  1'000'000;"
       (let (
	     (ab ("std::array<bool,nAb>" ))
	     (sizeofAb (sizeof ab))
	     (sizeAb (ab.size))
	     (bytesAb (- (ref (aref (ab.data) (- nAb 1)))
			 (ref (aref (ab.data) 0)))))
	 ,(lprint :msg "array<bool>" :vars `(sizeAb sizeofAb bytesAb)))
       ,(lprint :vars `((g_allocCount.load)))

       (comments "i just read section about initializer-lists")
       (compute (curly ))
      (compute (curly .maxIterations=10))
      (compute (curly .maxIterations=20))
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
