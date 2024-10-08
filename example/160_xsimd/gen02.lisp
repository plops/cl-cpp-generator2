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
  (defparameter *source-dir* #P"example/160_xsimd/source02/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

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
     (include<>
      iostream
      format
      cstddef
      vector
      cmath
      random
      numeric
      memory
      )

     (include "xsimd/xsimd.hpp")

     "using namespace xsimd;"

     "using Scalar = float;"
     "using ScalarI = const Scalar;"
     
     "using XVec = std::vector<Scalar,xsimd::default_allocator<Scalar>>;"
     "using XBatch = xsimd::batch<Scalar,avx2>;"


     "using Vec = std::vector<Scalar>;"
     "using VecI = const Vec;"
  

     (defclass+ Fitab ()
       "public:"
       (defmethod Fitab (xx yy)
	 (declare (type VecI& xx yy)
		  (values :constructor)
		  (construct (ndata (static_cast<int> (xx.size)))
			     (x xx)
			     (y yy)
			     (b .0f)
			     (chi2 .0f)
			     (sigdat .0f)))
	 (let ((sx .0f)
	       (sy .0f))
	   (dotimes (i ndata)
	     (incf sx (aref x i))
	     (incf sy (aref y i))))
	 (let ((ss (static_cast<Scalar> ndata))
	       (sxoss (/ sx ss)))
	  )
	 
	 (let ((st2 .0f)
	       (tt .0f)
	       )
	    (dotimes (i ndata)
	      (incf tt (- (aref x i)
			  sxoss))
	      (incf st2 (* tt tt))
	      (incf b (* tt (aref y i)))))
	 (comments "solve for a, b, sigma_a and sigma_b")
	 (/= b st2)
	 (setf a (/ (- sy (* b sx))
		    ss))
	 (setf siga (std--sqrt (/ (+ 1s0 (/ (* sx sx)
					    (* ss st2)) )
				  ss))
	       sigb (std--sqrt (/ 1s0 st2)))
	 (comments "compute chi2")
	 (dotimes (i ndata)
	   (let ((p (- (aref y i)
		       a
		       (* b (aref x i))))))
	   (incf chi2 (* p p)))
	 (when (< 2 ndata)
	   (setf sigdat (std--sqrt (/ chi2
				      (- (static_cast<Scalar> ndata) 2s0)))))
	 (*= siga sigdat)
	 (*= sigb sigdat)
	 
	 )
      ; "private:"
       "int ndata;"
       "Scalar a, b, siga, sigb, chi2, sigdat;"
       "VecI &x, &y;")
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))

       (let ((gen (std--mt19937 ;42
				"std::random_device{}()"))
	     (dis (std--normal_distribution<float> 0s0 1s0))))

       (let ((lin (lambda (n a b sig repeat)
		   (let (;(n 8)
			 (x (Vec n))
			 (y (Vec n))
			 (fill_x (lambda ()
				   (std--iota (x.begin) (x.end) 0s0)))
			 (fill_y (lambda ()
				   (dotimes (i n)
	      			     (setf (aref y i) (+ (dis gen)
       	       						 b
							 (* a (aref x i))))))))
		     (fill_x)

		     (let ((ares (Vec repeat))
			   (bres (Vec repeat))
			   (stat (lambda (res)
				   (let ((mean 0s0)
					 (std 0s0))
				     (for-range (r res)
						(incf mean r))
				     (/= mean repeat)
				     (for-range (r res)
						(let ((d (- mean r))))
						(incf std (* d d)))
				     (/= std (- repeat 1s0))
				     (setf std (std--sqrt std))
				     (return (std--make_shared<Vec> (curly mean std))))))))

		     (dotimes (j repeat)
		       (fill_y)
		       (let ((f (Fitab x y)))
			 (setf (aref ares j) f.a)
			 (setf (aref bres j) f.b))
		       )
		     (stat ares)
		     (stat bres)))))
	 (dotimes (i 30)
	   (let ((a (+ .3 (* .01 (dis gen))))
		 (b (+ 17 (* .1 (dis gen))))
		 (sig (+ 3 (* .1 (dis gen))))))
	   (lin 18 a b sig 100)))


       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
