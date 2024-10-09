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

(let ((l-fit `(a b siga sigb chi2 sigdat)))
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
      ;cstddef
      vector
      cmath
      random
      numeric
      algorithm
      ;memory
      )

     ;(include "xsimd/xsimd.hpp")

     ;"using namespace xsimd;"

     "using Scalar = float;"
     ;"using ScalarI = const Scalar;"
     
     ;"using XVec = std::vector<Scalar,xsimd::default_allocator<Scalar>>;"
     ;"using XBatch = xsimd::batch<Scalar,avx2>;"


     "using Vec = std::vector<Scalar>;"
     "using VecI = const Vec;"
  

     (comments "From Numerical Recipes")
     (defclass+ Fitab ()
       "public:"
       (defmethod Fitab (xx yy)
	 (declare (type VecI& xx yy)
		  (values :constructor)
		  (construct (ndata (static_cast<int> (xx.size)))
			     (x xx)
			     (y yy)
			     ))
	 #+nil (let ((sx .0f)
		     (sy .0f))
		 (dotimes (i ndata)
		   (incf sx (aref x i))
		   (incf sy (aref y i))))
	 (letc ((sx (std--accumulate (x.begin)
				     (x.end)
				     0s0))))
	 (letc ((sy (std--accumulate (y.begin)
				     (y.end)
				     0s0))))
	 (letc ((ss (static_cast<Scalar> ndata))
		(sxoss (/ sx ss)))
	       )
	 
	 (letc ((st2 (std--accumulate (x.begin)
				      (x.end)
				      0s0
				      (lambda (accum xi)
					(return (+ accum (std--pow (- xi sxoss) 2s0))))) ; .0f
		     )
		)
	      
	        (dotimes (i ndata)
		   
		       (letc ((tt (- (aref x i)
				     sxoss))))
		  
		       (incf b (* tt (aref y i)))
					;,(lprint :vars `(i tt b))
		       ))
	 #+nil  (setf b (std--inner_product (x.begin)
					    (x.end)
					    (y.begin)
					    0s0
					    (lambda (accum value) (return (+ accum value)))
					    (lambda (xi yi) (return (* tt yi)))))
	 (comments "solve for a, b, sigma_a and sigma_b")
	 (/= b st2)
	 (setf a (/ (- sy (* b sx))
		    ss))
	 (setf siga (std--sqrt (/ (+ 1s0 (/ (* sx sx)
					    (* ss st2)) )
				  ss))
	       sigb (std--sqrt (/ 1s0 st2)))
	 (comments "compute chi2")
	 (setf chi2 (std--inner_product
		     (x.begin)
		     (x.end)
		     (y.begin)
		     0s0
		     (lambda (accum value)
		       (return (+ accum value)))
		     (lambda (xi yi)
		       (declare (capture this))
		       (let ((p (- yi a (* b xi))))
			 (return (* p p))))
		     ))
	 #+nil (setf chi2 (std--accumulate (x.begin)
					   (x.end)
					   0s0
					   (lambda (accum xi)
					     ())))
	 #+nil
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
       "int ndata{0};"
       ,(format nil "Scalar ~{~a{.0f}~^, ~};" l-fit)
       "VecI &x, &y;")
     (defun getSignificantDigits (num)
       (declare (type Scalar num)
		(values int))
       (when (== num 0s0)
	 (return 1))
       (when (< num 0)
	 (setf num -num))
       (let ((significantDigits 0))
	 (while (<= num 1s0)
		(*= num 10s0)
		(incf significantDigits))
	 (return significantDigits)))
     (defun printStat (md ;m d
		       )
       (declare ;(type Scalar m d)
	(type "std::pair<Scalar,Scalar>" md)
	(values "std::string"))
       (let (((bracket m d) md)))
       (letc ((precision  (getSignificantDigits d))
	     (fmtm (+ (std--string (string "{:."))
		      (std--to_string (+ 1 precision))
		      (string "f}")))
	    #+nil (fmtd (+ (std--string (string "{:."))
		      (std--to_string precision)
		      (string "f}")))
	     (format_str (+  fmtm (string " ± ") fmtm))
	     )
	 (return (std--vformat format_str (std--make_format_args m d)))
	 ))
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))

       (let ((gen (std--mt19937		;42
		   "std::random_device{}()"))
	     (dis (std--normal_distribution<float> 0s0 1s0))))

       (let ((lin (lambda (n A B Sig repeat)
		    (let (		;(n 8)
			  (x (Vec n))
			  (y (Vec n))
			  (fill_x (lambda ()
				    (std--iota (x.begin) (x.end) 1s0)))
			  #+nil (fill_y
			    (lambda ()
				    (dotimes (i n)
	      			      (setf (aref y i) (+ (* Sig (dis gen))
       	       						  A
							  (* B (aref x i))))))))
		      (fill_x)
		      
		      (let ((stat (lambda (fitres filter)
				    (comments "Numerical Recipes 14.1.2")
				    (letc ((data (Vec (fitres.size)))))
				    (std--transform (fitres.begin)
						    (fitres.end)
						    (data.begin)
						    filter)
				    (letc ((mean (/ (std--accumulate (data.begin)
								   (data.end)
								   0s0)
						    (static_cast<Scalar> (data.size)))))
					  (letc ((sq_sum (std--inner_product (data.begin)
									(data.end)
									(data.begin)
									0s0))
					    (stdev (std--sqrt (- (/ sq_sum
								    (static_cast<Scalar> (data.size)))
								 (* mean mean))))))
				      (return (std--make_pair mean stdev)))))))
		      (let ((generate_fit (lambda ()
					    ;(setf y (curly 2.1s0 2.3s0 2.6s0))
					    
					    (std--transform (x.begin)
							    (x.end)
							    (y.begin)
					    
							    (lambda (xi)
							      (declare (type Scalar xi))
							      (return (+ (* Sig (dis gen))
       	       								 A
									 (* B xi))))
							    )
					    #+nil (fill_y)
					    (return (Fitab x y))))
			    (fitres (std--vector<Fitab>))
			   
			    ))
		      (fitres.reserve repeat)
		      (std--generate_n (std--back_inserter fitres)
				       repeat
				       generate_fit)
		      ,@(loop for e in l-fit
			      collect
			      `(let ((,e (stat fitres (lambda (f)
							(declare (type "const  Fitab&" f))
							(return (dot f ,e))))))))
		     
					
		      (return (std--make_tuple ,@l-fit))))))
	 (dotimes (i 3)
	   (let #+nil
	     ((A .249999999999s0	;(+ 17 (* .1 (dis gen)))
		 )
	      (B 1.833333333333s0	;(+ .3 (* .01 (dis gen)))
		 )
	      (Sig 0s0 #+nil (+ .003 (* .001 (dis gen)))
		   ))
	     ((A (+ 17 (* .1 (dis gen)))
		 )
	      (B (+ .3 (* .01 (dis gen)))
		 )
	      (Sig 10s0 ;(+ .3 (* .001 (dis gen)))
		   )))
	   (let (((bracket ,@l-fit) (lin 8133 A B Sig 7117)))
	     (letc (,@(loop for e in l-fit collect `(,(format nil "p~a" e) (printStat ,e))))
	      ,(lprint :vars `(A B Sig ,@(loop for e in l-fit collect (format nil "p~a" e))))))))


       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
