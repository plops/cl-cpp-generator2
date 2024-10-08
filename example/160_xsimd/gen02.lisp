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
      cstddef
      vector
      cmath
      random
      numeric
      algorithm
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
       ,(format nil "Scalar ~{~a~^, ~};" l-fit)
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
       (let ((precision  (getSignificantDigits d))
	     (fmtm (+ (std--string (string "{:."))
		      (std--to_string (+ 1 precision))
		      (string "f}")))
	     (fmtd (+ (std--string (string "{:."))
		      (std--to_string precision)
		      (string "f}")))
	     (format_str (+  fmtm (string " ± ") fmtd))
	     )
	 (declare (type "const std::string" format_str))
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
				    (std--iota (x.begin) (x.end) 0s0)))
			  (fill_y (lambda ()
				    (dotimes (i n)
	      			      (setf (aref y i) (+ (* Sig (dis gen))
       	       						  B
							  (* A (aref x i))))))))
		      (fill_x)

		      (let ((stat (lambda (fitres filter)
				    (let ((data (Vec (fitres.size)))))
				    (std--transform (fitres.begin)
						    (fitres.end)
						    (data.begin)
						    filter)
				    (let ((mean (std--accumulate (data.begin)
								 (data.end)
								 0s0)))
				     
				      (/= mean (data.size))
				      (let ((sq_sum (std--inner_product (data.begin)
									(data.end)
									(data.begin)
									0s0))
					    (stdev (std--sqrt (- (/ sq_sum
								    (data.size))
								 (* mean mean))))))
				      (return (std--make_pair mean stdev)))))))
		      (let ((generate_fit (lambda ()
					    (fill_y)
					    (return (Fitab x y))))
			    (fitres (std--vector<Fitab>))
			   
			    ))
		      (fitres.reserve repeat)
		      (std--generate_n (std--back_inserter fitres)
				       repeat
				       generate_fit)
		      ,@(loop for e in l-fit
			      collect
			      (let ((am (format nil "~am" e))
				    (ad (format nil "~ad" e))
				    (amd (format nil "~amd" e)))
				`(let ((,e (stat fitres (lambda (f)
									 (declare (type "const  Fitab&" f))
									 (return (dot f ,e)))))))))
		     
					
		      (return (std--make_tuple ,@l-fit))))))
	 (dotimes (i 30)
	   (let ((A (+ .3 (* .01 (dis gen))))
		 (B (+ 17 (* .1 (dis gen))))
		 (Sig (+ 3 (* .1 (dis gen))))))
	   (let (((bracket ,@l-fit) (lin 18 A B Sig 100)))
	     ,(lprint :vars `(A B Sig ,@(loop for e in l-fit collect `(printStat ,e)))))))


       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
