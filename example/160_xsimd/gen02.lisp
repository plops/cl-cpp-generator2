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
     (defun printStat (m_md_d_dd ;m d
		       )
       (declare ;(type Scalar m d)
	(type "std::tuple<Scalar,Scalar,Scalar,Scalar>" m_md_d_dd)
	(values "std::string"))
       (let (((bracket m md d dd) m_md_d_dd)))
       (letc ((mprecision  (getSignificantDigits md))
	      (dprecision  (getSignificantDigits dd))
	      (fmtm (+ (std--string (string "{:."))
		       (std--to_string mprecision)
		       (string "f}")))
	      (fmtd (+ (std--string (string "{:."))
		      (std--to_string dprecision)
		      (string "f}")))
	     (format_str (+  fmtm (string "±") fmtd))
	     )
	 (return (std--vformat format_str (std--make_format_args m d)))
	 ))
     (defun select (k arr)
       (declare (type "const int" k)
		(type "Vec&" arr)
		(values Scalar))
       (comments "Numerical Recipes 8.5: select a random partitioning element `a`  and iterate through array."
		 "move smaller elements to the left and larger to the right. (this is like quicksort)"
		 "sentinels at either end of the subarray reduce work in the inner loop. leftmost sentienel is <= a, rightmost sentinel is>=a")
       (letc ((n (static_cast<int> (arr.size))))
	     )
       "Scalar a;"
       (let ((ir (- n 1))
	     (l 0)
	     (i 0)
	     (j 0)
	     (mid 0)
	     )
	 (while true
		(if (<= ir (+ l 1))
		    (do0
		     (comments "Active partition contains 1 or 2 elements")
		     (when (logand (== ir (+ l 1))
				   (< (aref arr ir)
				      (aref arr l)))
		       (comments "Case of two elements")
		       (std--swap (aref arr l)
				  (aref arr ir)))
		     (return (aref arr k)))
		    (do0
		     (comments "Choose median of left, center and right elements as partitioning element a"
			       "Also rearrange so that arr[l] <= arr[l+1], arr[ir]>=arr[l+1]")
		     (setf mid (>> (paren (+ l ir))
				   1))
		     (std--swap (aref arr mid)
				(aref arr (+ l i)))
		     ,@(loop for (e f) in `((ir l) (ir (+ l 1)) ((+ l 1) l))
			     collect
			     `(when (< (aref arr ,e)
				       (aref arr ,f))
				(std--swap (aref arr ,e)
					   (aref arr ,f))))
		     (comments "Initialize pointers for partitioning")
		     (setf i (+ l 1)
			   j ir
			   a (aref arr (+ l 1)))
		     (comments "Inner loop")
		     (while true
			    (comments "Scan up to find element > a")
			    (space do (progn (incf i) )
				   while (paren (< (aref arr i) a)))
			    (comments "Scan down to find element < a")
			    (space do (progn (decf j) )
				   while (paren (< a (aref arr j))))
			    (when (< j i)
			      (comments "Pointers crossed. Partitioning complete")
			      break)
			    (comments "Insert partitioning element")
			    (std--swap (aref arr i)
				       (aref arr j)))
		     (comments "Insert partitioning element")
		     (setf (aref arr (+ l 1))
			   (aref arr j))
		     (setf (aref arr j) a)
		     (comments "Keep active the partition that contains the kth element")
		     (when (<= k j)
		       (setf ir (- j 1)))
		     (when (<= j k)
		       (setf l i))
		     )))))
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
		      
		      (let ((stat_median (lambda (fitres filter)
					   (comments "compute median and median absolute deviation Numerical recipes 8.5 and 14.1.4")
					   (let ((data (Vec (fitres.size)))))
					   (data.resize (fitres.size))
					   (std--transform (fitres.begin)
							   (fitres.end)
							   (data.begin)
							   filter)
					   (letc ((N (static_cast<Scalar> (data.size)))
						  (median (select (/ (- (static_cast<int> (data.size)) 1) 2)
								  data))
						  (adev
						   (/ (std--accumulate
						       (data.begin)
						       (data.end)
						       0s0
						       (lambda (acc xi)
							 (declare (capture "median"))
							 (return (+ acc (std--abs (- xi median))))))
						      N))
						  ;; error in the mean due to sampling
						  (mean_stdev (/ adev (std--sqrt N)))
						  ;; error in the standard deviation due to sampling
						  (stdev_stdev (/ adev (std--sqrt (* 2 N)))))
					  
						 (return (std--make_tuple median mean_stdev adev stdev_stdev)))
					   ))
			    #+nil(stat_mean (lambda (fitres filter)
				    (comments "compute mean and standard deviation Numerical Recipes 14.1.2 and 14.1.8")
				    (let ((data (Vec (fitres.size)))))
				    (data.resize (fitres.size))
				    (std--transform (fitres.begin)
						    (fitres.end)
						    (data.begin)
						    filter)
				    (letc ((N (static_cast<Scalar> (data.size)))
					   (mean (/ (std--accumulate (data.begin)
								     (data.end)
								     0s0)
						    N))
					
					   ;; 14.1.8 corrected two-pass algorithm from bevington 2002
					   (stdev
					    (std--sqrt (/ (- (std--accumulate
							      (data.begin)
							      (data.end)
							      0s0
							      (lambda (acc xi)
								(declare (capture "mean"))
								(return (+ acc (std--pow (- xi mean) 2s0)))))
							     (/ (std--pow
								 (std--accumulate
								  (data.begin)
								  (data.end)
								  0s0
								  (lambda (acc xi)
								    (declare (capture "mean"))
								    (return (+ acc (- xi mean)))))
								 2s0)
								N))
							  (- N 1s0))))
					   ;; error in the mean due to sampling
					   (mean_stdev (/ stdev (std--sqrt N)))
					   ;; error in the standard deviation due to sampling
					   (stdev_stdev (/ stdev (std--sqrt (* 2 N)))))
					  
					  (return (std--make_tuple mean mean_stdev stdev stdev_stdev)))))))
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
			      `(let ((,e (stat_median fitres (lambda (f)
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
	      (Sig 10s0			;(+ .3 (* .001 (dis gen)))
		   )))
	   (let (((bracket ,@l-fit) (lin 133 A B Sig 17)))
	     (letc (,@(loop for e in l-fit collect `(,(format nil "p~a" e) (printStat ,e))))
		   ,(lprint :vars `(A B Sig ,@(loop for e in l-fit collect (format nil "p~a" e))))))))


       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
