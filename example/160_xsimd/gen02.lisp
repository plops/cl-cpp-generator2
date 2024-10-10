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

(let ((l-fit `(a siga b sigb chi2 sigdat)))
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
	  (string ,(format nil "(~a~{:~a '{}'~^ ~})\\n"
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
      execution
      valarray
					;memory
      thread
      popl.hpp
      )
   
     "using namespace std;"
     "using namespace std::execution;"

     
					;(include "xsimd/xsimd.hpp")

					;"using namespace xsimd;"

     "using Scalar = float;"
					;"using ScalarI = const Scalar;"
     
					;"using XVec = vector<Scalar,xsimd::default_allocator<Scalar>>;"
					;"using XBatch = xsimd::batch<Scalar,avx2>;"


     "using Vec = vector<Scalar>;"
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
	 (letc ((sx (accumulate	;execution--par
		     (x.begin)
		     (x.end)
		     0s0)
		    )))
	 (letc ((sy (accumulate
					;execution--par
		     (y.begin)
		     (y.end)
		     0s0))))
	 (letc ((ss (static_cast<Scalar> ndata))
		(sxoss (/ sx ss)))
	       )
	 
	 (letc ((st2 (accumulate
					; execution--par
		      (x.begin)
		      (x.end)
		      0s0
		      (lambda (accum xi)
			(return (+ accum (pow (- xi sxoss) 2s0))))) ; .0f
		     )
		)
	       "#pragma omp parallel for reduction(+:b)"
	       (dotimes (i ndata)
		   
		 (letc ((tt (- (aref x i)
			       sxoss))))
		  
		 (incf b (* tt (aref y i)))
					;,(lprint :vars `(i tt b))
		 ))
	 #+nil  (setf b (inner_product (x.begin)
					    (x.end)
					    (y.begin)
					    0s0
					    (lambda (accum value) (return (+ accum value)))
					    (lambda (xi yi) (return (* tt yi)))))
	 (comments "solve for a, b, sigma_a and sigma_b")
	 (/= b st2)
	 (setf a (/ (- sy (* b sx))
		    ss))
	 (setf siga (sqrt (/ (+ 1s0 (/ (* sx sx)
					    (* ss st2)) )
				  ss))
	       sigb (sqrt (/ 1s0 st2)))
	 (comments "compute chi2")
	 (setf chi2 (inner_product
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
	 #+nil (setf chi2 (accumulate (x.begin)
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
	   (setf sigdat (sqrt (/ chi2
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
     (defun printStat (m_md_d_dd	;m d
		       )
       (declare				;(type Scalar m d)
	(type "tuple<Scalar,Scalar,Scalar,Scalar>" m_md_d_dd)
	(values "string"))
       (let (((bracket m md d dd) m_md_d_dd)))
       (letc ((rel (* 100s0 (/ d m)))
	      (mprecision  (getSignificantDigits md))
	      (dprecision  (getSignificantDigits dd))
	      (rprecision  (getSignificantDigits rel))

	      (fmtm (+ (string (string "{:."))
		       (to_string mprecision)
		       (string "f}")))
	      (fmtd (+ (string (string "{:."))
		       (to_string dprecision)
		       (string "f}")))
	      (fmtr (+ (string (string " ({:."))
		       (to_string rprecision)
		       (string "f}%)")))
	      (format_str (+  fmtm (string "±") fmtd fmtr))
	      
	      )
	     (return (vformat format_str (make_format_args m d rel)))
	     ))

     (defun select (k arr)
       (declare (type "const int" k)
		(type "Vec&" arr)
		(values Scalar))
       (comments "This implementation uses the STL and will not fall under the strict license of Numerical Recipes")
       (when (logand (<= 0 k)
		     (< k (arr.size)))
	 (nth_element		;(execution--par_unseq 6)
	  (arr.begin) (+ (arr.begin) k) (arr.end))
	 (return (aref arr k)))
       (throw (out_of_range (string "Invalid index for selection"))))
     #+nil
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
		       (swap (aref arr l)
				  (aref arr ir)))
		     (return (aref arr k)))
		    (do0
		     (comments "Choose median of left, center and right elements as partitioning element a"
			       "Also rearrange so that arr[l] <= arr[l+1], arr[ir]>=arr[l+1]")
		     (setf mid (>> (paren (+ l ir))
				   1))
		     (swap (aref arr mid)
				(aref arr (+ l 1)))
		     ,@(loop for (e f) in `((ir l) (ir (+ l 1)) ((+ l 1) l))
			     collect
			     `(when (< (aref arr ,e)
				       (aref arr ,f))
				(swap (aref arr ,e)
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
			    (swap (aref arr i)
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
      
       ,(let ((l `(
		   (:name numberRepeats :default 64 :short r)
		   (:name numberPoints :default 1024 :short p)
		   (:name numberTrials :default 3 :short d)
		   (:name generatorSlope :default .3s0 :short B :type Scalar)
		   (:name generatorDeltaSlope :default .01s0 :short b :type Scalar)
		   (:name generatorIntercept :default 17s0 :short A :type Scalar)
		   (:name generatorDeltaIntercept :default .1s0 :short a :type Scalar)
		   (:name generatorSigma :default 10s0 :short s :type Scalar
			  )
		   
		   )))
	  `(let ((op (popl--OptionParser (string "allowed options")))
		 ,@(loop for e in l collect
				    (destructuring-bind (&key name default short (type 'int)) e
				      `(,name (,type ,default))))
		 ,@(loop for e in `((:long help :short h :type Switch :msg "produce help message")
				    (:long verbose :short v :type Switch :msg "produce verbose output")
				    (:long mean :short m :type Switch :msg "Print mean and standard deviation statistics, otherwise print median and mean absolute deviation from it")
				    ,@(loop for f in l
					    collect
					    (destructuring-bind (&key name default short (type 'int)) f
					      `(:long ,name
						:short ,short
						:type ,type :msg "parameter"
						:default ,default :out ,(format nil "&~a" name))))

				    )
			 appending
			 (destructuring-bind (&key long short type msg default out) e
			   `((,(format nil "~aOption" long)
			      ,(let ((cmd `(,(format nil "add<~a>"
						     (if (eq type 'Switch)
							 "popl::Switch"
							 (format nil "popl::Value<~a>" type)))
					    (string ,short)
					    (string ,long)
					    (string ,msg))))
				 (when default
				   (setf cmd (append cmd `(,default)))
				   )
				 (when out
				   (setf cmd (append cmd `(,out)))
				   )
				 `(dot op ,cmd)
				 ))))
			 ))
	     (op.parse argc argv)
	     (when (helpOption->is_set)
	       (<< cout
		   op
		   endl)
	       (exit 0))

	     ))

       ,(lprint :vars `((thread--hardware_concurrency)))

       (let ((gen (mt19937		;42
		   "random_device{}()"))
	     (dis (normal_distribution<float> 0s0 1s0))))

       (let ((lin (lambda (n A B Sig repeat)
		    (let (		;(n 8)
			  (x (Vec n))
			  (y (Vec n))
			  (fill_x (lambda ()
				    (iota (x.begin) (x.end) 1s0)))
			  #+nil (fill_y
				  (lambda ()
				    (dotimes (i n)
	      			      (setf (aref y i) (+ (* Sig (dis gen))
       	       						  A
							  (* B (aref x i))))))))
		      (fill_x)
		      
		      (let ((stat_median (lambda (fitres filter)
					   (declare (type "const auto&" fitres))
					   (comments "compute median and median absolute deviation Numerical recipes 8.5 and 14.1.4")
					   (let ((data (Vec (fitres.size)))))
					   (data.resize (fitres.size))
					   (transform (fitres.begin)
							   (fitres.end)
							   (data.begin)
							   filter)
					   (letc ((N (static_cast<Scalar> (data.size)))
						  (median (select (/ (- (static_cast<int> (data.size)) 1) 2)
								  data))
						  (adev
						   (/ (accumulate
						       (data.begin)
						       (data.end)
						       0s0
						       (lambda (acc xi)
							 (declare (capture "median"))
							 (return (+ acc (abs (- xi median))))))
						      N))
						  ;; error in the mean due to sampling
						  (mean_stdev (/ adev (sqrt N)))
						  ;; error in the standard deviation due to sampling
						  (stdev_stdev (/ adev (sqrt (* 2 N)))))
					  
						 (return (make_tuple median mean_stdev adev stdev_stdev)))
					   ))
			    (stat_mean (lambda (fitres filter)
					(declare (type "const auto&" fitres))
					 (comments "compute mean and standard deviation Numerical Recipes 14.1.2 and 14.1.8")
					      (let ((data (Vec (fitres.size)))))
					      (data.resize (fitres.size))
					      (transform (fitres.begin)
							      (fitres.end)
							      (data.begin)
							      filter)
					      (letc ((N (static_cast<Scalar> (data.size)))
						     (mean (/ (accumulate (data.begin)
									       (data.end)
									       0s0)
							      N))
					
						     ;; 14.1.8 corrected two-pass algorithm from bevington 2002
						     (stdev
						      (sqrt (/ (- (accumulate
									(data.begin)
									(data.end)
									0s0
									(lambda (acc xi)
									  (declare (capture "mean"))
									  (return (+ acc (pow (- xi mean) 2s0)))))
								       (/ (pow
									   (accumulate
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
						     (mean_stdev (/ stdev (sqrt N)))
						     ;; error in the standard deviation due to sampling
						     (stdev_stdev (/ stdev (sqrt (* 2 N)))))
					  
						    (return (make_tuple mean mean_stdev stdev stdev_stdev)))))
			    (stat (lambda (fitres filter)
				    (declare (type "const auto&" fitres))
				    (if (meanOption->is_set)
					(return (stat_mean fitres filter))
					(return (stat_median fitres filter)))))))
		      (let ((generate_fit (lambda ()
					;(setf y (curly 2.1s0 2.3s0 2.6s0))
					    
					    (transform (x.begin)
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
			    (fitres (vector<Fitab>))
			   
			    ))
		      (fitres.reserve repeat)
		      (generate_n (back_inserter fitres)
				       repeat
				       generate_fit)
		      ,@(loop for e in l-fit
			      collect
			      `(letc ((,e (stat fitres (lambda (f)
							(declare (type "const  Fitab&" f))
							(return (dot f ,e))))))))
		     
					
		      (return (make_tuple ,@l-fit))))))
	 (dotimes (i numberTrials)
	   (letc #+nil
		 ((A .249999999999s0	;(+ 17 (* .1 (dis gen)))
		     )
		  (B 1.833333333333s0	;(+ .3 (* .01 (dis gen)))
		     )
		  (Sig 0s0 #+nil (+ .003 (* .001 (dis gen)))
			   ))
		 ((dA generatorDeltaIntercept)
		  (A (+ generatorIntercept (* dA (dis gen)))
		     )
		  (dB generatorDeltaSlope)
		  (B (+ generatorSlope (* dB (dis gen)))
		     )
	      
		  (Sig generatorSigma		;(+ .3 (* .001 (dis gen)))
		       )))
	   (let (((bracket ,@l-fit) (lin numberPoints A B Sig numberRepeats)))
	     (letc (,@(loop for e in l-fit collect `(,(format nil "p~a" e) (printStat ,e))))
		   ,(lprint :vars `(A dA B dB Sig ,@(loop for e in l-fit collect (format nil "p~a" e))))))))


       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
