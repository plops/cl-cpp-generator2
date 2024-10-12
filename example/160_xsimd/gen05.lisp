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
  (defparameter *source-dir* #P"example/160_xsimd/source05/src/")
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
  (defun begin (arr)
    `(ref (aref ,arr 0)) )
  (defun end (arr)
    `(+ (ref (aref ,arr 0)) (dot ,arr (size))))
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
     (include "xsimd/xsimd.hpp")
   
     "using namespace std;"


     "using namespace xsimd;"
     "using Scalar = float;"


          
     "using Vec = std::vector<Scalar,xsimd::default_allocator<Scalar>>;"
     "using VecI = const Vec;"
     "using Batch = xsimd::batch<Scalar,avx2>;"


     



     "constexpr auto Pol = std::execution::par_unseq;"


     (comments "= sum(arr) ")
     (let ((sum (lambda (arr)
		  ;; = sum(arr)
		  (declare (capture "")
			   (type VecI& arr)
			   (values Scalar))
		  (letc ((inc Batch--size)
			 (size (arr.size))
			 (vec_size (- size
				      (% size inc)))))
		  (let ((sum 0s0))
		    (dotimes (i vec_size inc)
		      (incf sum (reduce_add (Batch--load_aligned (ref (aref arr i)))))))
		  (return sum)))))
     
     (do0
      (comments "res = arr - s .. array subtract")
      (let ( (asub (lambda (res arr s)
		     ;; res = arr - s
		     (declare (capture "")
			      (type VecI& arr)
			      (type Vec& res)
			      (type Scalar s)
			      )
		     (letc ((inc Batch--size)
			    (size (arr.size))
			    (vec_size (- size
					 (% size inc)))))
		     (dotimes (i vec_size inc)
		       (letc ((a (- (Batch--load_aligned (ref (aref arr i)))
				    s)))
			     (a.store_aligned
			      (ref (aref res i))))))))))

     (do0
      (comments "= sum(| arr - s |) .. subtract scalar, compute absolute value, sum")
      (let ( (subabssum (lambda (arr s)
		    
			  (declare (capture "")
				   (type VecI& arr)
				   (type Scalar s)
				   (values Scalar)
				   )
			  (letc ((inc Batch--size)
				 (size (arr.size))
				 (vec_size (- size
					      (% size inc)))))
			  (let ((sum 0s0)))
			  (dotimes (i vec_size inc)
			    (incf sum
				  (reduce_add
				   (abs (- (Batch--load_aligned (ref (aref arr i)))
					   s)))))
			  (return sum))))))
     (do0 (comments "= sum(arr**2)")
	  (let ( (squaredsum (lambda (arr)
			       ;; = sum(arr**2)
			       (declare (capture "")
					(type VecI& arr)
					(values Scalar))
			       (letc ((inc Batch--size)
				      (size (arr.size))
				      (vec_size (- size
						   (% size inc)))))
			       (let ((sum 0s0))
				 (dotimes (i vec_size inc)
				   (incf sum (reduce_add (pow
							  (Batch--load_aligned
							   (ref (aref arr i)))
							  2)))))
			       (return sum))))))
     (do0
      (comments "= sum( tt*y / st2 ) ")
      (let (  (compute_b (lambda (tt y st2)
			   ;; = sum( tt*y / st2 ) 
			   (declare (capture "")
				    (type VecI& tt y)
				    (type Scalar st2)
				    (values Scalar))
			   (letc ((inc Batch--size)
				  (size (tt.size))
				  (vec_size (- size
					       (% size inc)))))
			   (let ((sum 0s0))
			     (dotimes (i vec_size inc)
			       (let ((att (Batch--load_aligned
					   (ref (aref tt i))))
				     (ay (Batch--load_aligned
					  (ref (aref y i)))))
				 (incf sum (reduce_add (/ (* att ay) st2))))))
			   (return sum))))))

     (do0
      (comments "= sum((y-a-b*x)**2)")
      (let (  (compute_chi2 (lambda (y a b x)
			      (declare (capture "")
				       (type VecI& x y)
				       (type Scalar a b)
				       (values Scalar))
			      (letc ((inc Batch--size)
				     (size (y.size))
				     (vec_size (- size
						  (% size inc)))))
			      (let ((sum 0s0))
				(dotimes (i vec_size inc)
				  (let ((ax (Batch--load_aligned
					     (ref (aref x i))))
					(ay (Batch--load_aligned
					     (ref (aref y i)))))
				    (incf sum (reduce_add (pow (- ay a (* ax b)) 2))))))
			      (return sum))))))
     
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
	 (letc ((inc Batch--size)
		(size (x.size))
		(vec_size (- size
			     (% size inc)))))
	 (letc ((sx (sum x))
		(sy (sum y))))
	 
	 (letc ((ss (static_cast<Scalar> ndata))
		(sxoss (/ sx ss)))
	       )
	 
	 (letc ((tt ((lambda ()
		       (let ((res (Vec size)
				  
					;(- x sxoss)
				  ))
			 (asub res x sxoss))
		       (return res))))
		(st2 (squaredsum tt))
		)
	       )
	 (setf b (compute_b tt y st2))
	 (setf a (/ (- sy (* b sx))
		    ss))
	 (setf siga (sqrt (/ (+ 1s0 (/ (* sx sx)
				       (* ss st2)) )
			     ss))
	       sigb (sqrt (/ 1s0 st2)))
	 (setf chi2 (compute_chi2 y a b x) )
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

	      (fmtm (+ (std--string (string "{:."))
		       (to_string mprecision)
		       (string "f}")))
	      (fmtd (+ (std--string (string "{:."))
		       (to_string dprecision)
		       (string "f}")))
	      (fmtr (+ (std--string (string " ({:."))
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
	 (nth_element			;(execution--par_unseq 6)
	  (ref (aref arr 0 ))
	  (+ (ref (aref arr 0)) k)
	  (+ (ref (aref arr 0)) (arr.size))
	  )
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
		    (comments "number points must be divisible by 8 (avx2 batch size)")
		    (assert (== (% n Batch--size) 0) )
		    (let (		;(n 8)
			  (x (Vec n))
			  (y (Vec n))
			  (fill_x (lambda ()
				    (iota ,(begin 'x) ,(end 'x) 1s0)))
			  #+nil (fill_y
				  (lambda ()
				    (dotimes (i n)
	      			      (setf (aref y i) (+ (* Sig (dis gen))
       	       						  A
							  (* B (aref x i))))))))
		      (fill_x)
		      
		      (let ((stat_median (lambda (fitres filter)
					   (declare (type "const auto&" fitres)
						    (values "tuple<Scalar,Scalar,Scalar,Scalar>"))
					   (comments "compute median and median absolute deviation Numerical recipes 8.5 and 14.1.4")
					   (let ((data (Vec (fitres.size)))))
					   (data.resize (fitres.size))
					   (transform (fitres.begin)
						      (fitres.end)
						      ,(begin 'data)
						      filter)
					   (letc ((N (static_cast<Scalar> (data.size)))
						  (median (select (/ (- (static_cast<int> (data.size)) 1) 2)
								  data))
						  (adev 
						   (/ (subabssum data median)
						    
						      N))
						  ;; error in the mean due to sampling
						  (mean_stdev (/ adev (sqrt N)))
						  ;; error in the standard deviation due to sampling
						  (stdev_stdev (/ adev (sqrt (* 2 N)))))
					  
						 (return (make_tuple median mean_stdev adev stdev_stdev)))
					   ))
			    (stat_mean (lambda (fitres filter) 
					 (declare (type "const auto&" fitres)
						  (values "tuple<Scalar,Scalar,Scalar,Scalar>"))
					 (comments "compute mean and standard deviation Numerical Recipes 14.1.2 and 14.1.8")
					 (let ((data (Vec (fitres.size)))))
					 (data.resize (fitres.size))
					 (transform (fitres.begin)
						    (fitres.end)
						    ,(begin 'data)
						    filter)
					 (do0
					  (comments " = sum( (data-s) ** 2 )")
					  (let ((var_pass1 (lambda (arr s)
		    
							     (declare (capture "")
								      (type VecI& arr)
								      
								      (type Scalar s)
								      (values Scalar))
							     (letc ((inc Batch--size)
								    (size (arr.size))
								    (vec_size (- size
										 (% size inc)))))
							     (let ((sum 0s0)))
							     (dotimes (i vec_size inc)
							       (incf sum
								     (reduce_add
								      (pow (- (Batch--load_aligned (ref (aref arr i)))
									      s)
									   2))))
							     (return sum))))))
					 (do0
					  (comments " = sum( data-s )")
					  (let ((var_pass2 (lambda (arr s)
		    
							     (declare (capture "")
								      (type VecI& arr)
								      
								      (type Scalar s)
								      (values Scalar))
							     (letc ((inc Batch--size)
								    (size (arr.size))
								    (vec_size (- size
										 (% size inc)))))
							     (let ((sum 0s0)))
							     (dotimes (i vec_size inc)
							       (incf sum
								     (reduce_add
								      (- (Batch--load_aligned (ref (aref arr i)))
									 s)
								      )))
							     (return sum))))))
					 (letc ((N (static_cast<Scalar> (data.size)))
						(mean (/ (sum data)
							 N))
					
						;; 14.1.8 corrected two-pass algorithm from bevington 2002
						
						(stdev
						 (sqrt (/ (- (var_pass1 data mean)
							     (/ (pow
								 (var_pass2 data mean)
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
					    (transform ,(begin 'x)
						       ,(end 'x)
						       ,(begin 'y)					    
						       (lambda (xi)
							 (declare (type Scalar xi))
							 (return (+ (* Sig (dis gen))
       	       							    A
								    (* B xi))))
						       )
					    (return (Fitab x y))))
			    
			    ))
		      (let ((fitres (vector<Fitab>))))
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
	      
		  (Sig generatorSigma	;(+ .3 (* .001 (dis gen)))
		       )))
	   (let (((bracket ,@l-fit) (lin numberPoints A B Sig numberRepeats)))
	     (letc (,@(loop for e in l-fit collect `(,(format nil "p~a" e) (printStat ,e))))
		   ,(lprint :vars `(A dA B dB Sig ,@(loop for e in l-fit collect (format nil "p~a" e))))))))


       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
