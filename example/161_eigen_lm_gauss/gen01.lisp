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

(let ((l-par `(amplitude mean sigma)))
  (defparameter *source-dir* #P"example/161_eigen_lm_gauss/source01/src/")
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

     "#define EIGEN_VECTORIZE_AVX2"
     "#define EIGEN_VECTORIZE_AVX"
     "#define EIGEN_VECTORIZE"
     "#define EIGEN_VECTORIZE_FMA"
     (include<>
      iostream
      format
					;cstddef
      vector
      cmath
      random
      numeric
      algorithm
      ;execution
      Eigen/Core ;Dense
					;memory
      thread
      popl.hpp
      )
   
     "using namespace std;"
     "using namespace Eigen;"

     
					;(include "xsimd/xsimd.hpp")

					;"using namespace xsimd;"

     "using Scalar = float;"
					;"using ScalarI = const Scalar;"
     
					;"using XVec = vector<Scalar,xsimd::default_allocator<Scalar>>;"
					;"using XBatch = xsimd::batch<Scalar,avx2>;"


     (comments "dynamic rows, 1 column")
     "using Vec = Matrix<Scalar,Dynamic,1,0,8192,1>;"
     (comments "dynamic rows, 3 columns")
     "using Mat = Matrix<Scalar,Dynamic,3,0,8192,1>;"
					;"using Vec = Matrix<Scalar,8192,1,0,8192,1>;"
					
     "using VecI = const Vec;"
     "using MatI = const Mat;"
					; "constexpr auto Pol = std::execution::par_unseq;"

     (defun gaussian (x ,@l-par)
       (declare (type Scalar x ,@l-par)
		(values Scalar)
		)
       (let ((x0 (- x mean)))
	(return (* amplitude
		   (exp (/ (* x0 x0)
			   (* -2s0 sigma sigma)))))))
     
     (defclass+ GaussianModel ()
       "public:"
       (defmethod GaussianModel (xx yy)
	 (declare (type Vec xx yy)
		  (values :constructor)
		  (construct 
			     (x (std--move xx))
			     (y (std--move yy))
			     )))
       (defmethod "operator()" (parameters residuals)
	 (declare (type VecI& parameters)
		  (type Vec& residuals))
	 ,@(loop for e in l-par
		 and e-i from 0
		 collect
		 `(let ((,e (parameters ,e-i)))))
	 (letc ((gaussian_values
		 (* amplitude (dot (paren
				    (/ (dot (paren (- (dot x (array)) mean))
					    (array)
					    (square)
					    (array))
				       (* -2s0 sigma sigma)))
				   (array)
				   (exp))))
		)
	       (setf residuals (- (y.array)
				  (dot gaussian_values (array)))
				  ))
	 )

       (defmethod jacobian (parameters jac)
	 (declare (type VecI& parameters)
		  (type Mat& jac))
	 ,@(loop for e in l-par
		 and e-i from 0
		 collect
		 `(letc ((,e (parameters ,e-i)))))
	 (letc ((gaussian_values
		 (* amplitude (dot (paren (/ (dot (paren (- (dot x (array)) mean))
					    (array)
						  (square)
						  (array))
					     (* -2s0 sigma sigma)))
				   (array)
				   (exp))))
		))
	 (comments "Derivative with respect to amplitude")
	 (setf (jac.col 0) gaussian_values
	       )
	 (letc ((diff_x (- (dot x (array)) mean))
		(exp_term_matrix (dot (exp (/ (dot x (array)
						   (square)
						   (colwise))
					      (* -2s0 sigma sigma)))
				      (matrix)))
		(denominator (* sigma sigma)))
	       (comments "Derivative with respect to mean")
	       (setf (jac.col 1)
		     (* 
			(dot diff_x (array) (rowwise))
			(exp_term_matrix.rowwise)
			(/ amplitude denominator)))
	       (comments "Derivative with respect to sigma")
	       (setf (jac.col 2)
		     (* 
			(dot diff_x (array) (square) (rowwise))
			(exp_term_matrix.rowwise)
			(/ amplitude
			   (* denominator sigma sigma)))))
	 )
					;"private:"

       "Vec x, y;")

     

     (defun lm (model initial_guess lambda)
       (declare (type "const GaussianModel&" model)
		(type "VecI&" initial_guess)
		(type Scalar lambda)
		(values Vec&))
       (letc ((maxIter 100)
	      (tolerance 1e-4)))
       (let ((parameters initial_guess)
	     (residuals (Vec (model.x.size)))
	     (jacobian (Mat (model.x.size) 3)))
	 (dotimes (iter maxIter)
	   (model residuals
		  (model.jacobian parameters jacobian))
	   (let ((residual_norm (residuals.norm)))
	     (when (< residual_norm tolerance)
	       break)
	     (let ((jTj (+ (* (jacobian.transpose)
			      jacobian)
			   (* lambda (MatrixXd--Identity 3 3))))
		   (delta (* (-jacobian.transpose)
			     residuals))
		   (parameters_new (+ parameters delta))
		   (residual_norm_new (dot (model parameters_new
						  (model.jacobian parameters_new jacobian))
					   (norm))))
	       (if (< residual_norm_new residual_norm)
		   (do0
		    (setf parameters parameters_new)
		    (/= lambda 10s0))
		   (do0
		    (*= lambda 10s0))))))
	 (return parameters))

       )
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
      
       ,(let ((l `(
		   (:name numberPoints :default 1024 :short p)
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

       (let ((gen (mt19937		; 42	
		   "random_device{}()"))
	     (dis (normal_distribution<float> 0s0 1s0))))

       
       (let ((x (Vec (-2 -1.5 -1 -.5 0 .5 1 1.5 2)))
	     (y (Vec (.0674 .1358 .2865 .4933 1. .4933 .2865 .1358 .0674)))
	     (initial_guess (Vec 1s0 0s0 1s0))
	     (lamb .1s0)
	     (parameters (lm (GaussianModel x y)
			     initial_guess lamb))))

       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
