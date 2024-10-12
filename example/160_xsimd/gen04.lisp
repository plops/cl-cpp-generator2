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
  (defparameter *source-dir* #P"example/160_xsimd/source04/src/")
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
     (include<>
      
      iostream
      format
      cstddef
      vector
      chrono
      cmath
      valarray
      )

     (include "xsimd/xsimd.hpp")

     "using namespace std;"
     "using namespace std::chrono;"
					; "using namespace std::chrono::high_resolution_clock;"
     "using namespace xsimd;"
     "using Scalar = float;"


     
     "using AVec = valarray<Scalar>;"
     "using AVecI = const AVec;"
     
     "using Vec = std::vector<Scalar,xsimd::default_allocator<Scalar>>;"
     "using VecI = const Vec;"
     "using Batch = xsimd::batch<Scalar,avx2>;"

     "using Timebase = std::chrono::milliseconds;"
     
     "constexpr int N = 8*1024;"

     (defun fun_valarray (a b c)
       (declare (type AVecI& a b)
		(type Scalar c)
		(values Scalar))
       (return (dot (std--pow (* a (+ b c)) 2)
		    (sum))))
     
     (defun fun_simd (a b c)
       (declare (type VecI& a b)
		(type Scalar c)
		(values Scalar))
       (let ((inc Batch--size)
	     (size (a.size))
	     (sum (Scalar 0s0))
	     (vec_size (- size (% size inc))))
	 (comments "size for which vecotorization is possible")
	 (for ((= "std::size_t i" 0)
	       (< i vec_size)
	       (incf i inc))
	      (let ((avec (Batch--load_aligned (ref (aref a i))))
		    (bvec (Batch--load_aligned (ref (aref b i))))
		    (rvec (pow (* avec (+ c bvec))
			       2))
		    )
		(incf sum (reduce_add rvec))
		))
	 (return sum)
	 ))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       ,@(loop for (e f) in `((valarray AVec) (simd Vec))
	       collect
	       (let ((fun (format nil "fun_~a" e)))
		 `(progn
		    (let ((a (,f N))
			  (b (,f N)))
		      (dotimes (i N)
			(setf (aref a i) (sin (* .1s0 i)))
			(setf (aref b i) (+ (sin (* .01s0 i)) 2s0)))
		      (letc ((start  ("high_resolution_clock::now")
				     ))
			    (let ((res 0s0))
			      (dotimes (i "100'000")
				(let ((c (* 1s-4 i)))
				  (incf res (,fun a b c)))))
			    (letc ((end ("high_resolution_clock::now"))
				   (duration (duration_cast<Timebase> (- end start))))))
		      ,(lprint :vars `((,fun  a b .1s0)
				       res
				       duration))
		      ))))
      
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
