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
      cmath
      valarray
      )

     (include "xsimd/xsimd.hpp")

     "using namespace std;"
     "using namespace xsimd;"
     "using Scalar = float;"


     
     "using AVec = valarray<Scalar>;"
      "using AVecI = const AVec;"
     
     "using Vec = std::vector<Scalar,xsimd::default_allocator<Scalar>>;"
     "using VecI = const Vec;"
     "using Batch = xsimd::batch<Scalar,avx2>;"

     "constexpr int N = 8*1024;"

     (defun fun_valarray (a b)
       (declare (type AVecI& a b)
		(values Scalar))
       (return (dot (std--pow (* a b) 2)
		    (sum))))
     
     (defun fun_simd (a b)
       (declare (type VecI& a b)
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
		    (rvec (pow (* avec bvec)
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
       (let ((aa (AVec N))
	      (ab (AVec N)))
	     (dotimes (i N)
	       (setf (aref aa i) (sin (* .1s0 i))
		     )
	       (setf (aref ab i) 2s0 ;(+ (cos (* .3s0 i)) (/ 1s0 i))
		     ))
	     ,(lprint :vars `(
			      (fun_valarray aa ab))))
       (let ((a (Vec N))
	     (b (Vec N)))
	 (dotimes (i N)
	       (setf (aref a i) (sin (* .1s0 i)))
	       (setf (aref b i) 2s0))
	 
	 ,(lprint :vars `((fun_simd  a b))))
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
