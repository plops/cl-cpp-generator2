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

     "constexpr int N = 8192;"

     (defun fun_valarray (a b)
       (declare (type AVecI& a b)
		(values Scalar))
       (return (dot (std--pow (* a b) 2)
		    (sum))))
     
     #+nil (defun fun_simd (a b)
       (declare (type VecI& a b)
		(values Scalar))
       (let ((inc Batch--size)
	     (size (a.size))
	     (vec_size (- size (% size inc))))
	 (comments "size for which vecotorization is possible")
	 #+nil (for ((= "std::size_t i" 0)
	       (< i vec_size)
	       (incf i inc))
	      (let ((avec (Batch--load_aligned (ref (aref a i))))
		    (bvec (Batch--load_aligned (ref (aref b i))))
		    (rvec (* avec bvec))
		    )
		;(rvec.store_aligned (ref (aref a i)))
		))
	 (return 0s0)
	 ))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (letc ((aa (AVec N))
	     (ab (AVec N)))
	 ,(lprint :vars `(
			  (fun_valarray aa ab))))
       #+nil (let ((a (Vec N))
	     (b (Vec N)))
	 ,(lprint :vars `((fun_simd  a b))))
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
