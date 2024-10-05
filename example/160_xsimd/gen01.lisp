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
  (defparameter *source-dir* #P"example/160_xsimd/source01/src/")
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
      )

     (include "xsimd/xsimd.hpp")

     "using namespace xsimd;"

     "using Voc = std::vector<float,xsimd::default_allocator<float>>;"

     (defun mean (a b res)
       (declare (type "const Voc&" a b)
		(type Voc& res))
       "using Batch = xsimd::batch<float,avx2>;"
       (let ((inc Batch--size)
	     (size (res.size))
	     
	     (vec_size (- size (% size inc))))
	 (comments "size for which vecotorization is possible")
	 (for ((= "std::size_t i" 0)
	       (< i vec_size)
	       (incf i inc))
	      (let ((avec (Batch--load_aligned (ref (aref a i))))
		    (bvec (Batch--load_aligned (ref (aref b i))))
		    (rvec (* (+ avec bvec) .5))
		    )
		(rvec.store_aligned (ref (aref res i)))))
	 (comments "remaining part that can't be vectorized")
	 (for ((= "std::size_t i" vec_size)
	       (< i size)
	       (incf i))
	      (setf (aref res i)
		    (* (+ (aref a i)
			  (aref b i))
		       .5)))))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (let ((a (Voc (curly 1.5 2.5 3.5 4.5 1 2 3 4)))
	     (b (Voc (curly 2.5 3.5 4.5 5.5 2 3 4 5)))
	     (r (Voc (b.size)))
	     )
	 (mean a b r)
	 ,(lprint :vars `(,@(loop for i below 8 collect
						`(aref r ,i))))
	 )
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
