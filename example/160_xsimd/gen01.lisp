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
     "using Batch = xsimd::batch<float,avx2>;"
     (defun mean (a b res)
       (declare (type "const Voc&" a b)
		(type Voc& res))
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

     (defun transpose (a b)
       (declare (type "const Voc&" a)
		(type Voc& b))
       (let ((N (static_cast<std--size_t> (std--sqrt (a.size))))
	     (inc Batch--size))
	 (dotimes (i N)
	   (for ((= "std::size_t j" 0)
		 (< i N)
		 (incf i inc))
	     (let ((block (Batch--load_aligned (ref (aref a (+ (* i N)
							       j))))))
	       (dot block
		    (store_aligned 
		     (ref (aref b (+ (* j N)
				   i))))))
	     ))))

     #+nil
     (do0 
      (comments "certainly not working;")
      (defun matmul (a b res)
	(declare (type "const Voc&" a b)
		 (type Voc& res))
	"constexpr int N = 8;"
	
	(let ((m (/ (a.size) (* N N)))
	      (n (/ (b.size) (* N N)))
	      (k (/ (res.size) (* N N))))
	  (dotimes (i m)
	    (dotimes (j n)
	      (dotimes (l k)
		(let ((sum (Batch .0)))
		  (dotimes (aa N)
		    (let ((pos (+ (* i N N)
				  (* aa N)))
			  (ablock (Batch--load_aligned (ref (aref a pos))))
			  (bblock (Batch--load_aligned (ref (aref b pos)))))
		      (incf (aref sum aa) (reduce_add (* ablock bblock)))))
		  (sum.store_aligned (ref (aref res (+ (* i N n)
						       (* j N))))
				     ))))))))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))

       ,(let ((l-a `(1.5 2.5 3.5 4.5 1 2 3 4))
	      (l-b `(2.5 3.5 4.5 5.5 2 3 4 5)))
	 `(let ((a (Voc (curly ,@l-a)))
	       (b (Voc (curly ,@l-b)))
	       (e (Voc (curly ,@(loop for a in l-a and b in l-b collect (* .5 (+ a b))))))
	       (r (Voc (b.size)))
	       )
	   (mean a b r)
	   
	   ,(lprint :vars `(,@(loop for i below 8 collect
				    `(aref r ,i))
			    ,@(loop for i below 8 collect
				    `(-
				      (aref r ,i)
				      (aref e ,i)))))
	   ))

       (progn
	,(let ((l-a (loop for i below (* 8 8)
			  appending
			  (loop for j below (* 8 8)
				collect
				(+ (* (* 8 8) i)
				   j))))
	       (l-e (loop for j below (* 8 8)
			  appending
			  (loop for i below (* 8 8)
				collect
				(+ (* (* 8 8) i)
				   j)))))
	   `(let ((a (Voc (curly ,@l-a)))
		
		  (e (Voc (curly ,@l-e)))
		  (r (Voc (a.size)))
		  )
	      (transpose a r)
	   
	      ,(lprint :vars `(,@(loop for i below 8 collect
				       `(aref r ,i))
			       ,@(loop for i below 8 collect
				       `(-
					 (aref r ,i)
					 (aref e ,i)))))
	      )))
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
