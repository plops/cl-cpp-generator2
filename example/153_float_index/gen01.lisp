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
  (defparameter *source-dir* #P"example/153_float_index/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (load "util.lisp")
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      iostream
      cstdint
      cstring
      )

     (defun to_float (n)
       (declare (values float)
		(type uint32_t n))
       (incf n (- (<< 1u 23) 1))
       (if (and n (paren (<< 1u 31)))
	   (setf n (^ n (paren (<< 1u 31))))
	   (setf n ~n))
       "float f;"
       (memcpy &f &n 4)
       (return f))

     (defun float_to_index (f)
       (declare (values uint32_t)
		(type float f))
       "uint32_t n;"
       (memcpy &n &f (sizeof n))

       (if (and n (paren (<< 1u 31)))
	   (setf n (^ n (paren (<< 1 31))))
	   (setf n ~n))
       (return (- n (- (<< 1u 23) 1)))
       )

     
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       
       ,(lprint :vars `((to_float 12)
			(float_to_index (to_float 12))))
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))
