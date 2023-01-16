(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/102_simple_test/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (include
      ,@(loop for e in `(iostream)
	      collect
	      (format nil "<~a>" e)))

     (defun main (argc argv)
       (declare (type int argc)
		(type **char argv)
		(values int))
       
       (dotimes (i 23)
	 (declare (type char i))
	 (<< std--cout
	     (+ i (* a (+ a 1)) b)
	     std--endl))

       (for-range
	(th threads)
       	(declare (type "auto&" th))
	(th.join)
	;(<< std--cout (string "bla"))
	)
       ))))

