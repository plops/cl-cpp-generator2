(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/115_microbench/source00/")
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
     
     (include<> chrono
		iostream
		array
		cmath)
     ,(let* ((n (*
		 (expt 2 (- 10 0))
		 (expt 2 (- 10 0))))
	     (nmax (*
		    (expt 2 (- 10 0))
		    (expt 2 (- 10 0))))
	     (iter (floor
		    (* nmax 10000)
		    n)))
	`(do0
	  (space constexpr int64_t (setf N ,n))
	  (space constexpr int64_t (setf ITER ,iter))))
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
      
       (let ((array ("std::array<int,N>")))
	 (dotimes (i N)
	   (setf (aref array i) i))
	 (let ((sum (int32_t 0))
	       (start (std--chrono--high_resolution_clock--now)))
	   (dotimes (j ITER)
	     (dotimes (i N)
	       (incf sum (aref array i))))
	   (let ((end (std--chrono--high_resolution_clock--now))
		 (elapsed (- end start)))
	     (<< std--cout
		 (string "time per iteration and element: ")
		 (/ (elapsed.count)
		    (static_cast<double> (* N ITER)))
		 (string " ns")
		 std--endl
		 )
	     (<< std--cout
		 (string "sum: ")
		 sum
		 (string " N: ")
		 N
		 std--endl
		 ))
	   ))
       (return 0))))
  )


  
