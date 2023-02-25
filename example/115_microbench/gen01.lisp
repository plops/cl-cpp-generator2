(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/115_microbench/source01/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
    (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

  (load "util.lisp")
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     
     (include<> chrono
		iostream
		array
		immintrin.h
		fmt/core.h)
     "#define ARRAY_SIZE 1000000"

     ,(let ((l `(cycles ; branches mispredicts
		 )))
	`(defun main (argc argv)
	  (declare (type int argc)
		   (type char** argv)
		   (values int))
	   ,(lprint :msg
		    (multiple-value-bind
			  (second minute hour date month year day-of-week dst-p tz)
			(get-decoded-time)
   (declare (ignorable dst-p))
		      (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
			      hour
			      minute
			      second
			      (nth day-of-week *day-names*)
			      year
			      month
			      date
			      (- tz)))
		    )
	   (std--srand (std--time nullptr))
	   (let ((array ("std::array<int,ARRAY_SIZE>")))
	    (foreach (e array)
		     (setf e (std--rand))))
	   (comments "if rdpmc crashes, run this: echo 2 | sudo tee /sys/devices/cpu/rdpmc ")
	   (let (,@ (loop for e in l
			  and e-i from 0
			  collect
			  `(,e (__rdpmc (+ (<< 1 30)
					   ,e-i)))
			  ))
	     )

	   (let ((count 0))
	     (foreach (e array)
		      ;(declare (type "const auto&" e))
		      (when (== 0 (% e 2))
			(incf count))))
	   
	   (let (,@ (loop for e in l
			  and e-i from 0
			  collect
			  `(,(format nil "new_~a" e) (__rdpmc (+ (<< 1 30) ,e-i)))
			  ))
	     )

	   (let (,@ (loop for e in l
			  and e-i from 0
			  collect
			  `(,(format nil "~a_count" e)
			    (- ,(format nil "new_~a" e)
			       ,e))
			  ))
	     ,@(loop for e in l
		     collect
		     (lprint :vars `(,(format nil "~a_count" e))))
	     )
	   (return 0)))))
  )


  
;; https://community.intel.com/t5/Software-Tuning-Performance/How-to-read-performance-counters-by-rdpmc-instruction/td-p/1009043


;; echo 1 > /proc/sys/kernel/nmi_watchdog
;; cat /proc/sys/kernel/nmi_watchdog 
;; /sys/devices/cpu/rdpmc

;; echo 2 | sudo tee /sys/devices/cpu/rdpmc    # enable RDPMC always, not just when a perf event is open
