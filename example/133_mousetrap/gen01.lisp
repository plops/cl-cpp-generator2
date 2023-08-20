(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(setf *features* (set-difference *features* (list :more
						  )))
(setf *features* (set-exclusive-or *features* (list :more
						    )))


(progn 
  
  (progn
    (defparameter *source-dir* #P"example/133_mousetrap/source01/src/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  (defparameter *benchmark-counter* 0)
  (defun benchmark (code)
    (let ((start (format nil "startBenchmark~2,'0d" *benchmark-counter*))
	  (end (format nil "endBenchmark~2,'0d" *benchmark-counter*))
	  (elapsed (format nil "elapsed~2,'0d" *benchmark-counter*))
	  (elapsed_ms (format nil "elapsed_ms~2,'0d" *benchmark-counter*)))
      (incf *benchmark-counter*)
      `(let ((,start (std--chrono--high_resolution_clock--now)))
	 ,code
	 (let ((,end (std--chrono--high_resolution_clock--now))
	       (,elapsed (std--chrono--duration<double> (- ,end ,start)))
	       (,elapsed_ms (* 1000 (dot ,elapsed (count))))
	       )
	   ,(lprint :msg (format nil "benchmark ~2,'0d" (- *benchmark-counter* 1))
	     :vars `(,elapsed_ms))))))
    
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     ;(include<> mousetrap.hpp)
     (include /home/martin/moustrap/include/mousetrap.hpp)
     "using namespace mousetrap;"

     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       (let ((app (Application (string "test.app"))))
	 (app.connect_signal_activate
	  (lambda (app)
	    (declare (type "Application&" app))
	    (let ((window (Window app))
		  (label (Label (string "Hello World!")))
		  )
	      (window.set_child label)
	      (window.present))))
	 (return (app.run)))))
   :omit-parens t
   :format nil
   :tidy nil))
