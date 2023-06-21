(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/128_liquiddsp/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
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
      cmath
      complex
      vector)

     "using namespace std;"
     

     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"
       (let ((M (<< 1 3))
	     (numSymbols 1200)
	     (wLen 13)
	     (mu .05s0)
	     (alpha .6s0)
	     (w (vector<complex<float>> wLen))
	     (b (vector<complex<float>> wLen))
	     (xPrime (complex<float> 0s0)))
	 (comments "Initialize arrays")
	 (dotimes (i wLen)
	   (setf (aref w i)
		 (? (== i (/ wLen 2))
		    1s0
		    0s0)))
	 (let ((bufIndex 0))
	  (dotimes (n numSymbols)
	    (do0
	     (comments "x .. random transmitted phase-shift keying symbol"
		       "y .. computed received signal to be stored in buffer b")
	     (let ((x (exp (* (complex<float> 0 1)
			      2s0
			      (static_cast<float> M_PI)
			      (/ 
			       (static_cast<float> (% (rand)
						      M))
			       M))))
		   (y (+ (* (sqrt (- 1 alpha))
			    x)
			 alpha xPrime)))
	       (setf xPrime y
		     (aref b bufIndex) y
		     bufIndex (% (+ bufIndex 1)
				 wLen)))
	     (comments "compute equalizer output r")
	     (let ((r (complex<float> 0s0)))
	       (dotimes (i wLen)
		 (incf r (* (aref b (% (+ bufIndex i)
				       wLen))
			    (conj (aref w i))))))
	     (comments "compute expected signal (blind), skip first wLen symbols")
	     (let ((e (? (< n wLen)
			 (complex<float> 0s0)
			 (- r (/ r (abs r))))))
	       (dotimes (i wLen)
		 (decf (aref w i)
		       (* mu
			  (conj e)
			  (aref b (% (+ bufIndex i)
				     wLen)))))
	       ,(lprint :vars `(y r)))))))
       
       (return 0)))))


