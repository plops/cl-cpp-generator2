(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/126_quant_pow/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)

  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<> armadillo
		iostream)
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"

       (comments "N .. Number of discretizatino points"
		 "L .. Size of the box"
		 "dx .. Grid spacing")
       (let ((N 1000)
	     (L 1d0)
	     (dx (/ L (+ N 1)))
	     (H (arma--sp_mat N N)))
	 (dotimes (i N)
	   (when (< 0 i)
	     (setf (H i (- i 1))
		   (/ -1d0
		      (* dx dx))))
	   (setf (H i i)
		 (/ 2d0
		    (* dx dx)))
	   (when (< i (- N 1))
	     (setf (H i (+ i 1))
		   (/ -1d0
		      (* dx dx)))))
	 (let ((psi ("arma::randu<arma::vec>" N)))))
       
       (return 0)))
   :format t
   :tidy t))


