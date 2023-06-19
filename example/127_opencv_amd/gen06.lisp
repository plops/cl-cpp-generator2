(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/127_opencv_amd/source05/")
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
      opencv2/opencv.hpp
      )

     "using namespace cv;"
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"
       (let ((camera (VideoCapture 0))
	     (title (string "Webcam"))
	     )
	 (unless (camera.isOpened)
	   ,(lprint :msg "Error: Could not open camera.")
	   (return 1))
	 (namedWindow title
		      WINDOW_AUTOSIZE)
	 (let ((frame (Mat)))
	   (while true
		  (>> camera frame)
		  (when (frame.empty)
		    break)
		  (imshow title
			  frame)
		  (when (<= 0 (waitKey 1))
		    break))))
       
       (return 0)))))


