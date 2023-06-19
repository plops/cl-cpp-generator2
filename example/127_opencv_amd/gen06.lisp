(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/127_opencv_amd/source06/")
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
      opencv2/aruco.hpp
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
	 (let ((dict (makePtr<aruco--Dictionary>
		      (aruco--getPredefinedDictionary
		       aruco--DICT_6X6_250)))
	       (frame (Mat))
	       (ids (std--vector<int>))
	       (corners (std--vector<std--vector<Point2f>>))
	     )
	   (while true
		  (>> camera frame)
		  (when (frame.empty)
		    break)
		  (aruco--detectMarkers frame dict corners ids)
		  (when (< 0 (ids.size))
		    (aruco--drawDetectedMarkers frame corners ids))
		  (imshow title
			  frame)
		  (when (<= 0 (waitKey 1))
		    break))))
       
       (return 0)))))


