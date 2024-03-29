(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/127_opencv_amd/source04/")
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
      opencv2/aruco/charuco.hpp
      ;opencv2/aruco_detector.hpp
      opencv2/highgui.hpp
      opencv2/imgcodecs.hpp)

     "using namespace cv;"
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"

       (comments "https://github.com/opencv/opencv_contrib/blob/7a4c0dfa861bbd4e5df7081949f685696eb9a94f/modules/aruco/samples/tutorial_charuco_create_detect.cpp#L53")
       (let (
	     (dict (makePtr<aruco--Dictionary>
		    (aruco--getPredefinedDictionary
		     aruco--DICT_6X6_250)))
	     
	     (img (imread (string "/home/martin/charucoboard.png")
			  IMREAD_COLOR))
	     (markerIds (std--vector<int>))
	     (markerCorners (std--vector<std--vector<Point2f>>))
	     )
	 (declare (type "Ptr<aruco::Dictionary>" dict))
	 (aruco--detectMarkers img dict
			       markerCorners
			       markerIds)

	 (aruco--drawDetectedMarkers
	  img
	  markerCorners
	  markerIds)
	 
					
	 (imshow (string "charuco board with detected markers")
		 img)
	 (waitKey 0)
	 )

       
       (return 0)))))


