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
      opencv2/highgui.hpp
      opencv2/imgcodecs.hpp)

     "using namespace cv;"
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"

       (let (
	     (dict (aruco--getPredefinedDictionary
		    aruco--DICT_6X6_250))
	     
	     (img (imread (string "/home/martin/charucoboard.png")
			  IMREAD_COLOR))
	     (markerIds (std--vector<int>))
	     (markerCorners (std--vector<std--vector<Point2f>>))
	     )
	 (aruco--detectMarkers img dict
			       markerCorners
			       markerIds)
	 
					
	 (imshow (string "charuco board")
		 img)
	 (waitKey 0)
	 )

       
       (return 0)))))


