(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/127_opencv_amd/source07/")
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
      opencv2/aruco/charuco.hpp
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

	 (let ((x 8)
	     (y 3)
	     (square_len 4s-2)
	       (dict0 (aruco--getPredefinedDictionary
		    aruco--DICT_6X6_250))
	       (dict (makePtr<aruco--Dictionary>
			(aruco--getPredefinedDictionary
			 aruco--DICT_6X6_250))
		       )
	       (board
		 (new
		  (aruco--CharucoBoard
		   (Size x y) square_len
		   (* .5 square_len)
		   dict0)))
	     (img (Mat))
	     )
	 ;(declare (type "Ptr<aruco::CharucoBoard>" board))
	 (board->generateImage (cv--Size 800 600)
		      img
		      10 ;; marginsize
		      1 ;; bordebits
		      )
	   )

	 
	 
	 (unless (camera.isOpened)
	   ,(lprint :msg "Error: Could not open camera.")
	   (return 1))
	 (namedWindow title
		      WINDOW_AUTOSIZE)

	 (let ((waitTime 10))
	   (comments "waitTime in milliseconds")
	   
	   (let (
		 (frame (Mat))
		 (ids (std--vector<int>))
		 (corners (std--vector<std--vector<Point2f>>))
		 (allCorners (std--vector<std--vector<std--vector<Point2f>>>))
		 (allIds (std--vector<std--vector<int>>))
		 )
	     (while true
		    (do0
		     (comments "capture image")
		     (>> camera frame)
		     (when (frame.empty)
		       break))
		    (do0
		     (comments "detect markers")
		     (aruco--detectMarkers frame dict corners ids)
		     #+nil (when (< 0 (ids.size))
		       (aruco--drawDetectedMarkers frame corners ids)))

		    (when (< 0 (ids.size))
		     (do0 (comments "interpolate charuco corners")
			  (let ((charucoCorners (Mat))
				(charucoIds (Mat)))
			    (aruco--interpolateCornersCharuco
			     corners
			     ids
			     frame
			     board
			     charucoCorners
			     charucoIds)
			    (when (< 0 (charucoCorners.total))
			      (comments "If at leas one charuco corner detected, draw the corners")
			      (aruco--drawDetectedCornersCharuco
			       frame
			       charucoCorners
			       charucoIds)
			      (comments "Collect data for calibration")
			      (allCorners.push_back corners)
			      (allIds.push_back ids)))))
		    (imshow title
			    frame)
		    (let ((key (cast char (waitKey waitTime))))
		      (when (== key 27)
			break))
		    #+nil
		    (when (<= 0 (waitKey 1))
		      break))
	     (when (< 0 (allIds.size))
	       (let ((cameraMatrix (Mat))
		     (distCoeffs (Mat))
		     (rvecs (std--vector<Mat>))
		     (tvecs (std--vector<Mat>))
		     (repError (aruco--calibrateCameraCharuco allCorners
							      allIds
							      board
							      (Size 640 480)
							      cameraMatrix
							      distCoeffs
							      rvecs
							      tvecs))
		     (fs (FileStorage (string "calibration.yaml")
				      FileStorage--WRITE)))
		 (<< fs (string "cameraMatrix")
		     cameraMatrix
		     (string "distCoeffs")
		     distCoeffs)
		 (fs.release)
		 )))))
       
       (return 0)))))


