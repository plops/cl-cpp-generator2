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
      opencv2/objdetect/aruco_dictionary.hpp
      opencv2/objdetect/aruco_board.hpp
      opencv2/objdetect/aruco_detector.hpp
      opencv2/objdetect/charuco_detector.hpp
      )

     "using namespace cv;"
     

     ,(let ((l-param
	      `((:name DetectorParameters
		       :fields (   adaptiveThreshWinSizeMin 
			     adaptiveThreshWinSizeMax 
			     adaptiveThreshWinSizeStep 
			     adaptiveThreshConstant
			     minMarkerPerimeterRate 
			     maxMarkerPerimeterRate 
			     polygonalApproxAccuracyRate 
			     minCornerDistanceRate 
			     minDistanceToBorder 
			     minMarkerDistanceRate 
			     cornerRefinementMethod 
			     cornerRefinementWinSize 
			     cornerRefinementMaxIterations 
			     cornerRefinementMinAccuracy
			     markerBorderBits 
			     perspectiveRemovePixelPerCell 
			     perspectiveRemoveIgnoredMarginPerCell 
			     maxErroneousBitsInBorderRate
			     minOtsuStdDev 
			     errorCorrectionRate 
			     aprilTagQuadDecimate 
			     aprilTagQuadSigma 
			     aprilTagMinClusterPixels 
			     aprilTagMaxNmaxima
			     aprilTagCriticalRad 
			     aprilTagMaxLineFitMse 
			     aprilTagMinWhiteBlackDiff 
			     aprilTagDeglitch
			     detectInvertedMarker 
			     useAruco3Detection
			     minSideLengthCanonicalImg 
			     minMarkerLengthRatioOriginalImg ) )
		(:name RefineParameters
		       :fields  (
				  minRepDistance
				  errorCorrectionRate
				  checkAllOrders
				  ))
		(:name CharucoParameters
		       :fields (minMarkers 
				 tryRefineMarkers
				 ) ))))
	`(do0
	  ,@(loop for e in l-param
		  collect
		  (destructuring-bind (&key name fields) e
		    `(defun print (p ) ;,(format nil "print~a" name) (p)
		       (declare (type ,(format nil "aruco::~a const &" name) p))
		       ,@(loop for e in fields
			       collect
			       (lprint :msg name :vars `((dot p ,e)))))))))
     
     
     
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
	       #+nil(dict0 (aruco--getPredefinedDictionary
			    aruco--DICT_6X6_250))
	       (dict (makePtr<aruco--Dictionary>
		      (aruco--getPredefinedDictionary
		       aruco--DICT_6X6_250))
		     )
	       (board
		 (std--make_unique<aruco--CharucoBoard> ;new
		  (aruco--CharucoBoard
		   (Size x y) square_len
		   (* .5 square_len)
		   *dict)))
	       (img (Mat))
	       )
					;(declare (type "Ptr<aruco::CharucoBoard>" board))
	   (board->generateImage (cv--Size 800 600)
				 img
				 10 ;; marginsize
				 1  ;; bordebits
				 )

	   (let ((dp (aruco--DetectorParameters)))

	     (print dp)
	     
	     )
	   (let ((rp (aruco--RefineParameters)))
	     (print rp))
	   (let ((cp (aruco--CharucoParameters)))
	     (print cp))
	   
	   (let ((markerDetector (aruco--ArucoDetector *dict dp rp))
		 (boardDetector (aruco--CharucoDetector *board cp dp rp))))
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
		 (marker_rejected (std--vector<std--vector<Point2f>>))
		 (allCorners (std--vector<Mat>))
		 (allIds (std--vector<Mat>))
		 )
					;(allCorners.reserve 100)
					;(allIds.reserve 100)
	     (while true
		    (do0
		     (comments "capture image")
		     (let ((key (cast char (waitKey waitTime))))
		       (>> camera frame)
		       (when (logior (frame.empty)
				     (== key 27))
			 break))

		     

		     
		     (do0
		      (comments "detect markers")
		      (markerDetector.detectMarkers frame corners ids marker_rejected)
		      (markerDetector.refineDetectedMarkers frame *board corners ids marker_rejected)
		      #+nil (let ( ;(detector_params (makePtr<aruco--DetectorParameters> (aruco--DetectorParameters)))
				  )
			      (aruco--detectMarkers frame dict corners ids ;detector_params marker_rejected
					 	    ))


		      
		      #+nil (do0 (comments "refinement will possibly find more markers")

				 (aruco--refineDetectedMarkers frame board corners ids marker_rejected))

		      
		      
		      #+nil (when (< 0 (ids.size))
			      (aruco--drawDetectedMarkers frame corners ids)))

		     
		     #+nil 
		     (do0
		      (comments "https://github.com/kyle-bersani/opencv-examples/blob/master/CalibrationByCharucoBoard/CalibrateCamera.py")
		      (comments "corners ids = detectMarkers img dict"
				"(drawDetectedMakers img corners)"
				"charucoCorners charucoIds = interpolateCorners corners ids img board"
				"collect charucoCorners")
		      
		      (comments "https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html")
		      (comments "https://github.com/CopterExpress/charuco_calibration/blob/master/charuco_calibration/src/calibrator.cpp"
				"contains example of how to accumulate allCharucoCorners")

		      (comments "https://github.com/UoA-CARES/stereo_calibration"))
		     (unless (ids.empty)
		       ,(lprint :vars `((ids.size)))
		       (do0 (comments "interpolate charuco corners (checker board corners, not the aruco markers)")
			    (let ((charucoCorners (Mat))
				  (charucoIds (Mat)))
			      (boardDetector.detectBoard frame charucoCorners charucoIds corners ids )
			      
			      #+nil (let ((res0 (aruco--interpolateCornersCharuco
						 corners
						 ids
						 frame
						 board
						 charucoCorners
						 charucoIds)))
				      ,(lprint :vars `(res0)
					       ))
			      (aruco--drawDetectedMarkers frame corners ids)
			      (when (<= 4 (dot charucoCorners
					       (size)
					       height))
				,(lprint :vars `((dot charucoCorners
						      (size)
						      height)))
				(aruco--drawDetectedCornersCharuco frame charucoCorners charucoIds)
				(allCorners.push_back charucoCorners)
				(allIds.push_back charucoIds))
			      )))
		     (imshow title
			     frame)
		     )
		    #+nil
		    (when (<= 0 (waitKey 1))
		      break))
	     #+nil
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

		 (comments "rvecs and tvecs are camera pose estimates")
		 (<< fs (string "cameraMatrix")
		     cameraMatrix
		     (string "distCoeffs")
		     distCoeffs)
		 (fs.release)
		 )))))
       
       (return 0)))))


