(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/127_opencv_amd/source03/")
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
      opencv2/highgui.hpp)

     "using namespace cv;"
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"

       (let ((x 8)
	     (y 3)
	     (square_len 4s-2)
	     (dict (aruco--getPredefinedDictionary
		    aruco--DICT_6X6_250))
	     (board
	       (new
		(aruco--CharucoBoard
		 (Size x y) square_len
		 (* .5 square_len)
		 dict)))
	     (img (Mat))
	     )
	 ;(declare (type "Ptr<aruco::CharucoBoard>" board))
	 (board->generateImage (cv--Size 800 600)
		      img
		      10 ;; marginsize
		      1 ;; bordebits
		      )
	 (imshow (string "charuco board")
		     img)
	 (waitKey 0)
	 )

       
       (return 0)))))


