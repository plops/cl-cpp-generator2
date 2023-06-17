(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/127_opencv_amd/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  (let ((interface-name "CheckerboardDisplayInterface")
	(name `ArucoCheckerboardDisplay)
	(members `((squares-x :type int :param t)
		   (squares-y :type int :param t)
		   (square-length :type int :param t)
		   (dictionary :type "cv::Ptr<cv::aruco::Dictionary>" :param t)
		   #+nil (board-size :type "cv::Size" ;:initform (paren squaresX squaresY)
			       )
		   (board-image :type "cv::Mat")
		   )))

    (write-source 
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames (format nil "~a.hpp"
			       interface-name) 
		       *source-dir*))
     `(do0
       "#pragma once"
       (include<> 
	opencv2/aruco.hpp)
       (defclass+ ,interface-name ()
	 "public:"
	 ,(format nil "virtual ~~~a() = default;" interface-name)
	 (space virtual void
		(displayCheckerboard
		 "int squaresX"
		 "int squaresY"
		 "int squareLength"
		 "cv::Ptr<cv::aruco::Dictionary> dictionary")
		"= 0;")))
     )
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include ,(format nil "~a.hpp" interface-name)))
     :implementation-preamble
     `(do0
       )
     :code `(do0
	     (defclass ,name "public CheckerboardDisplayInterface"	 
	       "public:"
	       #+nil (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param (initform 0)) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param (initform 0)) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param (initform 0)) e
					(let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					      (nname_ (format nil "~a_"
							      (cl-change-case:snake-case (format nil "~a" name)))))
					  (cond
					    (param
					     `(,nname_ ,nname))
					    (initform
					     `(,nname_ ,initform)))))))
		   )
		  (explicit)	    
		  (values :constructor))
		 
		 )
		 (defmethod displayCheckerboard (squaresX squaresY squareLength dictionary)
		   (declare (type int squaresX squaresY squareLength)
			    (type "cv::Ptr<cv::aruco::Dictionary>" dictionary))
		   (cv--aruco--drawCharucoBoard (cv--Size squaresX squaresY) ; boardSize
						squareLength
					       (/ squareLength 2)
					       dictionary
					       boardImage)
		   (cv--imshow (string "checkerboard")
			       boardImage)
		   (cv--waitKey 0))
	       "private:"
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_)))))

	       ))))
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
      fruit/fruit.h
      iostream
      opencv2/core/ocl.hpp)
     (include "CheckerboardDisplayInterface.hpp"
	      "ArucoCheckerboardDisplay.h")

     "using fruit::Component;"
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"
       ,(lprint :vars `((cv--ocl--haveOpenCL)))

       (let ((injector (fruit--Injector<CheckerBoardDisplayInterface>
			(getCheckerboardDisplayComponent)))
	     (display (injector.get<CheckerboardDisplayInterface*>))
	     (dictionary (cv--aruco--getPredefinedDictionary cv--aruco--DICT_6x6_250))
	     )
	 (display->displayCheckerboard 5 7 100 dictionary))
       
       (return 0)))
   :format t
   :tidy t))


