(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(let ((log-preamble `(do0 (include <iostream>
				   <iomanip>
				   <chrono>
				   <thread>
				   <mutex>)
			  "extern std::mutex g_stdout_mutex;"
			  "extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;")))
  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file
    (defparameter *source* "02source")
    (defparameter *source-dir* (format nil "example/72_emsdk/~a/" *source*))
    (load "util.lisp")


    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `Charuco
     :headers `()
     :header-preamble `(do0 ,@(loop for e in `( core/cvstd core/mat ;videoio
						core imgproc   aruco/charuco
						)
				    collect
				    `(include ,(format nil "<opencv2/~a.hpp>" e)))
			    "namespace cv { namespace aruco { class CharucoBoard;class Dictionary; struct DetectorParameters; }}"
			    #+nil (do0 (include
					<opencv2/core/cvstd.hpp>
					<opencv2/core/mat.hpp>)
				       ,(format nil "namespace cv { ~{class ~a;~} }" `(Dictionary CharucoBoard DetectorParameters)))
			    )
     :implementation-preamble `(do0

				,log-preamble
				,@(loop for e in `( core/cvstd core/mat; videoio
						    aruco/charuco
						    aruco
						    aruco/dictionary
						    core/types
						    imgproc
						    )
					collect
					`(include ,(format nil "<opencv2/~a.hpp>" e)))
				)
     :code (let ((def-members `((:name squares_x :type int :default 8)
				(:name squares_y :type int :default 4)
				(:name square_length :type float :default .04s0)
				(:name marker_length :type float :default .02s0)
				(:name dict_int :type int :init-form cv--aruco--DICT_6X6_250)
				(:name board_dict :type "cv::Ptr<cv::aruco::Dictionary>"
				       :init-form (cv--aruco--getPredefinedDictionary
						   dict_int))
				(:name board :type "cv::Ptr<cv::aruco::CharucoBoard> " :init-form
				       (cv--aruco--CharucoBoard--create
					squares_x
					squares_y
					square_length
					marker_length
					board_dict))
				(:name params :type "cv::Ptr<cv::aruco::DetectorParameters>"
				       :init-form  (cv--aruco--DetectorParameters--create))
				(:name board_img :type "cv::Mat" :no-construct t)
				(:name board_img3 :type "cv::Mat" :no-construct t)
				(:name camera_matrix :type "cv::Mat" :no-construct t)
				(:name dist_coeffs :type "cv::Mat" :no-construct t)
				#+nil (:name cap_fn :type "std::string"
					     :default (string
						       ;;"/dev/video2"
						       "/dev/video0"
						       ))
				#+nil (:name cap :type "cv::VideoCapture" :init-form (cv--VideoCapture cap_fn))
				)))
	     `(do0
	       (defclass Charuco ()
		 "public:"
		 ,@(loop for e in def-members
			 collect
			 (destructuring-bind (&key name type init-form default no-construct) e
			   (format nil "~a ~a;" type name)))

		 (defmethod Charuco (&key
				       ,@(remove-if
					  #'null
					  (loop for e in def-members
						collect
						(destructuring-bind (&key name type init-form default no-construct) e
						  (when default
						    `(,(intern (string-upcase (format nil "~a_" name)))
						       ,default))))))
		   (declare
		    ,@(remove-if
		       #'null
		       (loop for e in def-members
			     collect
			     (destructuring-bind (&key name type init-form default no-construct) e
			       (when default
				 `(type ,type ,(intern (string-upcase (format nil "~a_" name))))))))

		    (construct
		     ,@(remove-if
			#'null
			(loop for e in def-members
			      collect
			      (destructuring-bind (&key name type init-form default no-construct) e
				(if init-form
				    `(,name ,init-form)
				    (unless no-construct
				      `(,name ,(intern (string-upcase (format nil "~a_" name)))))))))
		     )
		    (values :constructor))
		   (do0
		    ,(lprint :msg "opencv initialization")
		    (board->draw (cv--Size 1600 800)
				 board_img3
				 10 1
				 )
		    (cv--cvtColor board_img3 board_img cv--COLOR_BGR2RGBA)
		    ,(lprint :msg "charuco board has been converted to RGBA")
		    #+nil (if (cap.isOpened)
			      ,(lprint :msg "opened video device" :vars `(cap_fn (cap.getBackendName)))
			      ,(lprint :msg "failed to open video device" :vars `(cap_fn )))
		    ))
		 #+nil (defmethod PrintProperties ()
			 ,(let ((cam-props `(BRIGHTNESS CONTRAST SATURATION HUE GAIN EXPOSURE
							MONOCHROME SHARPNESS AUTO_EXPOSURE GAMMA
							BACKLIGHT TEMPERATURE AUTO_WB WB_TEMPERATURE)))
			    `(let ((cam_w (cap.get cv--CAP_PROP_FRAME_WIDTH))
				   (cam_h (cap.get cv--CAP_PROP_FRAME_HEIGHT))
				   (cam_fps (cap.get cv--CAP_PROP_FPS))
		       		   (cam_format (cap.get cv--CAP_PROP_FORMAT))
				   ,@(loop for e in cam-props
					   collect
					   `(,(string-downcase (format nil "cam_~a" e))
					      (dot cap (get ,(format nil "cv::CAP_PROP_~a" e))))))
			       ,(lprint :vars `(cam_w cam_h cam_fps cam_format
						      ,@(loop for e in cam-props
							      collect
							      (string-downcase (format nil "cam_~a" e)))))

			       )))

		 #+nil(defmethod Capture ()
			(declare (values "cv::Mat"))
			(do0; "cv::Mat img3,img;"
			 (>> cap img3)
					;(cv--split img spl)
					;,(lprint :msg "received camera image")
					;(cv--cvtColor img3 img cv--COLOR_BGR2RGBA)
					;(cv--cvtColor img3 img cv--COLOR_BGR2GRAY)
					;,(lprint :msg "converted camera image")
			 )
			(return img3))
		 ,@(remove-if
		    #'null
		    (loop for e in def-members
			  collect
			  (destructuring-bind (&key name type init-form default no-construct) e
			    `(defmethod ,(format nil "get_~a" name) ()
			       (declare (values ,type))
			       (return ,name)))))
		 (defmethod Shutdown ()
		   #+nil
		   (do0 ,(lprint :msg "release video capture device (should be done implicitly)")
			(cap.release))
		   )

		 ))))

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `index
     :headers `()
     :header-preamble `(do0
			(include<>
			 emscripten.h

			 ))
     :implementation-preamble `(do0
				,log-preamble
				(include<>
				 opencv2/imgproc.hpp)
				(include "Charuco.h"))
     :code `(do0
	     (do0 "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"
		  "std::mutex g_stdout_mutex;")
	     (defun main (argc argv)
	       (declare (type int argc)
			(type char** argv)
			(values "extern \"C\" int"))
	       (setf g_start_time ("std::chrono::high_resolution_clock::now"))
	       (progn
		 ,(lprint :msg "enter program" :vars `(argc (aref argv)))
		 (let ((ch (Charuco)))
		   (ch.Shutdown))
		 ,(lprint :msg "exit program")
		 (return 0)))
	     ))

    (let ((fn (format nil "build_opencv.sh" *source*)))
      ;; pip install beautysh
      (with-open-file (s fn
			 :direction :output
			 :if-exists :supersede
			 :if-does-not-exist :create)
	(let ((contrib-modules-off
	       `(alphamat		; aruco
		 barcode  bgsegm  bioinspired  ccalib  cnn_3dobj  cvv  datasets  dnn_objdetect  dnn_superres  dnns_easily_fooled  dpm  face  freetype  fuzzy  hdf  hfs  img_hash  intensity_transform  julia  line_descriptor  matlab  mcc  optflow  ovis  phase_unwrapping  plot  quality  rapid  reg  rgbd  saliency  sfm  shape  stereo  structured_light  superres  surface_matching  text  tracking  videostab  viz  wechat_qrcode  xfeatures2d  ximgproc  xobjdetect  xphoto ))
	      (build-options `(build_wasm
			       threads
			       simd
					;webnn

			       ))
	      )
	  (macrolet ((out (fmt &rest rest)
		       `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	    (out "cd /home/martin/src/opencv")
	    (out "source /home/martin/src/emsdk/emsdk_env.sh")
	    (out "emcmake python  ./platforms/js/build_js.py build_wasm  ~{--~a~^ ~}  --cmake_option=\"-DOPENCV_EXTRA_MODULES_PATH=/home/martin/src/opencv_contrib/modules ~{-DBUILD_opencv_~a=OFF~^ ~}\""
		 build-options
		 contrib-modules-off))))
      (sb-ext:run-program "/home/martin/.local/bin/beautysh"
			  (list "-i"  (namestring fn)))
      )

    (let ((fn (format nil "~a/CMakeLists.txt" *source*)))
      ;; pip install cmakelang
      (with-open-file (s fn
			 :direction :output
			 :if-exists :supersede
			 :if-does-not-exist :create)
	(let ((dbg "-ggdb -O0 ")
	      (asan "" ; "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
		)
	      (show-err " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow  -Wswitch-default -Wundef -Werror  -Wno-unused -Wno-unused-parameter"
		;;
		;; -Wold-style-cast -Wsign-conversion
		;; "-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "
		))
	  (macrolet ((out (fmt &rest rest)
		       `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	    (out "cmake_minimum_required( VERSION 3.0 )")

	    (out "project( example LANGUAGES CXX )")

	    (out "set( CMAKE_CXX_STANDARD 17 )")
	    (out "set( CMAKE_CXX_STANDARD_REQUIRED True )")
	    (out "set( OpenCV_DIR /home/martin/src/opencv/build_wasm/ )")
	    (out "set( OpenCV_STATIC ON )")
	    (out "find_package( OpenCV REQUIRED )")
	    (out "include_directories( ${OpenCV_INCLUDE_DIRS} /home/martin/src/opencv_contrib/modules/aruco/include )")
	    (out "option( BUILD_WASM \"Build Webassembly\" ON )")
	    (progn
	      (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
					;(out "set( USE_FLAGS \"-s USE_SDL=2\" )")
					;(out "set( CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${USE_FLAGS}\" )")
	      (out "set( CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG}  ~a ~a ~a \")"
		   dbg asan show-err)

	      )

	    (out "set( CMAKE_EXECUTABLE_SUFFIX \".html\" )")

	    (out "set( SRCS ~{~a~^~%~} )"
		 (append
		  (directory (format nil "~a/*.cpp" *source*))
		  (directory "/home/martin/src/opencv_contrib/modules/aruco/src/*.cpp")))

	    (out "add_executable( index ${SRCS} )")
	    (out "target_link_libraries( index ${OpenCV_LIBS} )")
	    )))
      (sb-ext:run-program "/home/martin/.local/bin/cmake-format"
			  (list "-i"  (namestring fn))))))

