(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ()
  (defparameter *source-dir* #P"example/148_opencv_capture/source02/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  
  (let* ((class-name `Screenshot)
	 (members0 `((:name display :type Display* :initform nil)
		     (:name init :type bool :initform nil)
		     (:name root :type Window  :param nil  :initform nil)
		     (:name window-attributes :type XWindowAttributes  :initform nil)
		     (:name screen :type Screen*  :initform nil)
		     (:name shminfo :type XShmSegmentInfo :param nil  :initform nil)
		     (:name ximg :type XImage* :param nil  :initform nil)
		     (:name x :type int :param t)
		     (:name y :type int :param t)
		     (:name width :type int :param t)
		     (:name height :type int :param t)
		     ;(:name :type :param)
		     ))
	 (members (loop for e in members0
			collect
			(destructuring-bind (&key name type param (initform 0)) e
			  `(:name ,name
			    :type ,type
			    :param ,param
			    :initform ,initform
			    :member-name ,(intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))
			    :param-name ,(when param
					   (intern (string-upcase (cl-change-case:snake-case (format nil "~a" name)))))
			    #+nil (when param
			      (intern
			       (format nil "~a_"
				       (string-upcase
					(cl-change-case:snake-case (format nil "~a" name)))))))))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name class-name
     :headers `()
     :header-preamble `(do0
			(include<>
			 opencv2/opencv.hpp
			 X11/Xlib.h
			 X11/Xutil.h
			 X11/extensions/XShm.h
			 sys/ipc.h
			 sys/shm.h
      
			 )
			)
     :implementation-preamble
     `(do0
       (include<> stdexcept
		  memory
		  format
		  cerrno
		  cstring)
       )
     :code `(do0
	     
	     (defclass ,class-name ()
	       "public:"
	       
	       (defmethod ,class-name (,@(remove-if #'null
					  (loop for e in members
						collect
						(destructuring-bind (&key name type param initform param-name member-name) e
						  param-name))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (&key name type param initform param-name member-name) e
				       (when param
					 `(type ,type ,param-name)))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (&key name type param initform param-name member-name) e
					(cond
					  (param
					   `(,member-name ,param-name)) 
					  (initform
					   `(,member-name ,initform)))))))
		  (explicit)	    
		  (values :constructor))
		 (setf display (XOpenDisplay nullptr) )
		 (unless display
		     (throw (std--runtime_error (string "Failed to open display"))))
		 (setf root (DefaultRootWindow display))
		 (XGetWindowAttributes display root &window_attributes)
		 (setf screen window_attributes.screen)
		 (setf ximg (XShmCreateImage
			     display
			     (DefaultVisualOfScreen screen)
			     (DefaultDepthOfScreen screen)
			     ZPixmap
			     nullptr
			     &shminfo
			     width
			     height)
		       )
		 (setf shminfo.shmid (shmget IPC_PRIVATE (* ximg->bytes_per_line
							    ximg->height)
					     (or IPC_CREAT "0777")))
		 (when (< shminfo.shmid 0)
		   ,(lprint :vars `(errno (strerror errno)))
		   (throw (std--runtime_error (string "Fatal shminfo error!"))))
		 (setf ximg->data (reinterpret_cast<char*> (shmat shminfo.shmid 0 0))
		       shminfo.shmaddr ximg->data
		       shminfo.readOnly False
		       )
		 (unless (XShmAttach display &shminfo)
		   (throw (std--runtime_error (string "XShmAttach failed"))))
		 (setf init true))

	       (defmethod ~Screenshot ()
		 (declare (values :constructor))
		 (unless init
		   (XDestroyImage ximg))
		 (XShmDetach display &shminfo)
		 (shmdt shminfo.shmaddr)
		 (shmctl shminfo.shmid IPC_RMID 0)
		 (XCloseDisplay display))
	       
	       (defmethod "operator()" (cv_img)
		 (declare (type "cv::Mat&" cv_img))
		 (when init
		   (setf init false))
		 (XShmGetImage display root ximg x y "0x00ffffff")
		 (setf cv_img (cv--Mat height width CV_8UC4 ximg->data)))

	       
	       ,@(remove-if
		  #'null
	          (loop for e in members
			appending
			(destructuring-bind (&key name type param initform param-name member-name) e
			  (let ((get (cl-change-case:pascal-case (format nil "get-~a" name)))
				(set (cl-change-case:pascal-case (format nil "set-~a" name)))
				(const-p (let* ((s  (format nil "~a" type))
						(last-char (aref s (- (length s)
								      1))))
					   (not (eq #\* last-char)))))
			    `((defmethod ,get ()
				(declare ,@(if const-p
					       `((const)
						 (values ,(format nil "const ~a&" type)))
					       `((values ,type))))
				(return ,member-name))
			      (defmethod ,set (,member-name)
				(declare (type ,type ,member-name))
				(setf (-> this ,member-name)
				      ,member-name)))))))
	       "private:"
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (&key name type param initform param-name member-name) e
				    `(space ,type ,member-name))))))))


  
  (make-class 'PPOCRDet
	      :members0 `((:name model-path :type "std::string" :param t
			   :initform (string "text_detection_en_ppocrv3_2023may_int8.onnx")
			   )
			  (:name input-size :type "cv::Size" :param t  :initform (cv--Size 736 736)
				 :setter (model.setInputParams (/ 1s0 255s0)
			       			   input_size
						   (cv--Scalar 122.67891434 116.66876762 104.00698793)))
			  (:name binary-threshold :type float  :param t  :initform .3s0
				 :setter (model.setBinaryThreshold binary_threshold))
			  (:name polygon-threshold :type float  :param t  :initform .5s0
				 :setter (model.setPolygonThreshold polygon_threshold))
			  (:name max-candidates :type int  :param t  :initform 200
				 :setter (model.setMaxCandidates max_candidates))
			  (:name unclip-ratio  :type double  :param t  :initform 2.0
				 :setter (model.setUnclipRatio unclip_ratio))
			  (:name backend-id :type "cv::dnn::Backend" :param t :initform cv--dnn--DNN_BACKEND_DEFAULT
				 :setter (model.setPreferableBackend backend_id))
			  (:name target-id :type "cv::dnn::Target" :param t :initform cv--dnn--DNN_TARGET_CPU
				 :setter  (model.setPreferableTarget target_id))
			  (:name model :type "cv::dnn::TextDetectionModel_DB"
			   :initform (cv--dnn--TextDetectionModel_DB (cv--dnn--readNet model_path))))
	      :header-preamble `(do0 (comments "heaeder")
			      (include<> string
					 opencv2/opencv.hpp)
			    
			      )
	      :implementation-preamble
	      `(do0
		(comments "implementation"))
	      :constructor-code `(do0
				  ,@(loop for e in `((setPreferableBackend backend_id)
						     (setPreferableTarget target_id)
						     (setBinaryThreshold binary_threshold)
						     (setPolygonThreshold polygon_threshold)
						     (setUnclipRatio unclip_ratio)
						     (setMaxCandidates max_candidates)
						     (setInputParams (/ 1s0 255s0)
								     input_size
								     (cv--Scalar 122.67891434 116.66876762 104.00698793)))
					  collect
					  `(dot model ,e)))
	      :class-code
	      `((defmethod infer (image)
		  (declare (values "std::pair<std::vector<std::vector<cv::Point>>,std::vector<float>>")
			   (type "cv::Mat" image))
		  (CV_Assert (logand (== image.rows
					 input_size.height)
				     (string "height of input image != net input size")))
		  (CV_Assert (logand (== image.cols
					 input_size.width)
				     (string "width of input image != net input size")))
		  (let ((pt ("std::vector<std::vector<cv::Point>>"))
			(confidence ("std::vector<float>")))
		    (model.detect image pt confidence)
		    (return ("std::make_pair< std::vector<std::vector<cv::Point>> &, std::vector<float> &>"
			     pt confidence)))))
	      )

  (make-class 'CRNN
	      :members0 `((:name model-path :type "std::string" :param t
			   :initform (string "text_recognition_CRNN_EN_2022oct_int8.onnx")
			   )
			  (:name input-size :type "cv::Size" :param nil  :initform (cv--Size 100 32)
				 )
			 
			  (:name backend-id :type "cv::dnn::Backend" :param t :initform cv--dnn--DNN_BACKEND_DEFAULT
				 :setter (model.setPreferableBackend backend_id))
			  (:name target-id :type "cv::dnn::Target" :param t :initform cv--dnn--DNN_TARGET_CPU
				 :setter  (model.setPreferableTarget target_id))
			  (:name model :type "cv::dnn::Net"
			   :initform (cv--dnn--readNet model_path))
			  (:name charset :type "std::vector<std::u16string>"
				 :initform (loadCharset (string "CHARSET_EN_36")))
			  (:name target_vertices :type "cv::Mat" :initform ((lambda ()
									      (declare (capture this))
									      (let ((target_vertices (cv--Mat 4 1 CV_32FC2))))
									      ,@(loop for (e f) in `((0 (0 (- this->input_size.height 1)))
												     (1 (0 0))
												     (2 ((- this->input_size.width 1) 0))
												     (3 ((- this->input_size.width 1)
													 (- this->input_size.height 1))))
										      collect
										      `(setf (target_vertices.row ,e)
											     (cv--Vec2f ,@f)))
									      (return target_vertices)))
			   ))
	      :header-preamble `(do0 (comments "heaeder")
			      (include<> string
					 opencv2/opencv.hpp)
			    
			      )
	      :implementation-preamble
	      `(do0
		(comments "implementation")
		(include "charset_32_94_3944.h"))
	      :constructor-code `(do0
				  )
	      :class-code
	      `(

		(defmethod preprocess (image rbbox)
		  (declare (values "cv::Mat")
			   (type "cv::Mat" image rbbox))
		  (comments "remove conf, reshape and ensure all is np.float32")
		  
		  (let ((vertices (cv--Mat)))
		    (dot rbbox
		       (reshape  2 4)
		       (convertTo vertices CV_32FC2)))

		  (let ((rotationMatrix (cv--getPerspectiveTransform vertices
								     target_vertices))
			(cropped (cv--Mat)))
		    (cv--warpPerspective image cropped rotationMatrix input_size))

		  (comments "CN can detect digits 0-9 letters a-zA-z and some special characters")
		  (comments "FIXME: what about EN?")
		  (cv--cvtColor cropped cropped cv--COLOR_BGR2GRAY)
		  (return (cv--dnn--blobFromImage cropped
						  (/ 1s0 127.5s0)
						  input_size
						  (cv--Scalar--all 127.5s0)))
		  )

		(defmethod infer (image rbbox)
		  (declare (values "std::u16string")
			   (type "cv::Mat" image rbbox))
		  (let ((inputBlob (preprocess image rbbox))))
		  (model.setInput inputBlob)
		  (let ((outputBlob (model.forward))))
		  (return (postprocess outputBlob)))

		(defmethod postprocess (outputBlob)
		  (declare (values "std::u16string")
			   (type "cv::Mat" outputBlob))
		  (let ((character (outputBlob.reshape 1 (aref outputBlob.size 0)))
			(text (std--u16string "u\"\"")))
		    (dotimes (i character.rows)
		      (let ((minVal 1d12)
			    (maxVal -1d12)
			    (maxIdx (cv--Point)))
			(cv--minMaxLoc (character.row i)
				       &minVal
				       &maxVal
				       nullptr
				       &maxIdx)
			(comments "decode characters from outputBlob")
			(if (== 0 maxIdx.x)
			    (incf text "u\"-\"")
			    (incf text (aref charset (- maxIdx.x 1))))))
		    (comments "adjacent same letters and background text must be removed to get final output")
		    (let ((textFilter (std--u16string "u\"\"")))
		      (dotimes (i (text.size))
			(when (!= "u'-'"
					   (aref text i))
			  (when (not (logand (< 0 i)
					     (== (aref text i)
						 (aref text (- i 1)))))
			    (incf textFilter (aref text i)))))
		      (return textFilter)
		      ))))
	      )
  
  
  
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
      opencv2/opencv.hpp
      format
      ;iostream
      ;memory
      codecvt)

     (include Screenshot.h
	      PPOCRDet.h
	      CRNN.h)
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       ,(lprint :msg "start")
       (let ((img (cv--Mat))
	     (win (string "img"))
	     (frameRate 60s0)
	     (alpha .2s0)
	     (w (* 32 4) ; 320
		)
	     (h (* 32 3) ; 256
		))
	 (cv--namedWindow win		;cv--WINDOW_NORMAL
			  cv--WINDOW_AUTOSIZE
					;cv--WINDOW_GUI_EXPANDED
			  )
	 
	 (cv--moveWindow win w 100)
	 (cv--resizeWindow win w h)
	 ,(let ((l-gui `((:name x :start 20 :max (- 1920 w) :code (screen->SetX value) :param screen :param-type Screenshot)
			 (:name y :start 270 :max (- 1080 h) :code (screen->SetY value) :param screen :param-type Screenshot)
			 (:name clipLimit :start 13 :max 100))))
	    `(do0
	      ,@(loop for e in l-gui
		      collect
		      (destructuring-bind (&key name start max code param param-type) e
			`(let ((,name ,start))
			   )))
	      (let ((screen (Screenshot (static_cast<int> x) (static_cast<int> y) w h))))
	      ,@(loop for e in l-gui
		      collect
		      (destructuring-bind (&key name start max code param param-type) e
			(let ((args `(
				      (string ,name)
				      win
				      (ref ,name)
				      ,max
				      )))
			  (when code
			    (setf args (append args
					       `((lambda (value v)
						   (declare (type void* v)
							    (type int value)
							    (capture ""))
						   ,(if (and param param-type)
							`(let ((,param (,(format nil "reinterpret_cast<~a*>" param-type)
									v)))))
						   ,code))))
			    (when param
			      (setf args (append args
						 `(,(format nil "reinterpret_cast<void*>(&~a)" param))))))
			  `(do0
			    (cv--createTrackbar ,@args)))))
	     

	    
	      
	     
	      (handler-case
		  (do0
		   (let ((binary_threshold .3)
			 (polygon_threshold .5)
			 (max_candidates 200)
			 (unclip_ratio 2s0)
			 (detector (PPOCRDet (string "text_detection_en_ppocrv3_2023may_int8.onnx")
					     (cv--Size w h)
					     binary_threshold
					     polygon_threshold
					     max_candidates
					     unclip_ratio
					     cv--dnn--DNN_BACKEND_DEFAULT
					     cv--dnn--DNN_TARGET_CPU
					     )))
		     )
		   (let ((recognizer (CRNN))))
		   
		   (while true
			
			

			
			  (screen img)
			  (let ((img3 (cv--Mat)))
			    (cv--cvtColor img img3 cv--COLOR_BGRA2RGB)
			    (let ((result (detector.infer img3))))
			    (cv--polylines img result.first true (cv--Scalar 0 255 0)
					   4)
			    #+nil(for-range (pts results.first)
					    (dotimes (i 4)
					      ()))
			    (when (logand (< 0 (result.first.size))
					       (< 0 (result.second.size)))
			      (let ((texts (std--u16string))
				    (score (result.second.begin)))
				(for-range (box result.first)
					   (let ((res (dot (cv--Mat box)
							   (reshape 2 4))))
					     (incf texts (+ "u'-'"
							    (recognizer.infer img3 res)
							    "u'-'"))
					     ))
				(let ((converter ("std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t>")))
					       (let ((ctext (converter.to_bytes texts)))
						 ,(lprint :vars `(ctext)))))))
			  (if (<= clipLimit 99)
			      (do0
			     
			       (let ((lab (cv--Mat)))
				 (cv--cvtColor img lab cv--COLOR_BGR2Lab)
				 (let ((labChannels (std--vector<cv--Mat>)))
				   (cv--split lab labChannels))
				 (let ((clahe (cv--createCLAHE)))
				   (declare (type "cv::Ptr<cv::CLAHE>" clahe))
				   (-> clahe (setClipLimit clipLimit))
				   (let ((claheImage (cv--Mat)))
				     (clahe->apply (aref labChannels 0)
						   claheImage)
				     (claheImage.copyTo (aref labChannels 0)))
				   (let ((processedImage (cv--Mat)))
				     (cv--merge labChannels lab)
				     (cv--cvtColor lab processedImage cv--COLOR_Lab2BGR))))
			       (cv--imshow win processedImage))
			      (do0 
			       (cv--imshow win img)))
		     
			  #+nil (captureAndDisplay win screen img)
			  (when (== 27 (cv--waitKey (/ 1000 60)))
			    (comments "Exit loop if ESC key is pressed")
			    break)
			  ))
		("const std::exception&" (e)
		  ,(lprint :vars `((e.what)))
		  (return 1)))))
	 )
       
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))
