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
  (defparameter *source-dir* #P"example/148_opencv_capture/source01/src/")
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

  
  
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
      opencv2/opencv.hpp
      format
      iostream
      memory
      )

     (include Screenshot.h)

    #+nil  (defun captureAndDisplay (win screen img)
       (declare (type "Screenshot&" screen)
		(type "const char*" win)
		(type "cv::Mat&" img))
       (screen img)
       (cv--imshow win img))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       ,(lprint :msg "start")
       (let ((img (cv--Mat))
	     (win (string "img"))
	     (frameRate 60s0)
	     (alpha .2s0)
	     (w (/ 1920 2))
	     (h (/ 1080 2)))
	 (cv--namedWindow win cv--WINDOW_NORMAL
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
				     (string ,name)
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
		 (while true
			#+nil
			(do0 (screen img)
			     (cv--imshow win img))

			
			
			#-nil (do0
			       (screen img)
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
		     
			#+nil (captureAndDisplay win screen img)
			(when (== 27 (cv--waitKey (/ 1000 60)))
			  (comments "Exit loop if ESC key is pressed")
			  break)
			)
	       ("const std::exception&" (e)
		 ,(lprint :vars `((e.what)))
		 (return 1)))))
	 )
       
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))
