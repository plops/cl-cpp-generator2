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
		  format)
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
		 (XShmGetImage display root ximg 0 0 "0x00ffffff")
		 (setf cv_img (cv--Mat height width CV_8UC4 ximg->data)))

	       #+nil
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (&key name type param initform param-name member-name) e
				    (let ((get (cl-change-case:pascal-case (format nil "get-~a" name))))
				      `(defmethod ,get ()
					 (declare (values 
						   ,(format nil "const ~a&" type))
						  (const))
					 (return ,param-name))))))
	       	       
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
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       ,(lprint :msg "start")

       (let ((img (cv--Mat))
	    )
	 (handler-case
	     (let ((screen (Screenshot 0 0 1920 1080)))
	       (screen img)
	       (cv--imshow (string "img") img)
	       (cv--waitKey 0)
	       )
	   ("const std::exception&" (e)
	     ,(lprint :vars `((e.what)))
	     (return 1)))
	 )
       
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))

