(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/132_v4l2/source01/src/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  

  (let* ((name `V4L2Capture)
	 (members `((device :type "const std::string&" :param t)
		    (buffers :type "std::vector<buffer>" :param nil)
		    (fd :type "int" :param nil))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			
			)
     :implementation-preamble
     `(do0
       (include<> fcntl.h
		  unistd.h
		  sys/ioctl.h
		  sys/mman.h
		  linux/videodev2.h

		  fstream
		  iostream
		  vector
		  string
		  cstring
		  stdexcept))
     :code `(do0
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name (,@(remove-if #'null
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
		 (setf fd_ (open (dot device_ (c_str))
				 O_RDWR))
		 (when (== -1 fd_)
		   (throw (std--runtime_error (+ (string "opening video device failed")
						 (std--string (std--strerror errno)))))
		   ))

	       (defmethod ~V4L2Capture ()
		 (declare (values :constructor))
		 (for-range (b buffers_)
			    (munmap b.start b.length))
		 (close fd_))

	       (defmethod startCapturing ()
		 (let ((type (v4l2_buf_type V4L2_BUF_TYPE_VIDEO_CAPTURE)))
		   (xioctl VIDIOC_STREAMON &type)))
	       
	       (defmethod stopCapturing ()
		 (let ((type (v4l2_buf_type V4L2_BUF_TYPE_VIDEO_CAPTURE)))
		   (xioctl VIDIOC_STREAMOFF &type)))

	       (defmethod setupFormat (width height pixelFormat)
		 (declare (type int width height pixelFormat))
		 (let ((f (v4l2_format (designated-initializer :type V4L2_BUF_TYPE_VIDEO_CAPTURE))))
		   (setf (dot f fmt pix pixelformat ) pixelFormat
			 (dot f fmt pix width) width
			 (dot f fmt pix height) height
			 )
		   (xioctl VIDIOC_S_FMT &f)
		   (let ((r (v4l2_requestbuffers (designated-initializer :count 1
									 :type  V4L2_BUF_TYPE_VIDEO_CAPTURE
									 :memory V4L2_MEMORY_MMAP))))
		     (xioctl VIDIOC_REQBUFS &r)
		     (buffers_.resize r.count)
		     (dotimes (i r.count)
		       (let ((buf (v4l2_buffer
				   (designated-initializer
				    :index i
				    :type  V4L2_BUF_TYPE_VIDEO_CAPTURE
				    :memory V4L2_MEMORY_MMAP
				    ))))
			 (xioctl VIDIOC_QUERYBUF &buf)
			 (setf (dot (aref buffers_ i)
				    length) buf.length
			       (dot (aref buffers_ i)
				    start) (mmap nullptr buf.length (or PROT_READ
									PROT_WRITE)
						 MAP_SHARED fd_ buf.m.offset) 
				    )
			 (when (== MAP_FAILED (dot (aref buffers_ i) start))
			   (throw (std--runtime_error (string "mmap failed")))))))))
	       	       
	       "private:"
	       (defstruct0 buffer
		 (start void*)
		 (length size_t))

	       (defmethod xioctl (request arg)
		 (declare (type "unsigned long" request)
			  (type void* arg))
		 (let ((r 0))
		   (space do
			  (progn
			    (setf r (ioctl fd_ request arg)))
			  while (paren (logand (== -1 r)
					       (== EINTR errno))))
		   (when (== -1 r)
		     (throw (std--runtime_error (std--strerror errno))))))

	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_)))))))))

  
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      iostream
      string
      ;complex
      vector
      ;algorithm
      
      ;chrono

      filesystem
      unistd.h
      cstdlib

      cmath)
     #+nil (include
      immapp/immapp.h
      implot/implot.h
      imgui_md_wrapper.h
      )
     (include 
	      V4L2Capture.h)
					     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       
       (let ((cap (V4L2Capture  (string "/dev/video0"))))
	 (cap.startCapturing)
       
	 (return 0))))
   :omit-parens t
   :format nil
   :tidy nil))

