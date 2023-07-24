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

  (defun xioctl (args)
    (destructuring-bind (&key request var) args
     `(xioctl ,(cl-change-case:constant-case (format nil "vidioc-~a" request))
	      ,var
	      (string ,request))))

  (let* ((name `V4L2Capture)
	 (members `((device :type "const std::string&" :param t)
		    (buffer-count :type int :param t)
		    (buffers :type "std::vector<buffer>" :param nil)
		    (fd :type "int" :param nil)
		    (width :type int)
		    (height :type int))))
    
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
						 (std--string (std--strerror errno)))))))

	       (defmethod ~V4L2Capture ()
		 (declare (values :constructor))
		 (for-range (b buffers_)
			    (munmap b.start b.length))
		 (close fd_))

	       (defmethod startCapturing ()
		 ,(lprint :msg "startCapturing")
		 (let ((type (v4l2_buf_type V4L2_BUF_TYPE_VIDEO_CAPTURE)))
		   ,(xioctl `(:request STREAMON :var &type))))
	       
	       (defmethod stopCapturing ()
		 ,(lprint :msg "stopCapturing")
		 (let ((type (v4l2_buf_type V4L2_BUF_TYPE_VIDEO_CAPTURE)))
		   ,(xioctl `(:request STREAMOFF :var &type))
		   ))

	       (defmethod setupFormat (width height pixelFormat)
		 (declare (type int width height pixelFormat))
		 ,(lprint :msg "setupFormat"
			  :vars `(width height pixelFormat))
		 (let ((f (v4l2_format (designated-initializer :type V4L2_BUF_TYPE_VIDEO_CAPTURE))))
		   (setf (dot f fmt pix pixelformat ) pixelFormat
			 (dot f fmt pix width) width
			 (dot f fmt pix height) height
			 (dot f fmt pix field) V4L2_FIELD_ANY
			 )
		   ,(xioctl `(:request s-fmt :var &f))
		   (unless (== f.fmt.pix.pixelformat pixelFormat)
		     ,(lprint :msg "warning: we don't get the requested pixel format"
			      :vars `(f.fmt.pix.pixelformat
				      pixelFormat)))
		   (setf width_ (dot f fmt pix width)
			 height_ (dot f fmt pix height))
		   (let ((r (v4l2_requestbuffers (designated-initializer :count buffer_count_
									 :type  V4L2_BUF_TYPE_VIDEO_CAPTURE
									 :memory V4L2_MEMORY_MMAP))))
		     ,(lprint :msg "prepare several buffers"
			      :vars `(buffer_count_))
		     ,(xioctl `(:request reqbufs :var &r)) 
		     (buffers_.resize r.count)
		     (dotimes (i r.count)
		       (let ((buf (v4l2_buffer
				   (designated-initializer
				    :index i
				    :type  V4L2_BUF_TYPE_VIDEO_CAPTURE
				    :memory V4L2_MEMORY_MMAP))))
			 
			 ,(xioctl `(:request querybuf :var &buf))
			 (setf (dot (aref buffers_ i)
				    length)
			       buf.length
			       (dot (aref buffers_ i)
				    start) (mmap nullptr buf.length (or PROT_READ
									PROT_WRITE)
						 MAP_SHARED fd_ buf.m.offset) 
				    )
			 ,(lprint :msg "mmap memory for buffer"
				  :vars `(i buf.length (dot (aref buffers_ i) start)))
			 (when (== MAP_FAILED (dot (aref buffers_ i) start))
			   (throw (std--runtime_error (string "mmap failed"))))
			 ,(xioctl `(:request qbuf :var &buf)))))))

	       (defmethod getFrame (filename)
		 (declare (type "std::string" filename))
		 (let ((buf (v4l2_buffer (designated-initializer :type V4L2_BUF_TYPE_VIDEO_CAPTURE
								 :memory V4L2_MEMORY_MMAP))))
					
		   ,(xioctl `(:request dqbuf :var &buf))
		   )
		 (let ((outFile (std--ofstream filename std--ios--binary)))
		   (<< outFile (string "P6\\n")
		       width_
		       (string " ")
		       height_
		       (string " 255\\n")
		       
		       )
		   (outFile.write (static_cast<char*> (dot (aref buffers_ buf.index)
							   start))
				  buf.bytesused)
		   (outFile.close)
		   ,(xioctl `(:request qbuf :var &buf))))
	       	       
	       "private:"
	       (defstruct0 buffer
		   (start void*)
		 (length size_t))

	       (defmethod xioctl (request arg str)
		 (declare (type "unsigned long" request)
			  (type void* arg)
			  (type "const std::string&" str))
		 (let ((r 0))
		   (space do
			  (progn
			    (setf r (ioctl fd_ request arg)))
			  while (paren (logand (== -1 r)
					       (== EINTR errno))))
		   (when (== -1 r)
		     (throw (std--runtime_error (+ (string "ioctl ")
						   str
						   (string " ")
						   (std--strerror errno)
						   ))))))

	       
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

      cmath
      linux/videodev2.h)
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
       (handler-case
	   (let ((cap (V4L2Capture  (string "/dev/video0")
				    3)))
	     (cap.setupFormat 320 240 V4L2_PIX_FMT_RGB24)
	     (cap.startCapturing)
	     (usleep 64000)
	     (dotimes (i 9)
	       (cap.getFrame (+ (string "/dev/shm/frame_")
				(std--to_string i)
				(string ".ppm"))))
	     (cap.stopCapturing)
	     
	     )
	 ("const std::runtime_error&" (e)
	   ,(lprint :msg "error"
		    :vars `((e.what)))
	   (return 1)))
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil))

