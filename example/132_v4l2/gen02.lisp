(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(setf *features* (set-difference *features* (list :more)))
(setf *features* (set-exclusive-or *features* (list :more)))

(progn
  (progn
    (defparameter *source-dir* #P"example/132_v4l2/source02/src/")
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
	       #+more (string ,request))))

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
		       (include<> functional)
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
						 #+more (std--string (std--strerror errno)))))))

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

		 (let ((str (v4l2_streamparm (designated-initializer :type V4L2_BUF_TYPE_VIDEO_CAPTURE))))
		   ,(xioctl `(:request g-parm :var &str))
		   (setf (dot str parm capture timeperframe numerator) 1)
		   (setf (dot str parm capture timeperframe denominator  ) 10)
		   ,(xioctl `(:request s-parm :var &str)))
		 
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

	       (defmethod getFrameAndStore (filename)
		 (declare (type "const std::string&" filename))
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

	       (defmethod getFrame (fun)
		 (declare (type "std::function<void(void*,size_t)>" fun))
		 (let ((buf (v4l2_buffer (designated-initializer :type V4L2_BUF_TYPE_VIDEO_CAPTURE
								 :memory V4L2_MEMORY_MMAP))))
					
		   ,(xioctl `(:request dqbuf :var &buf)))
		 (let ((b  (aref buffers_ buf.index)))
		   (fun b.start b.length))
		 ,(xioctl `(:request qbuf :var &buf)))
	       	       
	       "private:"
	       (defstruct0 buffer
		   (start void*)
		 (length size_t))

	       (defmethod xioctl (request arg #+more str)
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
		     (throw (std--runtime_error  (+ (string "ioctl ")
						    #+more str
						   #+more (string " ")
						   #+more (std--strerror errno)
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
      thread
      
      filesystem
      unistd.h
      cstdlib

      cmath
      linux/videodev2.h

      )

     (include<>
      glad/gl.h)
     
     (include<>

      GLFW/glfw3.h
      
      glm/glm.hpp
      imgui.h
      imgui_impl_glfw.h
      imgui_impl_opengl3.h
      
      
      )
     (comments "wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h")
     
     (include 
      stb_image.h
      V4L2Capture.h)

     (setf "const char *vertexShaderSrc"
	   (string-r
	    ,(emit-c :code `(do0
			     "#version 450"
			     (space layout (paren (= location 0)) in vec2 aPos)
			     (defun main ()
			       (setf gl_Position (vec4 aPos 1 1))))
		     :omit-redundant-parentheses t)))

     (setf "const char *fragmentShaderSrc"
	   (string-r
	    ,(emit-c :code `(do0
			     "#version 450"
			     (space layout (paren (= location 0)) out vec4 outColor)
			     (defun main ()
			       (setf outColor (vec4 1 0 0 1))))
		     :omit-redundant-parentheses t)))

     #+more (defun message_callback (source type id severity length message user_param)
       (declare (type GLenum source type severity)
		(type GLuint id)
		(type GLsizei length)
		(type "GLchar const *" message)
		(type "void const *" user_param))
       ,(lprint :msg "gl"
		:vars `(source type id severity message)))
					     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))

       (do0
	(glfwInit)
	(glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 4)
	(glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 5)
	(glfwWindowHint GLFW_OPENGL_PROFILE GLFW_OPENGL_CORE_PROFILE)

	(let ((window (glfwCreateWindow 800 600
					(string "v4l")
					nullptr nullptr)))
	  (unless window
	    (throw (std--runtime_error (string "Error creating glfw window")))))

	(glfwMakeContextCurrent window)
	(glfwSwapInterval 1)

	(unless (gladLoaderLoadGL)
	  (throw (std--runtime_error (string "Error initializing glad")))
	  )

	(do0
	 (IMGUI_CHECKVERSION)
	 (ImGui--CreateContext)
	 (ImGui_ImplGlfw_InitForOpenGL window true)
	 (ImGui_ImplOpenGL3_Init (string "#version 450 core"))
	 (ImGui--StyleColorsClassic))

	(do0
	 (glEnable GL_CULL_FACE)
	 #+more (glEnable GL_DEBUG_OUTPUT)
	 #+more (glDebugMessageCallback message_callback nullptr))

	(do0
	 #+more ,(lprint :msg "Compile shader")
	 (let ((success 0)
	       
	       (vertexShader (glCreateShader GL_VERTEX_SHADER)))
	   (glShaderSource vertexShader 1 &vertexShaderSrc 0)
	   (glCompileShader vertexShader)

	   (glGetShaderiv vertexShader GL_COMPILE_STATUS &success)
	   (unless success
	     #+more (let ((n 512) (infoLog (std--vector<char> n)))
	       (glGetShaderInfoLog vertexShader n nullptr (infoLog.data) )
	       ,(lprint :msg "vertex shader compilation failed"
			:vars `((std--string (infoLog.begin)
					     (infoLog.end))))
	       )
	     (exit -1))
	   )
	 (let ((fragmentShader (glCreateShader GL_FRAGMENT_SHADER)))
	   (glShaderSource fragmentShader 1 &fragmentShaderSrc 0)
	   (glCompileShader fragmentShader)

	   (glGetShaderiv fragmentShader GL_COMPILE_STATUS &success)
	   (unless success
	     #+more (let ((n 512) (infoLog (std--vector<char> n)))
	       (glGetShaderInfoLog fragmentShader n nullptr (infoLog.data) )
	       ,(lprint :msg "fragment shader compilation failed"
			:vars `((std--string (infoLog.begin)
					     (infoLog.end))))
	       )
	     (exit -1))
	   )
	 (let ((program (glCreateProgram )))
	   (glAttachShader program vertexShader)
	   (glAttachShader program fragmentShader)
	   (glLinkProgram program)
	   (glGetProgramiv program GL_LINK_STATUS &success)
	   (unless success
	     #+more (let ((n 512) (infoLog (std--vector<char> n)))
	       (glGetShaderInfoLog program n nullptr (infoLog.data) )
	       ,(lprint :msg "shader linking failed"
			:vars `((std--string (infoLog.begin)
					     (infoLog.end))))
	       )
	     (exit -1))
	   )
	 (glDetachShader program vertexShader)
	 (glDetachShader program fragmentShader))
	
	
	)

       (do0
	#+more
	(do0 ,(lprint :msg "Create vertex array and buffers")
	     (comments "TBD")) 

	(glUseProgram program)
	(glClearColor 1 1 1 1)

	
	)

       
       
       (handler-case
	   (let ((cap (V4L2Capture  (string "/dev/video0")
				    3)))
	     (let ((w 320 ;1280
		      )
		   (h 180 ; 720
		      ))
	      (cap.setupFormat w h
			       V4L2_PIX_FMT_RGB24
					;V4L2_PIX_FMT_YUYV
			       ;V4L2_PIX_FMT_YUV420
			       ))
	     (cap.startCapturing)

	     (do0
	      (let ((texture (GLuint 0)))
		(glGenTextures 1 &texture)
		(glBindTexture GL_TEXTURE_2D texture)
		
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR)))
	     
	     (while (!glfwWindowShouldClose window)
		    (glfwPollEvents)
					;(glDrawElements GL_TRIANGLES 6 GL_UNSIGNED_INT nullptr)
		    (ImGui_ImplOpenGL3_NewFrame)
		    (ImGui_ImplGlfw_NewFrame)
		    (ImGui--NewFrame)
		    (cap.getFrame
		     (lambda (data size)
		       (declare (type void* data)
				(type size_t size)
				(capture "&"))
		       (glTexImage2D GL_TEXTURE_2D
				     0
				     GL_RGB
				     w h
				     0
				     GL_RGB
				     GL_UNSIGNED_BYTE
				     data
				     )
		       (ImGui--Begin (string "camera feed"))
		       (ImGui--Image (reinterpret_cast<void*> (static_cast<intptr_t> texture))
				     (ImVec2 w h))
		       (ImGui--End)
		  
		       ))

		    (space static bool (setf showDemo false))
		    (ImGui--ShowDemoWindow &showDemo)
		    (ImGui--Render)
		    (ImGui_ImplOpenGL3_RenderDrawData (ImGui--GetDrawData))
		    (glfwSwapBuffers window)
		    (glClear GL_COLOR_BUFFER_BIT)
		    (std--this_thread--sleep_for (std--chrono--milliseconds 16))
		    )
	     
	     
	     #+nil (cap.getFrame (+ (string "/dev/shm/frame_")
			      (std--to_string i)
			      (string ".ppm")))
	     (cap.stopCapturing)
	     
	     )
	 ("const std::runtime_error&" (e)
	   #+more ,(lprint :msg "error"
		    :vars `((e.what)))
	   (return 1)))

       (do0
	(ImGui_ImplOpenGL3_Shutdown)
	(ImGui_ImplGlfw_Shutdown)
	(ImGui--DestroyContext)
	(glfwDestroyWindow window)
	(glfwTerminate))
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil))

