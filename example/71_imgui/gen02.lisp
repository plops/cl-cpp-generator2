(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(let ((log-preamble `(do0 (include <iostream>
				   <iomanip>
				   <chrono>
				   <thread>
				   )
			  "extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;")))
  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file

    (defparameter *source-dir* #P"example/71_imgui/02source/")
    (load "util.lisp")
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `MainWindow
     :headers `()
     :header-preamble `(do0
			(include <vector>
;;;<GLFW/glfw3.h>
				 <functional>
				 <memory>
				 )
			"class GLFWwindow;"
			"class ImVec4;"
			"class ImGuiIO;"

			)
     :implementation-preamble `(do0
				,log-preamble
				(include "imgui_impl_opengl3_loader.h"

					 "imgui.h"
					 "imgui_impl_glfw.h"
					 "imgui_impl_opengl3.h"

					 <GLFW/glfw3.h>
					 )
				(include "implot.h")
				)
     :code `(do0
	     (defclass MainWindow ()
	       "public:"
	       "bool show_demo_window_;"
					;"ImVec4* clear_color_;"
	       "ImGuiIO& io;"
	       (defmethod MainWindow ()
		 (declare
		  (explicit)
		  (construct
		   (show_demo_window_ true)
		   (io ((lambda ()
			  (declare (values ImGuiIO&))
			  (do0 (IMGUI_CHECKVERSION)
			       (ImGui--CreateContext)
			       (ImPlot--CreateContext)
			       (return (ImGui--GetIO))))))
		   #+nil (clear_color_ (new (ImVec4 .45s0
						    .55s0
						    .6s0
						    1s0)))
		   )
		  (values :constructor))

		 ,(lprint)
		 )
	       (defmethod ~MainWindow ()
		 (declare
		  (values :constructor))
		 ,(lprint)
		 #+nil (do0
			(ImPlot--DestroyContext)
			(ImGui--DestroyContext)) )
	       (defmethod Init (window glsl_version )
		 (declare ;(type GLFWwindow* window)
		  (type "std::shared_ptr< GLFWwindow >" window)
		  (type "const char*" glsl_version)
		  )
		 ,(lprint)
		 (do0

		  (do0
		   (comments "enable keyboard controls, docking multi-viewport")
		   ,@(loop for e in `(NavEnableKeyboard
				      DockingEnable
					;ViewportsEnable
				      )
			   collect
			   `(do0
			     (setf io.ConfigFlags
				   (logior io.ConfigFlags
					   ,(format nil "ImGuiConfigFlags_~a" e)))
			     ,(lprint :msg (format nil "~a" e)
				      :vars `(,(format nil "ImGuiConfigFlags_~a" e)
					       io.ConfigFlags
					       (dot (ImGui--GetIO)
						    ConfigFlags)))))
		   )
		  (do0
		   ,(lprint :msg "setup ImGUI style")
		   (ImGui--StyleColorsDark)
					;(ImGui--StyleColorsClassic)
		   (let ((style (ImGui--GetStyle)))
		     #+nil (when
			       (logand
				io.ConfigFlags
				ImGuiConfigFlags_ViewportsEnable)
			     (setf style.WindowRounding 0s0
				   (dot style (aref Colors ImGuiCol_WindowBg)
					w)
				   1s0)))
		   (ImGui_ImplGlfw_InitForOpenGL (window.get) true)
		   (ImGui_ImplOpenGL3_Init glsl_version)
		   (let ((font_fn (string
				   ,(format nil "~a"
					    (elt
					     (directory "/home/martin/src/vcpkg/buildtrees/imgui/src/*/misc/fonts/DroidSans.ttf")
					     0))))
			 (font_size 16s0))
		     (let ((*font (-> io.Fonts (AddFontFromFileTTF font_fn font_size))))
		       (if (== font nullptr)
			   ,(lprint :msg "loading font failed" :vars `(font_fn font_size))
			   ,(lprint :msg "loaded font" :vars `(font_fn font_size)))))))
		 )
	       (defmethod NewFrame ()
		 (do0 (ImGui_ImplOpenGL3_NewFrame)
		      (ImGui_ImplGlfw_NewFrame)
		      (ImGui--NewFrame)
		      (ImGui--DockSpaceOverViewport)))
	       (defmethod Update (fun)
		 (declare (type "std::function<void(void)>" fun))
		 (do0
		  (when show_demo_window_
		    (ImGui--ShowDemoWindow &show_demo_window_)
		    (ImPlot--ShowDemoWindow)
		    )
		  (progn
		    (ImGui--Begin (string "hello"))
		    (ImGui--Checkbox (string "demo window")
				     &show_demo_window_)
		    (ImGui--Text (string "Application average %.3f ms/frame (%.1f FPS)")
				 (/ 1000s0 (dot (ImGui--GetIO) Framerate))
				 (dot (ImGui--GetIO) Framerate))
		    (ImGui--End))
		  (fun)
		  )
		 (ImGui--EndFrame)
		 )
	       (defmethod Render (window)
		 (declare ; (type GLFWwindow* window)
		  (type "std::shared_ptr< GLFWwindow >" window)
		  )
		 (do0

		  (let ((screen_width (int 0))
			(screen_height (int 0)))
		    (glfwGetFramebufferSize (window.get)
					    &screen_width
					    &screen_height)
					;,(lprint :msg "framebuffer" :vars `(screen_width screen_height))
		    (glViewport 0 0 screen_width screen_height)
		    #+nil(glClearColor (* clear_color_->x clear_color_->w)
				       (* clear_color_->y clear_color_->w)
				       (* clear_color_->z clear_color_->w)
				       clear_color_->w)
		    (glClear GL_COLOR_BUFFER_BIT)
		    (ImGui--Render)
		    (ImGui_ImplOpenGL3_RenderDrawData
		     (ImGui--GetDrawData)))
		  #+nil (do0
			 (comments "update and render additional platform windows")
			 (when
			     (logand
			      io.ConfigFlags
			      ImGuiConfigFlags_ViewportsEnable)
			   (let ((*backup_current_context
				  (glfwGetCurrentContext)))
			     (ImGui--UpdatePlatformWindows)
			     (ImGui--RenderPlatformWindowsDefault)
			     (glfwMakeContextCurrent backup_current_context)))))
		 (glfwSwapBuffers (window.get))
		 )
	       (defmethod Shutdown ()
		 (do0
		  ,(lprint :msg "destroy ImPlot Context")
		  (ImPlot--DestroyContext)
		  ,(lprint :msg "delete ImGui buffers and textures")
                  (ImGui_ImplOpenGL3_Shutdown)
		  ,(lprint :msg "delete ImGui callbacks and mouse cursor from GLFW")
		  (ImGui_ImplGlfw_Shutdown)
		  ,(lprint :msg "destroy ImGui Context")
		  (ImGui--DestroyContext))
		 ))))

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `ProcessFrameEvent
     :headers `()
     :header-preamble `(do0
			(include ;<memory>
			 <opencv2/core/core.hpp>
			 )
					;"namespace cv { class Mat; }"
			)
     :implementation-preamble `(do0
					;,log-preamble
					;(include (include <opencv2/core/core.hpp>))
				(include "ProcessFrameEvent.h")
				)
     :code (let ((def-members `((batch_idx int)
				(frame_idx int)
				(dim int)
				(fps float)
				(seconds double)
				(frame "cv::Mat")
				)))
	     `(do0
	       (defclass ProcessFrameEvent ()
		 "public:"
		 ,@(loop for (e f) in def-members
			 collect
			 (format nil "~a ~a;" f e))
					;"std::unique_ptr<cv::Mat> frame;"
		 (defmethod ProcessFrameEvent (,@(loop for (e f) in def-members
						       collect
						       (intern (string-upcase (format nil "~a_" e))))

					;  frame_
					       )
		   (declare
		    ,@(loop for (e f) in def-members
			    collect
			    `(type ,f ,(intern (string-upcase (format nil "~a_" e)))))

					;(type "std::unique_ptr<cv::Mat>" frame_)
		    (construct
		     ,@(loop for (e f) in def-members
			     collect
			     `(,e ,(format nil "~a_" e)))
					;(frame (std--move frame_))
		     )
		    (values :constructor))
		   )
		 #+nil (defmethod get_frame ()
			 (declare (values "std::unique_ptr<cv::Mat>"))
			 (return (std--move frame)))
		 ,@(loop for (e f) in def-members
			 collect
			 `(defmethod ,(format nil "get_~a" e) ()
			    (declare (values ,f))
			    (return ,e)))))))

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `ProcessedFrameMessage
     :headers `()
     :header-preamble `()
     :implementation-preamble `(do0
				(include "ProcessedFrameMessage.h")
				)
     :code (let ((def-members `((batch_idx int)
				(frame_idx int)
				(seconds double)
				;; i need to add the checkerboard corners herer
				)))
	     `(do0
	       (defclass ProcessedFrameMessage ()
		 "public:"
		 ,@(loop for (e f) in def-members
			 collect
			 (format nil "~a ~a;" f e))
		 (defmethod ProcessedFrameMessage (,@(loop for (e f) in def-members
							   collect
							   (intern (string-upcase (format nil "~a_" e))))
						   )
		   (declare
		    ,@(loop for (e f) in def-members
			    collect
			    `(type ,f ,(intern (string-upcase (format nil "~a_" e)))))
		    (construct
		     ,@(loop for (e f) in def-members
			     collect
			     `(,e ,(format nil "~a_" e))))
		    (values :constructor))
		   )
		 ,@(loop for (e f) in def-members
			 collect
			 `(defmethod ,(format nil "get_~a" e) ()
			    (declare (values ,f))
			    (return ,e)))))))

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `BoardProcessor
     :headers `()
     :header-preamble `(do0 (include <mutex>
				     "MessageQueue.h"
				     "ProcessFrameEvent.h"
				     "ProcessedFrameMessage.h"
				     <vector>
				     <future>))
     :implementation-preamble `(do0
				,log-preamble
				(include "ProcessedFrameMessage.h")
				(include <chrono>
					 <thread>
					;<opencv2/core/core.hpp>
					 <opencv2/imgproc/imgproc.hpp>
					 )
				)
     :code (let ((def-members `((run bool)
				(id int)

				(events "std::shared_ptr<MessageQueue<ProcessFrameEvent> >")
				(msgs "std::shared_ptr<MessageQueue<ProcessedFrameMessage> >")
				)))
	     `(do0
	       (defclass BoardProcessor ()
		 "public:"
		 ,@(loop for (e f) in def-members
			 collect
			 (format nil "~a ~a;" f e))
		 (defmethod BoardProcessor (,@(remove-if
					       #'null
					       (loop for (e f) in def-members
						     collect
						     (unless (eq e 'run)
						       (intern (string-upcase (format nil "~a_" e))))))
					    )
		   (declare
		    ,@(loop for (e f) in def-members
			    collect
			    `(type ,f ,(intern (string-upcase (format nil "~a_" e)))))
		    (construct
		     (run true)
		     ,@(remove-if #'null (loop for (e f) in def-members
					       collect
					       (unless (eq e 'run)
						 `(,e ,(format nil "~a_" e))))))
		    (values :constructor)))
		 ,@(loop for (e f) in def-members
			 collect
			 `(defmethod ,(format nil "get_~a" e) ()
			    (declare (values ,f))
			    (return ,e)))
		 (defmethod process ()
		   (while run
		     "using namespace std::chrono_literals;"
		     (std--this_thread--sleep_for 1ms)
		     (let ((event (events->receive)))
		       (processEvent event))
		     )
		   ,(lprint :msg "stopping BoardProcessor" :vars `(id)))
		 (defmethod processEvent (event)
		   (declare (type ProcessFrameEvent event))


		   (let ((dim (event.get_dim))
			 (frame (event.get_frame))
			 )
		     "cv::Mat gray;"
		     (cv--cvtColor frame
				   gray
				   cv--COLOR_RGB2GRAY)
		     (let ((msg (ProcessedFrameMessage (event.get_batch_idx)
						       (event.get_frame_idx)
						       (event.get_seconds)))
			   (sentCondition
			    (std--async
			     std--launch--async
			     "&MessageQueue<ProcessedFrameMessage>::send"
			     msgs
			     (std--move msg))))))
		   )
		 (defmethod stop ()
		   (setf run false))))))

    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"MessageQueue.h"
		       *source-dir*))
     `(do0
       (pragma once)
       (include <deque> <mutex> <condition_variable>)
       (defclass+ (MessageQueue :template "typename T") ()
	 "public:"
	 "std::mutex mutex_;"
	 "std::deque<T> queue_;"
	 "std::condition_variable condition_;"
	 (defmethod receive ()
	   (declare (values T))
	   (let (;(lock (std--unique_lock<std--mutex> mutex_))
		 )
	     "std::unique_lock<std::mutex> lock(mutex_);"
	     (dot condition_
		  (wait lock (lambda ()
			       (declare (capture this))
			       (return (not (dot queue_ (empty)))))))
	     (comments "remove last vector from queue")
	     (let ((msg (std--move (dot queue_ (back)))))
	       (queue_.pop_back)
	       (return msg))))
	 (defmethod send (msg)
	   (declare (type T&& msg))
	   (let (;(lock (std--lock_guard<std--mutex> mutex_))
		 )
	     "std::lock_guard<std::mutex> lock(mutex_);"
	     (dot queue_ (push_back (std--move msg)))
	     (dot condition_ (notify_one)))))))

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `GraphicsFramework
     :headers `()
     :header-preamble `(do0 (include ;"imgui_impl_opengl3_loader.h"
					;<GLFW/glfw3.h>
			     <memory>)
			    "class GLFWwindow;"
			    )
     :implementation-preamble `(do0
				(do0 (include "imgui_impl_opengl3_loader.h")
				     (include "imgui.h")
				     (include "imgui_impl_glfw.h")
				     (include "imgui_impl_opengl3.h")
				     (include




				      <GLFW/glfw3.h>)
				     ,log-preamble)
				)
     :code (let ((def-members `(#+nil (window "std::unique_ptr<GLFWwindow,DestroyGLFWwindow>"
					      )

				      )))
	     `(do0
	       #+nil  (defclass DestroyGLFWwindow ()
			(comments "https://gist.github.com/TheOpenDevProject/1662fa2bfd8ef087d94ad4ed27746120")
			"public:"
			(defmethod "operator()" (ptr)
			  (declare (type GLFWwindow* ptr))
			  ,(lprint :msg "Destroy GLFW window context.")
			  (glfwDestroyWindow ptr)
					;(glfwTerminate)
			  ))
	       (defclass GraphicsFramework ()
		 "public:"
					;"std::unique_ptr<GLFWwindow,DestroyGLFWwindow> window;"
		 "std::shared_ptr<GLFWwindow> window;"
		 ,@(loop for (e f) in def-members
			 collect
			 (format nil "~a ~a;" f e))
		 (defmethod GraphicsFramework (,@(remove-if
						  #'null
						  (loop for (e f) in def-members
							collect
							(unless (eq e 'run)
							  (intern (string-upcase (format nil "~a_" e))))))
					       )
		   (declare
		    ,@(loop for (e f) in def-members
			    collect
			    `(type ,f ,(intern (string-upcase (format nil "~a_" e)))))
		    (construct

		     ,@(remove-if #'null (loop for (e f) in def-members
					       collect
					       (unless (eq e 'run)
						 `(,e ,(format nil "~a_" e))))))
		    (values :constructor))

		   (do0
		    (glfwSetErrorCallback (lambda (err description)
					    (declare (type int err)
						     (type "const char*" description))
					    ,(lprint :msg "glfw error" :vars `(err description))))
		    (comments "glfw initialization")
		    (comments "https://github.com/ocornut/imgui/blob/docking/examples/example_glfw_opengl3/main.cpp")

		    (unless (glfwInit)
		      ,(lprint :msg "glfwInit failed."))

		    (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 3)
		    (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0)
					;"std::unique_ptr<GLFWwindow,DestroyGLFWwindow> window;"
		    (let ((w (glfwCreateWindow 1800 1000
					       (string "dear imgui example")
					       nullptr nullptr)))
		      (when (== nullptr w)
			,(lprint :msg "glfwCreatWindow failed."))
		      (dot window (reset w (lambda (ptr)
					     (declare (type GLFWwindow* ptr))
					     (glfwDestroyWindow ptr))))
		      (glfwMakeContextCurrent (window.get))
		      ,(lprint :msg "enable vsync")
		      (glfwSwapInterval 1))
		    (comments "imgui brings its own opengl loader"
			      "https://github.com/ocornut/imgui/issues/4445"))

		   )
		 (defmethod ~GraphicsFramework ()
		   (declare (values :constructor))
		   (do0 (do0 ,(lprint :msg "destroy window")
			     (glfwDestroyWindow (window.get)))
			(do0
			 ,(lprint :msg "disable GLFW error callback")
			 (glfwSetErrorCallback nullptr))
			(do0 ,(lprint :msg "terminate GLFW")
			     (glfwTerminate)))
		   )

		 ,@(loop for (e f) in def-members
			 collect
			 `(defmethod ,(format nil "get_~a" e) ()
			    (declare (values ,f))
			    (return ,e)))

		 (defmethod WindowShouldClose ()
		   (declare (values bool))
		   (return (glfwWindowShouldClose (window.get))))
		 (defmethod PollEvents ()
		   (glfwPollEvents))
		 #+nil (defmethod Shutdown ()
			 ,(lprint :msg "destroy window and terminate GLFW")
					;(glfwDestroyWindow (window.get))
					;(glfwTerminate)
			 )
		 (defmethod getWindow ()
		   (declare (values ;GLFWwindow*
			     "std::shared_ptr<GLFWwindow>"))
		   #+nil (comments "FIXME: how to handle this pointer as a smart pointer?"
				   "https://stackoverflow.com/questions/6012157/is-stdunique-ptrt-required-to-know-the-full-definition-of-t"
				   "maybe use this: https://github.com/janekb04/glfwpp/")
		   (return ;( window.get)
		     window))
		 ))))

    (let* ((def-tex `((:width img.cols :height img.rows :data img.data)
		      (:width board_img.cols :height board_img.rows :data board_img.data)))
	   (num-tex (length def-tex)))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name `Charuco
       :headers `()
       :header-preamble `(do0 ,@(loop for e in `(core videoio imgproc aruco/charuco)
				      collect
				      `(include ,(format nil "<opencv2/~a.hpp>" e)))
			      #+nil (do0 (include
					  <opencv2/core/cvstd.hpp>
					  <opencv2/core/mat.hpp>)
					 ,(format nil "namespace cv { ~{class ~a;~} }" `(Dictionary CharucoBoard DetectorParameters)))
			      )
       :implementation-preamble `(do0
				  ,log-preamble
				  (do0 (include "imgui_impl_opengl3_loader.h")
				       (include "imgui.h")
				       (include "imgui_impl_glfw.h")
				       (include "imgui_impl_opengl3.h")
				       (include  <GLFW/glfw3.h>)
				       ))
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
				  (:name img :type "cv::Mat" :no-construct t)
				  (:name img3 :type "cv::Mat" :no-construct t)
				  (:name textures :type "std::vector<uint32_t>" ;"GLuint[2]"
					 :init-form (curly 0 0))
				  (:name camera_matrix :type "cv::Mat" :no-construct t)
				  (:name dist_coeffs :type "cv::Mat" :no-construct t)
				  (:name cap_fn :type "std::string" :default (string "/dev/video2") )
				  (:name cap :type "cv::VideoCapture" :init-form (cv--VideoCapture cap_fn))
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
		      (comments "opencv initialization")

		      (board->draw (cv--Size 1600 800)
				   board_img3
				   10 1
				   )
		      (cv--cvtColor board_img3 board_img cv--COLOR_BGR2RGBA)

		      (if (cap.isOpened)
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
		   (defmethod Init ()
		     (comments "this function requires Capture() to have been called at least once, so that we know the image data.")
		     (glGenTextures ,num-tex (textures.data))
		     ,@(loop for e in def-tex
			     and e-i from 0
			     collect
			     (destructuring-bind (&key width height data) e
			       `(progn
					;,(lprint :msg (format nil "prepare texture ~a" data))
				  (let ((texture (aref textures ,e-i)))
				    (glBindTexture GL_TEXTURE_2D texture)
				    ,@(loop for e in `(MIN MAG) collect
					    `(glTexParameteri GL_TEXTURE_2D
							      ,(format nil "GL_TEXTURE_~a_FILTER" e)
							      GL_LINEAR))
				    (glPixelStorei GL_UNPACK_ROW_LENGTH
						   0)
				    (glTexImage2D GL_TEXTURE_2D ;; target
						  0 ;; level
						  GL_RGBA ;; internalformat
						  ,width ;; width
						  ,height ;; height
						  0 ;; border
						  GL_RGBA ;; format
						  GL_UNSIGNED_BYTE ;; type
						  ,data ;; data pointer
						  )))))
					;,(lprint :msg "uploaded texture")
		     )
		   (defmethod Render ()
		     (declare (capture &textures &img &board_img))

		     ,@(loop for e in def-tex
			     and e-i from 0
			     collect
			     (destructuring-bind (&key width height data) e
			       `(do0
				 (ImGui--Begin (string ,(format nil "~a" data)))
				 (glBindTexture GL_TEXTURE_2D (aref textures ,e-i))
				 (glTexImage2D GL_TEXTURE_2D ;; target
					       0	     ;; level
					       GL_RGBA ;; internalformat
					       ,width  ;; width
					       ,height ;; height
					       0       ;; border
					       GL_RGBA ;; format
					       GL_UNSIGNED_BYTE ;; type
					       ,data ;; data pointer
					       )

				 (ImGui--Image (reinterpret_cast<void*> (aref textures ,e-i))
					       (ImVec2 ,width ,height))
				 (ImGui--End))))
		     )
		   (defmethod Capture ()
		     (declare (values "cv::Mat"))
		     (do0; "cv::Mat img3,img;"
		      (>> cap img3)
					;(cv--split img spl)
					;,(lprint :msg "received camera image")
		      (cv--cvtColor img3 img cv--COLOR_BGR2RGBA)
					;,(lprint :msg "converted camera image")
		      )
		     (return img))
		   ,@(remove-if
		      #'null
		      (loop for e in def-members
			    collect
			    (destructuring-bind (&key name type init-form default no-construct) e
			      `(defmethod ,(format nil "get_~a" name) ()
				 (declare (values ,type))
				 (return ,name)))))
		   (defmethod Shutdown ()
		     ,(lprint :msg "delete textures")
		     (glDeleteTextures 2 (textures.data) )
		     #+nil
		     (do0 ,(lprint :msg "release video capture device (should be done implicitly)")
			  (cap.release))
		     )

		   )))))

    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"main.cpp"
				    *source-dir*))
		  `(do0
		    (do0 (include "imgui_impl_opengl3_loader.h")
			 (include "imgui.h")
			 (include "imgui_impl_glfw.h")
			 (include "imgui_impl_opengl3.h")
			 (include  <GLFW/glfw3.h>)
			 )

		    (include "MainWindow.h")
		    (include
					;<tuple>
					;<mutex>
		     <thread>
		     <iostream>
		     <iomanip>
		     <chrono>
		     <cmath>
		     <cassert>
					;  <memory>
		     )
		    (include "Charuco.h")
		    (include <opencv2/core.hpp>
			     <opencv2/videoio.hpp>
			     <opencv2/imgproc.hpp>
			     <opencv2/aruco/charuco.hpp>)
		    "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"

		    (include "implot.h")
		    (include "GraphicsFramework.h")



		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (setf g_start_time ("std::chrono::high_resolution_clock::now"))
		      (progn
			,(lprint :msg "start" :vars `(argc (aref argv 0)))
			(let ((framework (GraphicsFramework)))
					;(framework.Init)
			  (let (
				(charuco (Charuco)))
			    (do0





			     (do0
			      "MainWindow M;"
			      (M.Init (framework.getWindow) (string "#version 130"))
			      (charuco.Capture)
			      (charuco.Init)

			      (while (!framework.WindowShouldClose)
				(do0
				 (charuco.Capture)
				 #+nil (do0
					(>> cap img3)
					(do0
					 (comments "detect charuco board")
					 "std::vector<int> markerIds;"
					 "std::vector<std::vector<cv::Point2f> > markerCorners;"
					 (cv--aruco--detectMarkers img3 board->dictionary markerCorners markerIds params)
					 (when (< 0 (markerIds.size))
					;(cv--aruco--drawDetectedMarkers img3 markerCorners markerIds)
					   "std::vector<cv::Point2f> charucoCorners;"
					   "std::vector<int> charucoIds;"
					   (cv--aruco--interpolateCornersCharuco markerCorners
										 markerIds
										 img3
										 board
										 charucoCorners
										 charucoIds
										 cameraMatrix
										 distCoeffs)
					   (when (< 0 (charucoIds.size))
					     (let ((color (cv--Scalar 255 0 255)))
					       (cv--aruco--drawDetectedCornersCharuco img3 charucoCorners charucoIds color)
					       #+nil (do0 "cv::Vec3d rvec, tvec;"
							  (let ((valid (cv--aruco--estimatePoseCharucoBoard
									charucoCorners
									charucoIds
									board cameraMatrix distCoeffs
									rvec tvec)))
							    (when valid
							      (cv--aruco--drawAxis img3 cameraMatrix distCoeffs rvec tvec .1s0))))))
					   ))
					(cv--cvtColor img3 img cv--COLOR_BGR2RGBA)
					)
				 (framework.PollEvents)
				 (M.NewFrame)

				 (M.Update
				  (lambda () (declare (capture &charuco)) (charuco.Render))
				  )

				 (M.Render (framework.getWindow))
				 ))

			      )
			     (do0
			      ,(lprint :msg "run various cleanup functions")
			      (charuco.Shutdown)

			      (M.Shutdown)
					;(framework.Shutdown)


			      )

			     )

			    ,(lprint :msg "leave program")

			    (return 0)))))))

    (with-open-file (s "02source/CMakeLists.txt" :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      ;;https://clang.llvm.org/docs/AddressSanitizer.html
      ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
      ;;
      (let ((dbg "-ggdb -O0 ")
	    (asan "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope")
	    (show-err " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow  -Wswitch-default -Wundef -Werror -Wno-unused"
	      ;; -Wold-style-cast -Wsign-conversion
					;"-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "

	      ))
	(macrolet ((out (fmt &rest rest)
		     `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	  (out "cmake_minimum_required( VERSION 3.4 )")
	  (out "project( mytest LANGUAGES CXX )")
	  (out "set( CMAKE_CXX_COMPILER clang++ )")
	  (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	  (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a \")" dbg asan show-err)
	  (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 20 )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")

	  (out "set( SRCS ~{~a~^~%~} )"
	       (append
		(directory "02source/*.cpp")
					;(directory "/home/martin/src/vcpkg/buildtrees/implot/src/*/implot_demo.cpp")
		(directory "/home/martin/src/vcpkg/buildtrees/imgui/src/*/backends/imgui_impl_opengl3.cpp")
		))

	  (out "add_executable( mytest ${SRCS} )")
	  (out "target_compile_features( mytest PUBLIC cxx_std_17 )")

	  (loop for e in `(imgui implot)
		do
		(out "find_package( ~a CONFIG REQUIRED )" e))
	  ;; module definitions /usr/lib64/cmake/OpenCV/OpenCVModules.cmake
	  (out "find_package( OpenCV REQUIRED core videoio imgproc aruco )")
	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `("imgui::imgui"
		 "implot::implot"
		 "${OpenCV_LIBS}"))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))

