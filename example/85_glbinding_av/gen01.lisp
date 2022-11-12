(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(let ((log-preamble `(do0 (include <iostream>
					;<iomanip>
				   <chrono>
				   <thread>
				   )
			  "void lprint(std::initializer_list<std::string> il, std::string file, int line, std::string fun);"
			  "extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;")))

  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file
    (defparameter *source-dir* #P"example/85_glbinding_av/source01/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))

    (ensure-directories-exist *full-source-dir*)
    (load "util.lisp")

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `GlfwWindow
     :headers `()
     :header-preamble `(do0
			#+nil (include <vector>
				       <functional>
				       <memory>
				       )
			"class GLFWwindow;"
			"typedef void (*GLFWglproc)(void);"
			)
     :implementation-preamble `(do0
				,log-preamble
				(include <imgui.h>
					 <backends/imgui_impl_glfw.h>
					 <backends/imgui_impl_opengl3.h>)
				(do0 "#define GLFW_INCLUDE_NONE"
				     (include <GLFW/glfw3.h>))

				)
     :code `(do0
	     (defclass GlfwWindow ()
	       "GLFWwindow* m_window = nullptr;"
	       "public:"
	       (defmethod GlfwWindow ()
		 (declare
		  (explicit)
		  (construct
		   )
		  (values :constructor))
		 (do0

		  ,(lprint :msg "initialize GLFW3")
		  (unless (glfwInit)
		    ,(lprint :msg "glfwInit failed"))
		  (glfwWindowHint GLFW_VISIBLE true)
		  (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 3)
		  (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 2)
		  (glfwWindowHint GLFW_OPENGL_FORWARD_COMPAT true)
		  (glfwWindowHint GLFW_OPENGL_PROFILE
				  GLFW_OPENGL_CORE_PROFILE)

		  ,(lprint :msg "create GLFW3 window")
		  (let ((startWidth 800)
			(startHeight 600)
			(window (glfwCreateWindow startWidth startHeight
						  (string "glfw")
						  nullptr
						  nullptr))
			)
		    (declare (type "const auto" startWidth startHeight))
		    (unless window
		      ,(lprint :msg "can't create glfw window"))
		    ,(lprint :msg "initialize GLFW3 context for window")
		    (glfwMakeContextCurrent window)
		    (do0 (comments "configure Vsync, 1 locks to 60Hz, FIXME: i should really check glfw errors")
			 (glfwSwapInterval 0))
		    (setf m_window window))))
	       (defmethod ~GlfwWindow ()
		 (declare
		  (values :constructor))
		 ,(lprint)
		 (do0 (glfwDestroyWindow m_window)
		      (glfwTerminate)))
	       (defmethod WindowShouldClose ()
		 (declare (values bool))
		 (return (glfwWindowShouldClose m_window)))
	       (defmethod SwapBuffers ()
		 (glfwSwapBuffers m_window))
	       (defmethod GetWindow ()
		 (declare (values GLFWwindow*))
		 (return m_window))
	       (defmethod PollEvents ()
		 (glfwPollEvents))
	       (defmethod GetProcAddress (name)
		 (declare (type "const char*" name)
			  (static)
			  (values GLFWglproc))
		 ,(lprint :svars `(name))
		 (return (glfwGetProcAddress name)))
	       (defmethod GetWindowSize ()
		 (declare (values "std::pair<int,int>")
			  (const))
		 (let ((width (int 0))
		       (height (int 0)))
		   (glfwGetWindowSize m_window &width &height)
		   (return ("std::make_pair" width height))))
	       )))


    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `ImguiHandler
     :headers `()
     :header-preamble `(do0
			"class GLFWwindow;"
			)
     :implementation-preamble `(do0
				,log-preamble
				(include <imgui.h>
					 <backends/imgui_impl_glfw.h>
					 <backends/imgui_impl_opengl3.h>)
				(do0 "#define GLFW_INCLUDE_NONE"
				     (include <GLFW/glfw3.h>))

				)
     :code `(do0
	     (defclass ImguiHandler ()
	       "public:"
	       (defmethod ImguiHandler (window)
		 (declare
		  (type GLFWwindow* window)
		  (explicit)
		  (construct
		   )
		  (values :constructor))
		 (do0
		  ,(lprint :msg "initialize ImGui")
		  (IMGUI_CHECKVERSION)
		  (ImGui--CreateContext)
		  (let ((io (ImGui--GetIO)))
		    (setf io.ConfigFlags
			  (logior
			   io.ConfigFlags
			   ImGuiConfigFlags_NavEnableKeyboard)))
		  (ImGui--StyleColorsLight)
		  (progn
		    (let ((installCallbacks true))
		      (declare (type "const auto" installCallbacks))
		      (ImGui_ImplGlfw_InitForOpenGL window
						    installCallbacks)))

		  (let ((glslVersion (string "#version 150")))
		    (declare (type "const auto" glslVersion))
		    (ImGui_ImplOpenGL3_Init glslVersion))
		  (when (<= io.DeltaTime 0s0)
		    (setf io.DeltaTime 1s-6)))
		 )
	       (defmethod NewFrame ()
		 (do0
		  (ImGui_ImplOpenGL3_NewFrame)
		  (ImGui_ImplGlfw_NewFrame)


		  (ImGui--NewFrame)
		  (let ((showDemoWindow true))
		    (ImGui--ShowDemoWindow &showDemoWindow)))


		 )
	       (defmethod Render ()
		 (ImGui--Render))
	       (defmethod RenderDrawData ()
		 (ImGui_ImplOpenGL3_RenderDrawData
		  (ImGui--GetDrawData)))
	       (defmethod Begin (str)
		 (declare (type "const char*" str))
		 (ImGui--Begin str))
	       (defmethod End ()
		 (ImGui--End))
	       (defmethod Image (tex w h)
		 (declare (type int w h)
			  (type uint tex))
		 (ImGui--Image (reinterpret_cast<void*>
				(static_cast<intptr_t> tex))
			       (ImVec2 (static_cast<float> w)
				       (static_cast<float> h)))
		 )
	       (defmethod SliderFloat (label val min max fmt)
		 (declare (type "const char*" label fmt)
			  (type float* val)
			  (type float min max))

		 (ImGui--SliderFloat label
				     val
				     min max fmt
				     ))
	       (defmethod ~ImguiHandler ()
		 (declare
		  (values :constructor))

		 (do0
		  ,(lprint :msg "Shutdown ImGui")
		  (ImGui_ImplOpenGL3_Shutdown)
		  (ImGui_ImplGlfw_Shutdown)
		  (ImGui--DestroyContext)
		  ))

	       )))

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `Video
     :headers `()
     :header-preamble `(do0
			"class GLFWwindow;"
					;"namespace av { class FormatContext; class Stream; class Codec; class VideoDecoderContex; class Packet; class VideoFrame;}"

			(include <avcpp/av.h>
 				 <avcpp/ffmpeg.h>
				 )
			(include ;<avcpp/format.h>
			 <avcpp/formatcontext.h>
			 <avcpp/codec.h>
			 <avcpp/codeccontext.h>)
			)
     :implementation-preamble `(do0
				,log-preamble
				(do0
				 (include ;<avcpp/av.h>
 				  <avcpp/ffmpeg.h>)
				 ;; API2
				 (include ;<avcpp/format.h>
				  <avcpp/formatcontext.h>
				  <avcpp/codec.h>
				  <avcpp/codeccontext.h>))

				)
     :code `(do0
	     (defclass Video ()
	       "std::string fn;"
	       "av::FormatContext ctx;"

	       "av::Stream vst;"
	       "av::Codec codec;"
	       "std::error_code ec;"

	       "av::VideoDecoderContext vdec;"

	       "av::Packet pkt;"

	       "public:"
	       "ssize_t videoStream = -2;"
	       (defmethod Video (filename)
		 (declare
		  (type "std::string" filename)
		  (explicit)
		  (construct
		   (ctx (av--FormatContext))
		   )
		  (values :constructor))
		 (do0
		  (setf fn filename)

					;(av--setFFmpegLoggingLevel AV_LOG_DEBUG)
					;(setf ctx (av--FormatContext))
		  (let ()
		    (let ()

		      ,(lprint :msg "open video file"
			       :svars `(fn))
		      (ctx.openInput fn))
		    (ctx.findStreamInfo)
		    ,(lprint :msg "stream info"
			     :vars `((ctx.seekable)
				     (dot ctx (startTime) (seconds))
				     (dot ctx (duration) (seconds))
				     (ctx.streamsCount)))
		    (ctx.seek (curly ("static_cast<long int>" (floor (* 100 (* .5 (dot ctx (duration)
										       (seconds))))))
				     (curly 1 100)))
		    (do0
		     (
		      for ((= "size_t i" 0)
			   (< i (ctx.streamsCount))
			   "i++")
					;dotimes (i (ctx.streamsCount))
		      (let ((st (ctx.stream i)))
			(when (== AVMEDIA_TYPE_VIDEO
				  (st.mediaType))
			  (setf videoStream i
				vst st)
			  break)))
		     (when (vst.isNull)
		       ,(lprint :msg "Video stream not found"))
		     (do0

		      (when (vst.isValid)
			(setf vdec (av--VideoDecoderContext vst))
			(let (;(codec )
			      )
			  (setf codec (av--findDecodingCodec (-> (vdec.raw)
								 codec_id)))
			  (vdec.setCodec codec)
			  (vdec.setRefCountedFrames true)
			  (vdec.open (curly
				      (curly (string "threads")
					     (string "1")))
				     (av--Codec)
				     ec)
			  (when ec
			    ,(lprint :msg "can't open codec"))


			  ))))))
		 )
	       (defmethod readPacket ()
		 (declare (values "av::Packet"))
		 (setf pkt (ctx.readPacket ec)
		       )
		 (when ec
		   ,(lprint :msg "packet reading error"
			    :svars `((ec.message))))
		 (return  pkt))

	       (defmethod decode ()
		 (declare (values "av::VideoFrame"))
		 (let ((frame (vdec.decode pkt ec)))
		   (when ec
		     ,(lprint :msg "error"
			      :svars `((ec.message))))
		   (return frame))
		 )
	       (defmethod seek (val)
		 (declare (type float val))
		 (ctx.seek (curly ("static_cast<long int>" (floor (* 1000 val)))
				  (curly 1 1000))))
	       (defmethod startTime ()
		 (declare (values float))
		 (return (static_cast<float> (dot ctx (startTime) (seconds))) ))
	       (defmethod duration ()
		 (declare (values float))
		 (return
		   (static_cast<float> (dot ctx (duration) (seconds)))))
	       (defmethod ~Video ()
		 (declare
		  (values :constructor))
		 )

	       )))

    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"main.cpp"
		       *source-dir*))
     `(do0


       (include
					;<tuple>
					;<mutex>
	<thread>
	<iostream>
	<iomanip>
	<chrono>
	<cassert>
					;  <memory>
	)

       (do0
	(include <glbinding/gl32core/gl.h>
		 <glbinding/glbinding.h>
					; <glbinding/CallbackMask.h>
					;<glbinding/FunctionCall.h>
					;<glbinding/AbstractFunction.h>
		 )
	"using namespace gl32core;"
	"using namespace glbinding;")
       (do0
	#+nil (include <imgui.h>
		       <backends/imgui_impl_glfw.h>
		       <backends/imgui_impl_opengl3.h>)
	#+nil (do0 "#define GLFW_INCLUDE_NONE"
		   (include <GLFW/glfw3.h>
			    ))
	(include "GlfwWindow.h"
		 "ImguiHandler.h"
		 "Video.h")

	(do0
	 (include <avcpp/av.h>
					;	  <avcpp/ffmpeg.h>
		  )
	 ;; API2
	 (include ;<avcpp/format.h>
	  <avcpp/formatcontext.h>
					;<avcpp/codec.h>
					;<avcpp/codeccontext.h>
	  ))
	(include <cxxopts.hpp>)

	)

       "const std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time = std::chrono::high_resolution_clock::now();"


       ,(init-lprint)
       (defun main (argc argv)
	 (declare (type int argc)
		  (type char** argv)
		  (values int))
	 ,(lprint :msg "start" :vars `(argc))

	 (let ((options (cxxopts--Options
			 (string "gl-video-viewer")
			 (string "play videos with opengl"))))

	   (let ((positional (std--vector<std--string>)))
	     ((((dot
		 options
		 (add_options))
		(string "h,help")
		(string "Print usage"))
	       (string "i,internal-tex-format")
	       (string "data format of texture")
	       (-> (cxxopts--value<int>)
		   (default_value (string "3"))))
	      (string "filenames")
	      (string "The filenames of videos to display")
	      (cxxopts--value<std--vector<std--string>>
	       positional))



	     (options.parse_positional  (curly (string "filenames"))))
	   (let ((opt_res (options.parse argc argv)))
	     (when (opt_res.count (string "help"))
	       (<< std--cout
		   (options.help)
		   std--endl)
	       (exit 0))

	     ,(let ((tex-formats `((GL_RGBA)
				   (GLenum--GL_RGB8)
				   (GLenum--GL_R3_G3_B2)
				   (GLenum--GL_RGBA2 :default t :comment "weirdest effect")
				   (GLenum--GL_RGB9_E5 :comment "shared exponent, looks ok")
				   (GLenum--GL_SRGB8 :comment "this displays as much darker")
				   (GLenum--GL_RGB8UI :comment "displays as black")
				   (GLenum--GL_COMPRESSED_RGB :comment "framerate drops from +200fps to 40fps"))))
		`(let ((texFormatIdx (dot (aref opt_res (string "internal-tex-format"))
					  (as<int>))))
 		   (assert (<= 0 texFormatIdx))
		   (assert (< texFormatIdx ,(length tex-formats)))
		   (let ((texFormats (,(format nil "std::array<gl::GLenum,~a>" (length tex-formats))
				       (curly ,@(loop for e in tex-formats
						      collect
						      (destructuring-bind ( val &key comment default) e
							val)))))
			 (texFormat (aref texFormats texFormatIdx))))))
	     ))

	 (let ((win (GlfwWindow)))
	   (do0
	    ,(lprint :msg "initialize glbinding")
	    (comments "if second arg is false: lazy function pointer loading")
	    (glbinding--initialize win.GetProcAddress
				   false)
	    #+nil
	    (do0 (glbinding--setCallbackMask
		  (logior
		   CallbackMask--After
		   CallbackMask--ParametersAndReturnValue))
		 (glbinding--setAfterCallback
		  (lambda (call)
		    (declare (type "const glbinding::FunctionCall&"
				   call)
			     )
		    (let ((fun (dot call  (-> function (name)))))
		      ,(lprint :msg `fun)))))
	    ,(let ((l `(.4s0 .4s0 .2s0 1s0)))
	       `(progn
		  ,@(loop for e in l
			  and n in `(r g b a)
			  collect
			  `(let ((,n ,e))
			     (declare (type "const float" ,n))))
		  (glClearColor r g b a))))

	   (let ((imgui (ImguiHandler (win.GetWindow)))))

	   (do0
	    (av--init)
	    (let ((video (Video (dot positional (at 0)))))))

	   (let (;(radius 10s0)
		 )
	     (declare (type "const auto" radius))

	     (do0 "bool video_is_initialized_p = false;"
		  "int image_width = 0;"
		  "int image_height = 0;"
		  "GLuint image_texture = 0;")


	     ,(lprint :msg "start loop")
	     (while (not (win.WindowShouldClose))

	       (win.PollEvents)
	       (imgui.NewFrame)

	       (progn
		 (let ((oldwidth 0)
		       (oldheight 0))
		   (declare (type "static int" oldwidth oldheight))
		   (comments "react to changing window size")
		   (let (("[width,height]" (win.GetWindowSize)))
		     (when (or (!= width oldwidth)
			       (!= height oldheight))
		       ,(lprint :msg "window size has changed" :vars `(width height))
		       (glViewport 0 0 width height)
		       (setf oldwidth width
			     oldheight height)))))

	       (progn
		 "av::Packet pkt;"
		 (while (= pkt (video.readPacket))
		   (unless (== video.videoStream
			       (pkt.streamIndex))
		     continue)
		   (let ((ts (pkt.ts)))

		     (let ((frame (video.decode)))

		       (setf ts (frame.pts))
		       (when (and (frame.isComplete)
				  (frame.isValid))
			 (let ((*data (frame.data 0)))
			   (setf image_width (dot frame
						  (-> (raw)
						      (aref linesize 0)))
				 image_height (frame.height))
			   ,(flet ((make-tex (&key sub
						   (target 'GL_TEXTURE_2D)
						   (level 0)
						   (internal-format
						    `texFormat
					;'GL_RGBA
					;'GLenum--GL_RGB8
					;'GLenum--GL_R3_G3_B2
					;'GLenum--GL_RGBA2
					;'GLenum--GL_RGB9_E5 ;; shared exponent, looks ok
					; 'GLenum--GL_SRGB8 ;; this displays as much darker
					;'GLenum--GL_RGB8UI ;; displays as black
					;'GLenum--GL_COMPRESSED_RGB ;; framerate drops from +200fps to 40fps
						    )
						   (width 'image_width)
						   (height 'image_height)
						   (xoffset 0)
						   (yoffset 0)
						   (border 0)
						   (tex-format 'GLenum--GL_LUMINANCE)
						   (tex-type 'GL_UNSIGNED_BYTE)
						   )
				     (if sub
					 `(glTexSubImage2D  ,target
							    ,level
							    ,xoffset ,yoffset
							    ,width
							    ,height
							    ,tex-format
							    ,tex-type
							    data)
					 `(glTexImage2D ,target
							,level
							,internal-format
							,width
							,height
							,border
							,tex-format
							,tex-type
							nullptr))))
			      `(let (
				     (init_width image_width)
					;(init_height image_height)
				     )
				 (if !video_is_initialized_p
				     (do0 (comments "initialize texture for video frames")
					  (glGenTextures 1 &image_texture)
					  (glBindTexture GL_TEXTURE_2D image_texture)
					  ,@(loop for (key val) in `((WRAP_S REPEAT)
								     (WRAP_T REPEAT)
								     (MIN_FILTER LINEAR)
								     (MAG_FILTER LINEAR))
						  collect
						  `(glTexParameteri GL_TEXTURE_2D
								    ,(format nil "GL_TEXTURE_~A" key)
								    ,(format nil "GL_~A" val)))

					  (do0
					   ,(lprint :msg "prepare texture"
						    :vars `(init_width image_width image_height
								       (frame.width)
								       )
						    )
					   ,(make-tex :sub nil)
					   ,(make-tex :sub t))

					  (setf video_is_initialized_p true))
				     (do0
				      (comments "update texture with new frame")
				      (glBindTexture GL_TEXTURE_2D image_texture)
				      ,(make-tex :sub t))
				     )))
			   break)))))

		 (do0
		  (comments "draw frame")
		  (do0
		   (imgui.Begin  (string "video texture"))
					;(ImGui--Text (string "width = %d") image_width)
					;(ImGui--Text (string "fn = %s") (fn.c_str))
		   (imgui.Image image_texture image_width image_height)
		   #+nil (ImGui--Image (reinterpret_cast<void*>
					(static_cast<intptr_t> image_texture))
				       (ImVec2 (static_cast<float> image_width)
					       (static_cast<float> image_height)))
		   (let ((val_old (static_cast<float> (dot pkt (ts) (seconds))))
			      (val val_old))
			  (imgui.SliderFloat (string "time")
					     &val
					     (video.startTime) ;min
					     (video.duration)
					;max
					     (string "%.3f") ; format string
					     )
			  #+nil (ImGui--SliderFloat (string "time")
						    &val
						    (video.startTime) ;min
						    (video.duration)
					;max
						    (string "%.3f") ; format string
						    )
			  (unless (== val val_old)
			    (comments "perform seek operation")
			    (video.seek val)
			    ))
		   (imgui.End))
		  (imgui.Render)


		  (glClear GL_COLOR_BUFFER_BIT)
		  (imgui.RenderDrawData)

		  (win.SwapBuffers)
		  )))))



	 (return 0))))

    (with-open-file (s (format nil "~a/CMakeLists.txt" *full-source-dir*)
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      ;;https://clang.llvm.org/docs/AddressSanitizer.html
      ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
      ;;
      (let ((dbg "-ggdb -O0 ")
	    (asan ""
					;"-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
	      )
	    ;; make __FILE__ shorter, so that the log output is more readable
	    ;; note that this can interfere with debugger
	    ;; https://stackoverflow.com/questions/8487986/file-macro-shows-full-path
	    (short-file "-ffile-prefix-map=/home/martin/stage/cl-cpp-generator2/example/85_glbinding_av/source00/=")
	    (show-err "-Wall -Wextra";
					;" -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused"
					;"-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "

	      ))
	(macrolet ((out (fmt &rest rest)
		     `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	  (out "cmake_minimum_required( VERSION 3.13 )")
	  (out "project( mytest LANGUAGES CXX )")
	  #+nil(loop for e in `(xtl xsimd xtensor)
		     do
		     (format s "find_package( ~a REQUIRED )~%" e))
	  ;;(out "set( CMAKE_CXX_COMPILER clang++ )")
	  (out "set( CMAKE_CXX_COMPILER g++ )")
	  (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	  (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a ~a \")" dbg asan show-err short-file)
	  (out "set (CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ~a ~a ~a ~a \")" dbg asan show-err short-file)
	  (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 23 )")



	  (out "set( SRCS ~{~a~^~%~} )"
	       (append
		(directory (format nil "~a/*.cpp" *full-source-dir*))
					;(directory (format nil "/home/martin/src/bgfx/examples/common/imgui/imgui.cpp"))
		(directory (format nil "/home/martin/src/imgui/backends/imgui_impl_opengl3.cpp"))
		(directory (format nil "/home/martin/src/imgui/backends/imgui_impl_glfw.cpp"))


		(directory (format nil "/home/martin/src/imgui/imgui*.cpp"))

		))

	  (out "add_executable( mytest ${SRCS} )")
					;(out "include_directories( /usr/local/include/  )")
					; /home/martin/src/entt/src/
	  (out "target_include_directories( mytest PRIVATE
/home/martin/src/imgui/
/usr/local/include
 )")

	  (out "target_compile_features( mytest PUBLIC cxx_std_20 )")

	  (out "target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )")
	  (loop for e in `(glbinding glfw3 )
		do
		(out "find_package( ~a REQUIRED )" e))


	  ;; https://stackoverflow.com/questions/8487986/file-macro-shows-full-path
	  ;;(out "set( CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -D__FILENAME__='\\\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\\\"'\" )")

	  (progn
	    (out "add_library( libavcpp_static STATIC IMPORTED )")
	    (out "set_target_properties( libavcpp_static PROPERTIES IMPORTED_LOCATION /usr/local/lib64/libavcpp.a )")
	    )

	  (let ((avlibs `(avutil avdevice avfilter avcodec avformat
				 swscale postproc swresample)))
	    #+nil (loop for e in avlibs
			collect
			`(progn
			   (out "add_library( ~a STATIC IMPORTED )" e)
			   (out "set_target_properties( ~a PROPERTIES IMPORTED_LOCATION /usr/local/lib/lib~a.a )" e)
			   ))

	    (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
		 `(
		   "glbinding::glbinding"
		   glfw3 ;GL X11
		   libavcpp_static
		   ,@avlibs

					;dl  pthread
					;rt
					;imgui

					;"imgui::imgui"
					; "implot::implot"
					;"std::mdspan"
					;"xtensor"
					;"xtensor::optimize"
					;"xtensor::use_xsimd"

		   )))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))



