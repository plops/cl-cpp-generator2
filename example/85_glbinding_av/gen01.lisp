(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(let ((log-preamble
       `(do0
	 (include <iostream>
					;<iomanip>
		  <chrono>
		  <thread>
		  <spdlog/spdlog.h>
		  )
	 #+nil "void lprint(std::initializer_list<std::string> il, std::string file, int line, std::string fun);"
	 "extern const std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;")))

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

		  (ImGui--StyleColorsLight)
		  (progn
		    (let ((installCallbacks true))
		      (declare (type "const auto" installCallbacks))
		      (ImGui_ImplGlfw_InitForOpenGL window
						    installCallbacks)))

		  (let ((glslVersion (string "#version 150")))
		    (declare (type "const auto" glslVersion))
		    (ImGui_ImplOpenGL3_Init glslVersion))
		  (let ((&io (ImGui--GetIO)))
		    ,(lprint :msg "enable keyboard input for imgui")
		    (setf io.ConfigFlags

			  (logior
			   io.ConfigFlags
			   ImGuiConfigFlags_NavEnableKeyboard)))
		  )
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
	       "bool success;"
	       "public:"
	       "size_t videoStream = -1;"
	       (defmethod GetSuccess ()
		 (declare (values bool))
		 (return success))
	       (defmethod Seekable_p ()
		 (declare (values bool))
		 (return (and  success
			       (ctx.seekable)
			       )))
	       (defmethod Video (filename)
		 (declare
		  (type "std::string" filename)
		  (explicit)
		  (construct
		   (ctx (av--FormatContext))
		   (fn filename)
		   (success false))
		  (values :constructor))

		 (do0

					;(av--setFFmpegLoggingLevel AV_LOG_DEBUG)
		  (do0
		   ,(lprint :msg "open video file"
			    :svars `(fn))
		   (do0 (ctx.openInput fn ec)
			(when ec
			  ,(lprint :msg "can't open file" :svars `(fn (ec.message)))
			  return))
		   (do0 (ctx.findStreamInfo ec)
			(when ec
			  ,(lprint :msg "can't find stream info" :svars `((ec.message)))
			  return))
		   ,(lprint :msg "stream info"
			    :vars `((ctx.seekable)
				    (dot ctx (startTime) (seconds))
				    (dot ctx (duration) (seconds))
				    (ctx.streamsCount)))
		   (when (ctx.seekable)
		     (let ((center .5)
			   (timeResolution 100))
		       (declare (type "const auto" center timeResolution))
		       (comments "split second into 100 parts")
		       (ctx.seek
			(curly ("static_cast<long int>"
				(floor (* timeResolution
					  (* center (dot ctx (duration)
							 (seconds))))))
			       (curly 1 timeResolution))
			ec)
		       (when ec
			 ,(lprint :msg "can't seek" :svars `((ec.message)))
			 return)))
		   (do0
		    (for ((= "size_t i" 0)
			  (< i (ctx.streamsCount))
			  "i++")
			 (let ((st (ctx.stream i)))
			   (when (== AVMEDIA_TYPE_VIDEO
				     (st.mediaType))
			     (setf videoStream i
				   vst st)
			     break)))
		    (when (vst.isNull)
		      ,(lprint :msg "Video stream not found")
		      return)
		    (do0

		     (when (vst.isValid)
		       (setf vdec (av--VideoDecoderContext vst))
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
			 ,(lprint :msg "can't open codec")
			 return)

		       (setf success true)


		       )))))
		 )
	       (defmethod readPacket ()
		 (declare (values "av::Packet"))

		 (do0  (setf pkt (ctx.readPacket ec))
		       (when ec
			 ,(lprint :msg "packet reading error"
				  :svars `((ec.message)
					   )))
					;,(lprint :vars `((pkt.size)))
		       ;; https://www.ffmpeg.org/doxygen/3.2/avcodec_8h_source.html
		       ;; key 1, corrupt 2, discard 4
		       ;; ,(lprint :vars `((pkt.flags)))

		       )
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
		 (let ((timeResolution 1000))
		   (declare (type "const auto" timeResolution))
		   (when (and success (Seekable_p))
		     (ctx.seek (curly ("static_cast<long int>"
				       (floor (* timeResolution val)))
				      (curly 1 timeResolution))
			       ec)
		     (when ec
		       ,(lprint :msg "can't seek" :svars `((ec.message)))
		       return))))
	       (defmethod startTime ()
		 (declare (values float))
		 (when success
		   (return (static_cast<float> (dot ctx (startTime) (seconds))) ))
		 (return 0s0))
	       (defmethod duration ()
		 (declare (values float))
		 (when success
		   (return
		     (static_cast<float> (dot ctx (duration) (seconds)))))
		 (return 1s0))
	       (defmethod ~Video ()
		 (declare
		  (values :constructor))
		 )

	       )))

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `Texture
     :headers `()
     :header-preamble `(do0

			)
     :implementation-preamble `(do0
				,log-preamble
				(do0
				 (include <glbinding/gl32core/gl.h>
					  <glbinding/glbinding.h>
					; <glbinding/CallbackMask.h>
					;<glbinding/FunctionCall.h>
					;<glbinding/AbstractFunction.h>
					  )
				 "using namespace gl32core;"
				 "using namespace glbinding;")
				)
     :code
     (flet ((make-tex (&key sub
			    (target 'GL_TEXTURE_2D)
			    (level 0)
			    (internal-format
			     `internalFormat
					;'GL_RGBA
					;'GLenum--GL_RGB8
					;'GLenum--GL_R3_G3_B2
					;'GLenum--GL_RGBA2
					;'GLenum--GL_RGB9_E5 ;; shared exponent, looks ok
					; 'GLenum--GL_SRGB8 ;; this displays as much darker
					;'GLenum--GL_RGB8UI ;; displays as black
					;'GLenum--GL_COMPRESSED_RGB ;; framerate drops from +200fps to 40fps
			     )
			    (width 'w)
			    (height 'h)
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
       `(do0
	 (defclass Texture ()
	   "unsigned int image_texture;"
	   "bool initialized_p;"

	   "int m_internalFormat = 0;"
	   "int m_width = 0;"
	   "int m_height = 0;"
	   "int m_internalWidth = 0;"
	   "int m_internalHeight = 0;"

	   "public:"
	   (defmethod GetImageTexture ()
	     (declare (values "unsigned int"))
	     (return image_texture))

	   (defmethod GetWidth ()
	     (declare (values "int"))
	     (return m_width))
	   (defmethod GetHeight ()
	     (declare (values "int"))
	     (return m_height))

	   (defmethod Texture (w h internalFormat)
	     (declare
	      (type int  w h)
	      (type "int" internalFormat)
	      (explicit)
	      (construct
	       (image_texture 0)
	       (initialized_p false)
	       (m_internalFormat internalFormat)
	       (m_width w)
	       (m_height h)
	       )
	      (values :constructor))
	     (Reset nullptr w h internalFormat))
	   (defmethod Compatible_p (w h internalFormat)
	     (declare (values bool)
		      (type int w h)
		      (type "unsigned int" internalFormat))
	     (return
	       (and (== m_internalFormat internalFormat)
		    (== w m_width)
		    (== h m_height))))
	   (defmethod Reset (data w h internalFormat)
	     (declare
	      (type "int" internalFormat)
	      (type int w h)
	      (type "unsigned char*" data)
	      )
	     (when (and initialized_p
			(not (Compatible_p w h internalFormat)))
	       (glDeleteTextures 1 &image_texture)
	       (setf initialized_p false)
	       (glGenTextures 1 &image_texture)
	       )
	     (unless initialized_p
	       (do0
		(do0 (glBindTexture GL_TEXTURE_2D image_texture)
		     ,@(loop for (key val) in `((WRAP_S REPEAT)
						(WRAP_T REPEAT)
						(MIN_FILTER LINEAR)
						(MAG_FILTER LINEAR))
			     collect
			     `(glTexParameteri GL_TEXTURE_2D
					       ,(format nil "GL_TEXTURE_~A" key)
					       ,(format nil "GL_~A" val))))
		(do0
		 ,(make-tex :sub nil)
		 (setf m_internalWidth w
		       m_internalHeight h))))
	     (do0

	      (do0
	       (glBindTexture GL_TEXTURE_2D image_texture)
	       ,(make-tex :sub t)
	       (setf m_width w
		     m_height h)))

	     (setf initialized_p true))

	   (defmethod ~Texture ()
	     (declare
	      (values :constructor))
	     (when initialized_p
	       (setf initialized_p false)
	       (glDeleteTextures 1 &image_texture))
	     )) )))

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

		 <glbinding/CallbackMask.h>
		 <glbinding/FunctionCall.h>
		 <glbinding/AbstractFunction.h>
		 )
	"using namespace gl32core;"
	"using namespace glbinding;")
       (do0
	(include <imgui.h>
					;<backends/imgui_impl_glfw.h>
					;<backends/imgui_impl_opengl3.h>
		 )
	#+nil (do0 "#define GLFW_INCLUDE_NONE"
		   (include <GLFW/glfw3.h>
			    ))
	(include "GlfwWindow.h"
		 "ImguiHandler.h"
		 "Video.h"
		 "Texture.h")

	(do0
	 (include <avcpp/av.h>
					;	  <avcpp/ffmpeg.h>
		  )
	 ;; API2
	 (include			;<avcpp/format.h>
	  <avcpp/formatcontext.h>
					;<avcpp/codec.h>
					;<avcpp/codeccontext.h>
	  ))

	(include <popl.hpp>)
	,log-preamble

	)

       "const std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time = std::chrono::high_resolution_clock::now();"


       ,(init-lprint)
       (defun main (argc argv)
	 (declare (type int argc)
		  (type char** argv)
		  (values int))
	 ,(lprint :msg "start" :vars `(argc))

	 (let ((op (popl--OptionParser (string "allowed opitons")))
	       (varInternalTextureFormat (int 3))
	       ,@(loop for e in `((:long help :short h :type Switch :msg "produce help message")
				  (:long verbose :short v :type Switch :msg "produce verbose output")
				  (:long texformat :short T :type int :msg "choose internal texture format"
					 :default 3 :out &varInternalTextureFormat)

				  )
		       appending
		       (destructuring-bind (&key long short type msg default out) e
			 `(#+nil (,(format nil "~aVar" long) (,(case type
								 (`Switch `(bool false))
								 (`Switch `(bool false)))))
				 (,(format nil "~aOption" long)
				   ,(let ((cmd `(,(format nil "add<~a>"
							  (if (eq type 'Switch)
							      "popl::Switch"
							      (format nil "popl::Value<~a>" type)))
						  (string ,short)
						  (string ,long)
						  (string ,msg))))
				      (when default
					(setf cmd (append cmd `(,default)))
					)
				      (when out
					(setf cmd (append cmd `(,out)))
					)
				      `(dot op ,cmd)
				      ))))
		       ))
	   (op.parse argc argv)
	   (when (helpOption->count)
	     (<< std--cout
		 op
		 std--endl)
	     (exit 0)))

	 (do0
	  ,(let ((tex-formats `((GL_RGBA)
				(GLenum--GL_RGB8)
				(GLenum--GL_R3_G3_B2)
				(GLenum--GL_RGBA2 :default t :comment "weirdest effect")
				(GLenum--GL_RGB9_E5 :comment "shared exponent, looks ok")
				(GLenum--GL_SRGB8 :comment "this displays as much darker")
				(GLenum--GL_RGB8UI :comment "displays as black")
				(GLenum--GL_COMPRESSED_RGB :comment "framerate drops from +200fps to 40fps"))))
	     `(let ((texFormatIdx varInternalTextureFormat
		      #+nil (dot (aref opt_res (string "internal-tex-format"))
				 (as<int>)))
		    (numTexFormats ,(length tex-formats)))
		(declare (type "const auto" numTexFormats))
 		(assert (<= 0 texFormatIdx))
		(assert (< texFormatIdx numTexFormats))

		(let ((texFormatsString ("std::array<std::string,numTexFormats>"
					 (curly ,@(loop for e in tex-formats
							collect
							(destructuring-bind ( val &key comment default) e
							  `(string ,val))))))
		      (texFormats ("std::array<gl::GLenum,numTexFormats>"
				   (curly ,@(loop for e in tex-formats
						  collect
						  (destructuring-bind ( val &key comment default) e
						    val)))))
		      (texFormat (dot texFormats (at texFormatIdx))))))))

	 (let ((win (GlfwWindow)))
	   (do0
	    ,(lprint :msg "initialize glbinding")
	    (comments "if second arg is false: lazy function pointer loading")
	    (glbinding--initialize win.GetProcAddress
				   false)
	    (when
		(verboseOption->is_set)
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
			,(lprint :level "trace" :msg "cb" :svars `(fun)))))))
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
	    (let ()))
	   (let ((fn (dot op (non_option_args) (at 0)))
		 (video
		  ("std::make_unique<Video>" fn))
		 (initW 640)
		 (initH 480))
	     (declare (type "const auto" initW initH))


	     (let ((texture (Texture initW initH ("static_cast<int>" texFormat))))
	       (do0
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
		    (while ((lambda ()
			      (declare (capture &pkt &video)
				       (values bool))
			      (unless (video->GetSuccess)
				(return false))
			      (= pkt (video->readPacket))
			      (when (<= (pkt.size) 0)
				(return false))
			      (unless (pkt.flags)
				,(lprint :level "trace"
					 :msg "normal pkt"
					 :vars `((pkt.size)
						 (pkt.flags))))
			      (when (& 1 (pkt.flags))
				,(lprint :level "trace"
					 :msg "pkt contains keyframe"
					 :vars `((pkt.size)
						 (pkt.flags))))
			      (when (& 2 (pkt.flags))
				,(lprint :msg "pkt corrupt"
					 :vars `((pkt.size)
						 (pkt.flags)))
				(return false))
			      (when (& 4 (pkt.flags))
				,(lprint :msg "pkt discard"
					 :vars `((pkt.size)
						 (pkt.flags))))

			      (return true)
			      ))
		      (unless (== video->videoStream
				  (pkt.streamIndex))
			continue)
		      (let ((ts (pkt.ts)))
			(let ((frame (video->decode)))
			  (setf ts (frame.pts))
			  (when (and (frame.isComplete)
				     (frame.isValid))
			    (space auto* (data (frame.data 0)))
			    (let ((w (dot frame
					  (-> (raw)
					      (aref linesize 0))))
				  (h (frame.height)))
			      (texture.Reset data w h ("static_cast<int>" texFormat)))

			    break))))

		    (do0
		     (comments "draw frame")
		     (do0
		      (imgui.Begin  (string "video texture"))
					;(ImGui--Text (string "width = %d") image_width)
		      (ImGui--Text (string "fn = %s") (fn.c_str))

		      (if (video->GetSuccess)
			  (do0
			   (imgui.Image (texture.GetImageTexture)
					(texture.GetWidth)
					(texture.GetHeight))
			   (if (video->Seekable_p)
			       (let ((val_old (static_cast<float> (dot pkt (ts) (seconds))))
				     (val val_old))
				 (imgui.SliderFloat (string "time")
						    &val
						    (video->startTime)
						    (video->duration)
						    (string "%.3f"))
				 (unless (== val val_old)
				   ,(lprint :msg "perform seek operation")
				   (video->seek val)))
			       (do0
				(ImGui--Text (string "can't seek in file"))
				)))
			  (do0
			   (ImGui--Text (string "could not open video file"))))

		      (imgui.End))

		     (do0
		      (comments "window with file listing")
		      (imgui.Begin  (string "video files"))
		      (let ((item_current_idx (int 0))
			    (item_old_idx (int 0)))
			(declare (type "static int"
				       item_current_idx
				       item_old_idx))
			(let ((filesToShow 40s0))
			  (declare (type "const auto" filesToShow))
			  (ImGui--BeginListBox (string "files")
					       (ImVec2 -FLT_MIN
						       (* filesToShow (ImGui--GetTextLineHeightWithSpacing)))
					;(ImGui--GetItemRectSize)
					       ))
			(let ((i 0))
			  (for-range (arg (op.non_option_args))
				     (let ((selected_p (== i item_current_idx)))
				       (when (ImGui--Selectable (arg.c_str) selected_p)

					 (setf item_current_idx i))
				       (when selected_p
					 (ImGui--SetItemDefaultFocus)

					 (unless (== item_old_idx
						     item_current_idx)
					   ,(lprint :msg "change video")
					   (setf item_old_idx item_current_idx)
					   (setf  fn arg)
					   #+nil (do0 (delete video)
						      (setf video (new (Video fn))))
					   (setf video (std--make_unique<Video> fn)))))
				     (setf i (+ i 1))))
			(ImGui--EndListBox))
		      (imgui.End))

		     (progn
		       (comments "window with internal texture format listing")
		       (imgui.Begin  (string "internal texture format"))
		       (let ((fmt_current_idx varInternalTextureFormat)
			     (fmt_old_idx  varInternalTextureFormat))
			 (declare (type "static int"
					fmt_current_idx
					fmt_old_idx))
			 (let ((formatsToShow 15s0))
			   (declare (type "const auto" formatsToShow))
			   (ImGui--BeginListBox (string "files")
						(ImVec2 -FLT_MIN
							(* formatsToShow (ImGui--GetTextLineHeightWithSpacing)))
						))
			 (let ((i 0))
			   (for-range (arg texFormats)
				      (let ((selected_p (== i fmt_current_idx))
					    (argString (dot texFormatsString (at i))
					      #+nil (fmt--format (string "{}")
								 (static_cast<int> arg))))
					(when (ImGui--Selectable (dot argString (c_str)) selected_p)
					  (setf fmt_current_idx i))
					(when selected_p
					  (ImGui--SetItemDefaultFocus)

					  (unless (== fmt_old_idx
						      fmt_current_idx)
					    ,(lprint :msg "change texture format" :svars `(argString))
					    (setf fmt_old_idx fmt_current_idx)
					    (setf varInternalTextureFormat
						  fmt_current_idx)
					    (setf texFormat (dot texFormats (at varInternalTextureFormat)))
					    )))
				      (setf i (+ i 1))))
			 (ImGui--EndListBox))
		       (imgui.End))

		     (imgui.Render)


		     (glClear GL_COLOR_BUFFER_BIT)
		     (imgui.RenderDrawData)

		     (win.SwapBuffers)
		     )))))))



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
	    (short-file "-ffile-prefix-map=/home/martin/stage/cl-cpp-generator2/example/85_glbinding_av/source01/=")
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
/home/martin/src/popl/include/
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
	  #+nil
	  (progn
	    (out "add_library( fmt_static STATIC IMPORTED )")
	    (out "set_target_properties( fmt_static PROPERTIES IMPORTED_LOCATION /usr/local/lib64/libfmt.a )")
	    )

	  #+nil (progn
		  (out "add_library( spdlog_static STATIC IMPORTED )")
		  (out "set_target_properties( spdlog_static PROPERTIES IMPORTED_LOCATION /usr/local/lib64/libspdlog.a )")
		  )

	  (out "find_package( PkgConfig REQUIRED )")
	  (out "pkg_check_modules( spdlog REQUIRED spdlog )")

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
		   glfw3		;GL X11
		   libavcpp_static
		   ,@avlibs
		   spdlog
					;spdlog_static
					;fmt_static

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



