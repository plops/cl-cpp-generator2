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

    (defparameter *source-dir* #P"example/83_glfw_bgfx/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))

    (ensure-directories-exist *full-source-dir*)
    (load "../83_glfw_bgfx/util.lisp")

    #+nil
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
     :name `PosColorVertex
     :headers `()
     :header-preamble `(include <bgfx/bgfx.h>

				)
     :implementation-preamble `(include <bgfx/bgfx.h>

					)
     :code `(do0
	     (defclass PosColorVertex ()
	       "public:"
	       ,@(loop for e in `(x y z)
		       collect
		       (format nil "float m_~a;" e))
	       ,@(loop for e in `(abgr)
		       collect
		       (format nil "uint32_t m_~a;" e))
	       "static bgfx::VertexLayout ms_decl;"

	       (defmethod init ()
		 (declare
		  (static))
		 (dot ms_decl
		      (begin)
		      (add bgfx--Attrib--Position 3 bgfx--AttribType--Float)
		      (add bgfx--Attrib--Color0 4 bgfx--AttribType--Uint8 true)
		      (end)))

	       )))

    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"v_simple.sc"
		       *source-dir*))
     ;; vertex shader
     `(do0
       "$input a_position, a_color0"
       "$output v_color0"

       (include <bgfx_shader.sh>
		"shaderlib.sh")

       (defun main ()
	 (setf gl_Position (mul u_modelViewProj
				(vec4 a_position
				      1s0))
	       v_color a_color0)))
     :format nil)

    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"f_simple.sc"
		       *source-dir*))
     ;; fragment shader
     `(do0
       "$input v_color0"

       (include <bgfx_shader.sh>
		"shaderlib.sh")

       (defun main ()
	 (setf gl_FragColor v_color0)))
     :format nil)

    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"varying.def.sc"
		       *source-dir*))
     `(do0
       "vec4 v_color0 : COLOR0;"
       "vec3 a_position : POSITION;"
       "vec4 a_color0 : COLOR0;
"
       )
     :format nil)

    (sb-ext:run-program
     "/usr/bin/bash"
     `("/home/martin/stage/cl-cpp-generator2/example/83_glfw_bgfx/source00/setup01_compile_shader.sh")
     )

    (write-source (asdf:system-relative-pathname
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
					;  <memory>
		     )


		    (do0
					;"#define BX_CONFIG_DEBUG"
					;(include <bx/bx.h>)
		     (include <bgfx/bgfx.h>
			      <bgfx/platform.h>
			      )
		     (include "PosColorVertex.h")
		     (include <fstream>))
		    (do0
		     (include <GLFW/glfw3.h>
			      )
		     "#define GLFW_EXPOSE_NATIVE_X11"
		     (include
		      <GLFW/glfw3native.h>)
		     (include <imgui/imgui.h>)
		     )

					;"namespace stdex = std::experimental;"


		    "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"

		    "bgfx::VertexLayout PosColorVertex::ms_decl;"

		    (let (("s_cubeVertices[]"
			   (curly ,@(loop for e in `((.5 .5 .0 #xff0000ff)
						     (.5 -.5 .0 #xff0000ff)
						     (-.5 -.5 .0 #xff00ff00)
						     (-.5 .5 .0 #xff00ff00))
					  collect
					  (destructuring-bind (x y z col) e
					    `(curly ,x ,y ,z
						    (hex ,col)))))))
		      (declare (type "static PosColorVertex" "s_cubeVertices[]")))

		    (let (("s_cubeTriList[]"
			   (curly 0 1 3
				  1 2 3)))
		      (declare (type "static const uint16_t" "s_cubeTriList[]")))
		    ,@(loop for (e f) in `((VertexBufferHandle vbh)
					   (IndexBufferHandle ibh)
					   (ProgramHandle program))
			    collect
			    (format nil "bgfx::~a m_~a;"
				    e f))

		    ,(init-lprint)

		    (defun loadShader (_name)
		      (declare (type "const char*" _name)
			       (values "bgfx::ShaderHandle"))
		      ,(lprint :msg "loadShader" :vars `())
		      (let ((*data (new "char[2048]"))
			    (file (std--ifstream))
			    (fileSize (size_t 0)))
			(file.open _name)
			(if (file.is_open)
			    (do0
			     ,(lprint :msg "load shader from file")
			     (file.seekg 0 std--ios--end)
			     (setf fileSize (file.tellg))
			     (file.seekg 0 std--ios--beg)
			     (assert (< fileSize 2048))
			     (file.read data fileSize)
			     (file.close))
			    (do0
			     ,(lprint :msg "warning: can't open shader file")))
			(let ((mem (bgfx--copy data (+ fileSize 1))))
			  (setf (aref mem->data (- mem->size
						   1))
				(char "\\0"))
			  (let ((handle (bgfx--createShader mem)))
			    (bgfx--setName handle _name)
			    (return handle)))))

		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))


		      (setf g_start_time ("std::chrono::high_resolution_clock::now"))

		      ,(lprint :msg "start" :vars `(argc))

		      (let ((*window ((lambda ()
					(declare (values GLFWwindow*))
					(unless (glfwInit)
					  ,(lprint :msg "glfwInit failed"))
					(let ((window (glfwCreateWindow 800 600
									(string "hello bgfx")
									nullptr
									nullptr)))
					  (unless window
					    ,(lprint :msg "can't create glfw window"))
					  (return window))
					)))))
		      (let ((width (int 0))
			    (height (int 0)))

			((lambda ()
			   (declare (capture &width &height window ))
			   (comments "initialize bgfx")
			   (do0
			    (comments "call renderFrame before bgfx::init to signal to bgfx not to create a render thread")
			    (bgfx--renderFrame))
			   (let ((bi (bgfx--Init)))
			     (do0
			      (setf bi.platformData.ndt (glfwGetX11Display))
			      (setf bi.platformData.nwh (reinterpret_cast<void*> (static_cast<uintptr_t> (glfwGetX11Window window)))))
			     (glfwGetWindowSize window &width &height)
			     ,@(loop for (e f)
				     in `((type bgfx--RendererType--Count)
					  (resolution.width width)
					  (resolution.height height)
					  (resolution.reset BGFX_RESET_VSYNC))
				     collect
				     `(setf (dot bi ,e)
					    ,f))
			     (unless (bgfx--init bi)
			       ,(lprint :msg "bgfx init failed"))

			     (do0
			      (PosColorVertex--init)
			      (setf m_vbh (bgfx--createVertexBuffer
					   (bgfx--makeRef s_cubeVertices
							  (sizeof s_cubeVertices))
					   PosColorVertex--ms_decl))
			      (setf m_ibh (bgfx--createIndexBuffer
					   (bgfx--makeRef s_cubeTriList
							  (sizeof s_cubeTriList))))
			      (let ((vsh (bgfx--ShaderHandle (loadShader (string "v_simple.bin"))))
				    (fsh (bgfx--ShaderHandle (loadShader (string "f_simple.bin"))))
				    )
				(setf m_program (bgfx--createProgram vsh fsh true))))


			     (let ((debug BGFX_DEBUG_TEXT))
			       (bgfx--setDebug debug)
			       (bgfx--setViewClear 0
						   (logior BGFX_CLEAR_COLOR
							   BGFX_CLEAR_DEPTH)
						   (hex #x303030ff)
						   1s0
						   0)
			       (bgfx--setViewRect 0
						  0 0
						  bgfx--BackbufferRatio--Equal)
			       (imguiCreate))
			     )))

			(while (not (glfwWindowShouldClose window))
			  (glfwPollEvents)

			  #+nil ((lambda ()
				   (imguiBeginFrame)
				   (showExampleDialog )
				   (imguiEndFrame)))

			  ((lambda ()
			     (declare (capture &width &height window))
			     (comments "react to changing window size")
			     (let ((oldwidth width)
				   (oldheight height))
			       (glfwGetWindowSize window &width &height)
			       (when (or (!= width oldwidth)
					 (!= height oldheight))
				 (bgfx--reset width height BGFX_RESET_VSYNC)
				 (bgfx--setViewRect 0
						    0 0
						    bgfx--BackbufferRatio--Equal)))))
			  (do0
			   (comments "draw frame")
			   (bgfx--touch 0)
			   (bgfx--dbgTextClear)
			   (bgfx--dbgTextPrintf 0 0 #xf (string "press F1 to toggle stats"))
			   (bgfx--setDebug BGFX_DEBUG_STATS ;TEXT
					   )
			   (bgfx--frame))))
		      (bgfx--shutdown)
		      (glfwTerminate)




		      (return 0))))

    (with-open-file (s (format nil "~a/CMakeLists.txt" *full-source-dir*)
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      ;;https://clang.llvm.org/docs/AddressSanitizer.html
      ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
      ;;
      (let ((dbg "-ggdb -O0")
	    (asan "" ; "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
	      )
	    (show-err ;"";
	     " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused"
					;"-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "

	      ))
	(macrolet ((out (fmt &rest rest)
		     `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	  (out "cmake_minimum_required( VERSION 3.13 )")
	  (out "project( mytest LANGUAGES CXX )")
	  #+nil(loop for e in `(xtl xsimd xtensor)
		     do
		     (format s "find_package( ~a REQUIRED )~%" e))
					;(out "set( CMAKE_CXX_COMPILER clang++ )")
	  (out "set( CMAKE_CXX_COMPILER g++ )")
	  (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	  (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a \")" dbg asan show-err)
	  (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 23 )")



	  (out "set( SRCS ~{~a~^~%~} )"
	       (append
		(directory (format nil "~a/*.cpp" *full-source-dir*))
		(directory (format nil "/home/martin/src/bgfx/examples/common/imgui/imgui.cpp"))
		(directory (format nil "/home/martin/src/bgfx/3rdparty/dear-imgui/imgui*.cpp"))

		))

	  (out "add_executable( mytest ${SRCS} )")
					;(out "include_directories( /usr/local/include/  )")
	  (out "target_include_directories( mytest PRIVATE
/home/martin/src/bgfx/include/
/home/martin/src/bx/include/
/home/martin/src/bimg/include/
/home/martin/src/bgfx/examples/common
/home/martin/src/bgfx/3rdparty/ )")

	  (out "target_compile_features( mytest PUBLIC cxx_std_20 )")

					;(out "target_link_options( mytest PRIVATE -static -static-libgcc -static-libstdc++  )")
	  #+nil (loop for e in `(imgui implot)
		      do
		      (out "find_package( ~a CONFIG REQUIRED )" e))
					;(out "link_directories( /home/martin/src/bgfx/.build/linux64_gcc/bin/ )")

	  #+nil
	  (progn
	    (out "add_library( bgfx STATIC IMPORTED )")  ;; SHARED or STATIC
	    (out "set_target_properties( bgfx PROPERTIES
IMPORTED_LOCATION \"/home/martin/src/bgfx/.build/linux64_gcc/bin/libbgfxRelease.a\"
INTERFACE_INCLUDE_DIRECTORIES \"/home/martin/src/bgfx/include\" )")
	    )

	  #+nil
	  (loop for e in `(bx bgfx bimg)
		do
		(progn
		  (out "add_library( ~a STATIC IMPORTED )" e)
		  (out "set_target_properties( ~a PROPERTIES
IMPORTED_LOCATION \"/home/martin/src/bgfx/.build/linux64_gcc/bin/lib~aRelease.a\"
INTERFACE_INCLUDE_DIRECTORIES \"/home/martin/src/~a/include\" )" e e e)
		  ))


	  (progn
	    (out "add_library( bgfx-shared SHARED IMPORTED )")
	    (out "set_target_properties( bgfx-shared PROPERTIES
IMPORTED_LOCATION \"/home/martin/src/bgfx/.build/linux64_gcc/bin/libbgfx-shared-libRelease.so\"
INTERFACE_INCLUDE_DIRECTORIES \"/home/martin/src/bgfx/include\" )")
	    )



	  #+nil(progn
		 (out "add_library( imgui SHARED IMPORTED )")
		 (out "set_target_properties( imgui PROPERTIES
INTERFACE_INCLUDE_DIRECTORIES \"/home/martin/src/bgfx/examples/common\" )")
		 )

	  (out "add_definitions( -DBX_CONFIG_DEBUG )")



	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `(
		 bgfx-shared
					; bx bgfx bimg
		 GL X11 glfw
		 dl pthread rt
					;imgui

					;"imgui::imgui"
					; "implot::implot"
					;"std::mdspan"
					;"xtensor"
					;"xtensor::optimize"
					;"xtensor::use_xsimd"

		 ))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))



