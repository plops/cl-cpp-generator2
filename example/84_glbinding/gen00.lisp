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

    (defparameter *source-dir* #P"example/84_glbinding/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))

    (ensure-directories-exist *full-source-dir*)
    (load "../84_glbinding/util.lisp")

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


		    #+nil (do0

			   (include <fstream>
				    <array>))

		    (do0
		     (include <glbinding/gl32core/gl.h>
			      <glbinding/glbinding.h>
			      <glbinding/CallbackMask.h>
			      <glbinding/FunctionCall.h>
			      <glbinding/AbstractFunction.h>)
		     "using namespace gl32core;"
		     "using namespace glbinding;")
		    (do0
		     (include <imgui.h>
			      <backends/imgui_impl_glfw.h>
			      <backends/imgui_impl_opengl3.h>)
		     "#define GLFW_INCLUDE_NONE"
		     (include <GLFW/glfw3.h>
			      )
		     #+nil (do0 "#define GLFW_EXPOSE_NATIVE_X11"
				(include
				 <GLFW/glfw3native.h>))

		     (do0
					;(include <entt/entt.hpp>)
		      (include <imgui_entt_entity_editor.hpp>))


		     )



					;"namespace stdex = std::experimental;"


		    "const std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time = std::chrono::high_resolution_clock::now();"
		    (do0
		     (defclass+ Transform ()
		       "public:"
		       (let ((x 0s0)
			     (y 0s0))
			 (declare (type float x y))))

		     (defclass+ Velocity ()
		       "public:"
		       (let ((x 0s0)
			     (y 0s0))
			 (declare (type float x y))))

		     (defun computeVelocity (reg delta width height)
		       (declare (type "entt::registry&" reg)
				(type float delta width height))
		       (dot reg
			    ("view<Transform,Velocity>")
			    (each
			     (lambda (trans vel)
			       (declare (capture "&")
					(type Transform& trans)
					(type Velocity& vel))
			       (incf trans.x (* vel.x delta))
			       (incf trans.y (* vel.y delta))
			       (when (or (< trans.x 0s0)
					 (< width trans.x))
				 (setf trans.x (std--clamp trans.x
							   0s0
							   width))
				 (setf vel.x -vel.x))
			       (when (or (< trans.y 0s0)
					 (< height trans.y))
				 (setf trans.y (std--clamp trans.y
							   0s0
							   height))
				 (setf vel.y -vel.y))
			       ))))


		     (space
		      "namespace MM"
		      (progn
			,@(loop for e in `(Transform Velocity)
				collect
				`(space "template<>"
					(defun ,(format nil "ComponentEditorWidget<~a>" e) (reg e)
					  (declare (type "entt::registry&" reg)
						   (type "entt::registry::entity_type" e))
					  (let ((&t (,(format nil "reg.get<~a>" e) e))
						(step .1s0))
					    (declare (type "const auto" step))
					    (ImGui--DragFloat (string "x")
							      &t.x step)
					    (ImGui--DragFloat (string "y")
							      &t.y step)))))
			))

		     )

		    ,(init-lprint)


		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))


					;(setf g_start_time ("std::chrono::high_resolution_clock::now"))

		      ,(lprint :msg "start" :vars `(argc))

		      (let ((*window ((lambda ()
					(declare (values GLFWwindow*))
					(unless (glfwInit)
					  ,(lprint :msg "glfwInit failed"))
					(glfwWindowHint GLFW_VISIBLE true)
					(glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 3)
					(glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 2)
					(glfwWindowHint GLFW_OPENGL_FORWARD_COMPAT true)
					(glfwWindowHint GLFW_OPENGL_PROFILE
							GLFW_OPENGL_CORE_PROFILE)
					(comments "enable Vsync")
					(glfwSwapInterval 1)
					(let ((startWidth 800)
					      (startHeight 600)
					      (window (glfwCreateWindow startWidth startHeight
									(string "hello bgfx")
									nullptr
									nullptr))
					      )
					  (declare (type "const auto" startWidth startHeight))
					  (unless window
					    ,(lprint :msg "can't create glfw window"))
					  (glfwMakeContextCurrent window)
					  (return window))
					)))))
		      (let ((width (int 0))
			    (height (int 0)))

			(do0
			 (comments "if second arg is false: lazy function pointer loading")
			 (glbinding--initialize glfwGetProcAddress
						false
						)
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

			(do0
			 (IMGUI_CHECKVERSION)
			 (ImGui--CreateContext)
			 (let ((io (ImGui--GetIO)))
			   (setf io.ConfigFlags
				 (logior
				  io.ConfigFlags
				  ImGuiConfigFlags_NavEnableKeyboard)))
			 (ImGui--StyleColorsLight)
			 (let ((installCallbacks true))
			   (declare (type "const auto" installCallbacks))
			   (ImGui_ImplGlfw_InitForOpenGL window installCallbacks))

			 (let ((glslVersion (string "#version 150")))
			   (declare (type "const auto" glslVersion))
			   (ImGui_ImplOpenGL3_Init glslVersion)))

			(do0
			 ,(lprint :msg "initialize ENTT")

			 "entt::registry reg;"
			 "MM::EntityEditor<entt::entity> editor;"

			 (editor.registerComponent<Transform> (string "Transform"))
			 (editor.registerComponent<Velocity> (string "Velocity"))
			 (do0
			  "entt::entity e;"
			  (let ((n 1000))
			    (declare (type "const auto" n))
			    (dotimes (i n)
			      (setf e (reg.create))
			      (let ((range 5000)
				    (offset (/ range 2))
				    (scale .1s0))
				(declare (type "const auto" range offset scale))
				(reg.emplace<Transform> e
							(* scale (static_cast<float>
								  (% (rand)
								     range)))
							(* scale (static_cast<float>
								  (% (rand)
								     range)))))
			      (reg.emplace<Velocity> e
						     (* scale (static_cast<float>
							       (+ -offset
								  (% (rand)
								     range))))
						     (* scale (static_cast<float>
							       (+ -offset
								  (% (rand)
								     range)))))))))

			(while (not (glfwWindowShouldClose window))
			  (glfwPollEvents)
			  (let ((framesPerSecond 60s0))
			    (declare (type "const auto" framesPerSecond))
			    (computeVelocity reg
					     (/ 1s0
						framesPerSecond)
					     (static_cast<float> width)
					     (static_cast<float> height)))
			  (do0
			   (ImGui_ImplOpenGL3_NewFrame)
			   (ImGui_ImplGlfw_NewFrame)

			   #+nil
			   (do0
			    (ImGui--NewFrame)
			    (editor.renderSimpleCombo reg e)
			    (ImGui--Render))

			   (do0
			    (ImGui--NewFrame)
			    (let ((showDemoWindow true))
			      (ImGui--ShowDemoWindow &showDemoWindow))
			    (ImGui--Render)))


			  ((lambda ()
			     (declare (capture &width &height window))
			     (comments "react to changing window size")
			     (let ((oldwidth width)
				   (oldheight height))
			       (glfwGetWindowSize window &width &height)
			       (when (or (!= width oldwidth)
					 (!= height oldheight))
				 (comments "set view")
				 (glViewport 0 0 width height)))))
			  (do0
			   (comments "draw frame")
			   (glClear GL_COLOR_BUFFER_BIT)
			   (ImGui_ImplOpenGL3_RenderDrawData
			    (ImGui--GetDrawData))
			   (glfwSwapBuffers window)

			   )))

		      (do0
		       (ImGui_ImplOpenGL3_Shutdown)
		       (ImGui_ImplGlfw_Shutdown)
		       (ImGui--DestroyContext)
		       (glfwDestroyWindow window)
		       (glfwTerminate))




		      (return 0))))

    (with-open-file (s (format nil "~a/CMakeLists.txt" *full-source-dir*)
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      ;;https://clang.llvm.org/docs/AddressSanitizer.html
      ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
      ;;
      (let ((dbg "-ggdb -O0")
	    (asan ;""
	     "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
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
	  ;;(out "set( CMAKE_CXX_COMPILER clang++ )")
	  (out "set( CMAKE_CXX_COMPILER g++ )")
	  (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	  (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a \")" dbg asan show-err)
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
/home/martin/src/entt/src/
/home/martin/src/imgui_entt_entity_editor/
 )")

	  (out "target_compile_features( mytest PUBLIC cxx_std_20 )")

	  (out "target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )")
	  (loop for e in `(glbinding glfw3)
		do
		(out "find_package( ~a REQUIRED )" e))






	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `(
		 "glbinding::glbinding"
		 glfw3 ;GL X11
					;dl  pthread
					;rt
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



