(eval-when (:compile-toplevel :execute :load-toplevel)
					;(setf (readtable-case *readtable*) :upcase)
  (ql:quickload "spinneret")
  (ql:quickload "cl-cpp-generator2")
  )

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source* "05source")
  (defparameter *source-dir* (format nil "example/72_emsdk/~a/" *source*))

					;(setf (readtable-case *readtable*) :upcase)
  (load "util.lisp")
  (write-html
   (format nil  "~a/index.html" *source*)
   :str
   (spinneret:with-html-string
       (:doctype)
     (:html
      (:head
       (:meta :charset "utf-8")
       (:meta :http-equiv "Content-Type"
	      :content "text/html; charset=utf-8")
       (:title "test")
       #+nil
       (:style :type "text/css"
               (:raw
		(lass:compile-and-write
		 `(body :font-family "sans-serif")
		 `(.container :width 25% :margin auto)
		 `(.header :patting 15px
                           :text-align center
                           :font-size 2em
                           :background "#f2f2f2"
                           :margin-bottom 15px)
		 `(.header>a   :color inherit
                               :text-decoration none)
		 ))))
      (:body
       (:canvas :id "canvas"
		:oncontextmenu "event.preventDefault()"
		:width 800
		:height 600)
       (:script :type "text/javascript"
		"var Module = { canvas: (function() { return document.getElementById('canvas'); } )() };"
		)
       (:script :src "index.js")))))

  (setf (readtable-case *readtable*) :invert))

(let ((log-preamble `(do0 (include<> iostream
				     iomanip
				     chrono
				     thread
				     mutex)
			  "extern std::mutex g_stdout_mutex;"
			  "extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;")))
  (progn


    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file

    (assert (eq :invert
		(readtable-case *readtable*)))

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `index
     :headers `()
     :header-preamble `(do0
			(do0
			 "#define GLFW_INCLUDE_NONE"
			 (include<> GLFW/glfw3.h
				    flextgl12/flextGL.h
				    cmath
				    glu.h))
			(do0
			 "#ifdef __EMSCRIPTEN__"
			 (include<>
			  emscripten.h)
			 "#endif"))
     :implementation-preamble `(do0
				(include<> iostream
					   iomanip
					   chrono
					   thread
					   mutex)
				)
     :code `(do0
	     (do0
	      "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"
	      "std::mutex g_stdout_mutex;")

	     (defun reset_state ()
	       ,@(loop for (name value) in `((depth_test nil)
					     (blend nil)
					     (cull_face nil)
					     (texture_2d nil))
		       collect
		       `(,(if value
			      `glEnable
			      `glDisable)
			  ,(string-upcase (format nil "gl_~a" value))))
	       ,@(loop for e in `(
				  modelview
				  texture
				  projection)
		       collect
		       `(do0
			 (glMatrixModel ,(string-upcase (format nil "gl_~a" e)))
			 (glLoadIdentity))))
	     (defun draw_triangle ()
	       (reset_state)
	       (glBegin GL_TRIANGLES)
	       ,@(loop for e in `((:color (1 0 0 ) :uv (0 .5))
				  (:color (0 0 1 ) :uv (-.5 -.5))
				  (:color (0 1 0 ) :uv (.5 -.5)))
		       collect
		       (destructuring-bind (&key color uv) e
			 (destructuring-bind (a b c) color
			   (destructuring-bind (u v) uv
			     `(do0
			       (glColor3f (* 1s0 a) (* 1s0 b) (* 1s0 c))
			       (glVertex2f (* 1s0 u)
					   (* 1s0 v)))))))
	       (glEnd))

	     (defun main (argc argv)
	       (declare (type int argc)
			(type char** argv)
			(values int))
	       (do0 (glfwInit)
		    (glfwWindowHint GLFW_SAMPLES 4)
		    (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 1)
		    (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 2)
		    (let ((*w (glfwCreateWindwo 512 512
						(string "test glfw")
						nullptr
						nullptr)))
		      (glfwMakeContextCurrent w)
		      (glfwSwapInterval 1)
		      (flexInit)))
	       (while (!glfwWindowShouldClose w)
		 (let ((dw 0)
		       (dh 0))
		   (glfwGetFramebufferSize w   &dw &dh)
		   (let ((ww (/ dh 2))
			 (hh (/ dh  2))
			 (x0 (- (/ dw 2)
				hh))
			 (x1 (/ dw 2)
			   )
			 (y0 (/ dh 2))
			 (y1 0))
		     (glClearColor 0s0 0s0 0s0 1s0)
		     (glClearDepth 1s0)
		     (glClear (logior GL_COLOR_BUFFER_BIT
				      GL_DEPTH_BUFFER_BIT))
		     (do0 (glViewport x0 y0 ww hh)
			  (draw_triangle))
		     (glfwSwapBuffers w)
		     (glfwPollEvents))))
	       (glfwTerminate))

	     ))



    (write-cmake
     (format nil "~a/CMakeLists.txt" *source*)
     :code
     ;;  -s SAFE_HEAP=1 ;; maybe breaks debugging https://github.com/emscripten-core/emscripten/issues/8584
     (let ((dbg  "-O0" ; "-O0 -g4 -s ASSERTIONS=1 -s STACK_OVERFLOW_CHECK=1 -s DEMANGLE_SUPPORT=1"
	     ) ;; -gsource-map
	   (asan  "-fno-omit-frame-pointer -fsanitize=address "
	     ;; -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope
	     )
	   (show-err "-Wfatal-errors"; " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow  -Wswitch-default -Wundef   -Wunused -Wunused-parameter  -Wold-style-cast -Wsign-conversion "
	     ;;
	     ;; -Werror ;; i rather see the warnings
	     ;; "-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  " ;; not supported by emcc
	     ))
       (macrolet ((out (fmt &rest rest)
		    `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	 (out "cmake_minimum_required( VERSION 3.0 )")

	 (out "project( example LANGUAGES CXX C )")

	 (out "set( CMAKE_CXX_STANDARD 17 )")
	 (out "set( CMAKE_CXX_STANDARD_REQUIRED True )")
					;(out "set( OpenCV_DIR /home/martin/src/opencv/build_wasm/ )")
					;(out "set( OpenCV_STATIC ON )")
					;(out "find_package( OpenCV REQUIRED )")
					;(out "include_directories( ${OpenCV_INCLUDE_DIRS} /home/martin/src/opencv_contrib/modules/aruco/include )")
	 (out "include_directories( ~{~a~^ ~} )" `(/home/martin/src/sokol
						   /home/martin/src/imgui))

	 (progn
	   (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
					;(out "set( USE_FLAGS \"-s USE_SDL=2\" )")
					;(out "set( CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${USE_FLAGS}\" )")
	   (out "set( CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG}  ~a ~a ~a \")"
		dbg asan show-err)
	   )
	 (out "if( EMSCRIPTEN )")
	 (out "set( CMAKE_EXECUTABLE_SUFFIX \".html\" )")
	 (out "option( BUILD_WASM \"Build Webassembly\" ON )")
	 (out "else()")
	 (out "set( CMAKE_C_COMPILER clang )")
	 (out "set( CMAKE_CXX_COMPILER clang++ )")
	 (out "endif()")
	 (out "set( SRCS ~{~a~^~%~} )"
	      (append
	       (directory (format nil "~a/*.cpp" *source*))
	       (directory (format nil "~a/*.c" *source*))
					;(directory "/home/martin/src/opencv_contrib/modules/aruco/src/*.cpp")
	       ;; git clone -b docking --single-branch https://github.com/ocornut/imgui
					;(directory "/home/martin/src/imgui/imgui*.cpp")
	       ))

	 (out "add_executable( index ${SRCS} )")

	 #+nil (loop for e in `(imgui ;implot GLEW
				)
		     do
		     (out "find_package( ~a CONFIG REQUIRED )" e))

	 (out "target_link_libraries( index PRIVATE ~{~a~^ ~} )"
	      `(;"imgui::imgui"
					;"implot::implot"
					;"GLEW::GLEW"
		Xi Xcursor X11 GL
					;"${OpenCV_LIBS}"
		))
	 )))
    ))

