(eval-when (:compile-toplevel :execute :load-toplevel)
					;(setf (readtable-case *readtable*) :upcase)
  (ql:quickload "spinneret")
  (ql:quickload "cl-cpp-generator2")
  )

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source* "06source")
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
       )
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
			 ;; https://gist.github.com/ousttrue/0f3a11d5d28e365b129fe08f18f4e141
			 "#ifdef __EMSCRIPTEN__"
			 "#define SOKOL_GLES2"
			 (include<>
			  emscripten.h)
			 "#else"
			 "#define SOKOL_GLCORE33"
			 "#endif")
			(do0
		         "#define SOKOL_IMPL"
					;"#define SOKOL_GL_IMPL"
			 (include<> sokol_app.h
				    sokol_gfx.h
				    sokol_glue.h)


			 (include<> util/sokol_gl.h))
	       		)
     :implementation-preamble `(do0
				(include<> iostream
					   iomanip
					   chrono
					   thread
					   mutex
					   functional))
     :code `(do0
	     (do0
	      "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"
	      "std::mutex g_stdout_mutex;")

	     (defclass+ State ()
	       "public:"
	       "sg_pass_action pass_action;"
	       "sg_image img;"
	       "sgl_pipeline pip_3d;")
	     ,@(loop for e in `(init draw_triangle frame cleanup)
		     collect
		     `(do0
		       ,(format nil "std::function<void()> ~a;" e)
		       (defun
			   ,(format nil "~a_cfun" e)
			   ()
			 ( ,(format nil "~a" e)))))



	     (defun sokol_main (argc argv)
	       (declare (type int argc)
			(type char** argv)
			(values sapp_desc))
	       (do0
		"static State state;")

	       (comments "designated initializers require c++20")
	       (let ((sgd (space sg_desc
				 (designated-initializer
				  :context (sapp_sgcontext)))))
		 (sg_setup (ref sgd
				)))


	       (let ((sap (sapp_desc)))
		 ,@(loop for (e f) in `((init_cb init_cfun)
					(frame_cb frame_cfun)
					(cleanup_cb cleanup_cfun)
					;(event_cb event)
					(width 512)
					(height 512)
					(sample_count 4)
					(gl_force_gles2 true)
					(window_title (string "sokol_gl app"))
					(icon.sokol_default true))
			 collect
			 `(setf (dot sap ,e)
				,f)))
	       (return sap))

	     ))



    (write-cmake
     (format nil "~a/CMakeLists.txt" *source*)
     :code
     ;;  -s SAFE_HEAP=1 ;; maybe breaks debugging https://github.com/emscripten-core/emscripten/issues/8584
     (let ((dbg  "-O0" ; "-O0 -g4 -s ASSERTIONS=1 -s STACK_OVERFLOW_CHECK=1 -s DEMANGLE_SUPPORT=1"
	     ) ;; -gsource-map
	   (asan "" ; "-fno-omit-frame-pointer -fsanitize=address "
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

	 (out "set( CMAKE_CXX_STANDARD 20 )")
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

	 (out "set( SRCS ~{~a~^~%~} )"
	      (append
	       (directory (format nil "~a/*.cpp" *source*))
	       (directory (format nil "~a/*.c" *source*))
					;(directory "/home/martin/src/imgui/imgui*.cpp")
	       ))
	 (out "add_executable( index ${SRCS} )")

	 (out "if( EMSCRIPTEN )")
	 (out "set( CMAKE_EXECUTABLE_SUFFIX \".html\" )")
					;(out "option( BUILD_WASM \"Build Webassembly\" ON )")
	 (out "set_target_properties( index PROPERTIES LINK_FLAGS \"-s WASM=1 -s LEGACY_GL_EMULATION=1 -s USE_GLFW=3\" ) ")
	 (out "else()")
	 (out "set( CMAKE_C_COMPILER clang )")
	 (out "set( CMAKE_CXX_COMPILER clang++ )")
	 (out "target_link_libraries( index PRIVATE ~{~a~^ ~} )"
	      `(X11 GL Xext Xi Xcursor))
	 (out "endif()")




	 #+nil (loop for e in `(imgui ;implot GLEW
				)
		     do
		     (out "find_package( ~a CONFIG REQUIRED )" e))


	 )))
    ))

