(eval-when (:compile-toplevel :execute :load-toplevel)
					;(setf (readtable-case *readtable*) :upcase)
  (ql:quickload "spinneret")
  (ql:quickload "cl-cpp-generator2")
  )

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source* "04source")
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

    (write-impl-class
     :name `SokolApp
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :private-header-preamble
     `(do0
					;"#define SOKOL_GLES2"
       (include<>
	memory
					;sokol_app.h
					;sokol_gfx.h
					;sokol_glue.h

	)
       "class sg_desc;"
       "extern \"C\" struct sapp_desc;"
       "extern \"C\" struct sapp_event;")
     :private-implementation-preamble
     `(do0
       ,log-preamble
       (do0
	(do0
	 (do0
	  "#ifdef __EMSCRIPTEN__"
	  "#define SOKOL_GLES2"
	  "#elseif"
	  "#define SOKOL_GLCORE33"
	  "#endif")

	 "#define SOKOL_IMPL"
					;"#define SOKOL_APP_IMPL"
					;"#define SOKOL_GLUE_IMPL"
	 (include<>
	  sokol_app.h
	  sokol_gfx.h
	  sokol_glue.h
	  ))
	(do0
	 "#define SOKOL_IMGUI_IMPL"
	 (include<> imgui.h
		    util/sokol_imgui.h
		    vector)
	 ))
       )
     :private-members `(#+nil (:name desc :type "std::unique_ptr<sg_desc>"
				     :init-form (new (sg_desc))
				     ))
     :private-constructor-code `(do0
				 ,(lprint :msg (format nil "constructor "))
					;(setf desc.context (sapp_sgcontext))
					;(sg_setup (desc.get))
				 )
     :private-destructor-code `(do0
				(comments "destructor"))
     :private-code-outside-class (let ((tex-w 320)
				       (tex-h 240))
				   `(do0

				     "static bool show_test_window = true;"
				     "static sg_pass_action pass_action;"
				     "static int g_tex_id = 0;"
				     (defun init ()
				       (declare (values "extern \"C\" void"))
				       ,(lprint)
				       (let ((desc (sg_desc)))
					 (setf desc.context (sapp_sgcontext))
					 (sg_setup &desc))

				       (let ((s (simgui_desc_t)))
					 (simgui_setup &s)
					 (setf (dot (ImGui--GetIO)
						    ConfigFlags)
					       (logior (dot (ImGui--GetIO)
							    ConfigFlags)
						       ImGuiConfigFlags_DockingEnable)))
				       #+nil (let ((img_desc (sg_image_desc))
						   (texw ,tex-w)
						   (texh ,tex-h)
						   (tex_pixels (std--vector<uint8_t> (* 4 320 240))))
					       ,@(loop for (e f) in `((width texw)
								      (height texh)
								      (pixel_format SG_PIXELFORMAT_RGBA8)
								      (wrap_u SG_WRAP_CLAMP_TO_EDGE)
								      (wrap_v SG_WRAP_CLAMP_TO_EDGE)
								      (min_filter SG_FILTER_LINEAR)
								      (mag_filter SG_FILTER_LINEAR))
						       collect
						       `(setf (dot img_desc ,e)
							      ,f))
					       (setf (dot img_desc
							  data
							  (aref (aref subimage 0) 0)
							  ptr)
						     (tex_pixels.data))
					       (setf (dot img_desc
							  data
							  (aref (aref subimage 0) 0)
							  size)
						     (tex_pixels.size)
						     )
					       (let ((img (sg_make_image &img_desc))
						     (tex_id img.id))
						 (setf g_tex_id tex_id)))
				       (setf
					(dot pass_action (aref colors 0) action) SG_ACTION_CLEAR
					(dot pass_action (aref colors 0) value) (curly .3s0 .7s0 .5s0 1s0))
				       )
				     (defun frame ()
				       (declare (values "extern \"C\" void"))
				       (progn
					 "static int frame_count = 0;"
					 (when (== (% frame_count (* 10 60)) 0)
					   ,(lprint))
					 (incf frame_count))
				       (let ((w (sapp_width))
					     (h (sapp_height)))
					 (simgui_new_frame
					  (curly w
						 h
						 (sapp_frame_duration)
						 (sapp_dpi_scale)))
					 (progn
					   "static float f = 0.0f;"
					   (ImGui--Text
					    (string ,(format nil "build time ~a"
							     (multiple-value-bind (second minute
											  hour date
											  month year day-of-week dst-p tx)
								 (get-decoded-time)
							       (format  nil "~2,'0d:~2,'0d:~2,'0d of ~a"
									hour minute second (nth day-of-week '("Monday" "Tuesday" "Wednesday"
													      "Thursday" "Friday" "Saturday"
													      "Sunday")))))))
					   (ImGui--Text (string "drag windows"))
					   (ImGui--SliderFloat (string "float")
							       &f 0s0 1s0)
					   #+nil (do0
						  (ImGui--Image (reinterpret_cast<void*> g_tex_id)
								(ImVec2 ,tex-w ,tex-h))
						  (ImGui--End))
					   (when (ImGui--Button (string "window"))
					     (setf show_test_window
						   !show_test_window)))
					 (do0
					  (sg_begin_default_pass &pass_action
								 w
								 h)
					  (simgui_render)
					  (sg_end_pass)
					  (sg_commit))))
				     (defun cleanup ()
				       (declare (values "extern \"C\" void"))
				       ,(lprint)
				       (simgui_shutdown)
				       (sg_shutdown))
				     (defun input (event)
				       (declare (type "const sapp_event*" event)
						(values "extern \"C\" void"))
					;,(lprint)
				       (simgui_handle_event event))
				     (defun sokol_main (argc argv)
				       (declare (type int argc)
						(type char** argv)
						(values "extern \"C\" sapp_desc"				))
				       (setf g_start_time ("std::chrono::high_resolution_clock::now"))
				       ,(lprint :msg "enter program" :vars `(argc (aref argv)))

				       (let ((s (sapp_desc)))
					 ,@(loop for (e f) in `((width 640)
								(height 480)
								(init_cb init)
								(frame_cb frame)
								(cleanup_cb cleanup)
								(event_cb input)
								(gl_force_gles2 true)
								(window_title (string "imgui docking"))
								(ios_keyboard_resizes_canvas false)
								(icon.sokol_default true)
								)
						 collect
						 `(setf (dot s ,e) ,f))
					 ,(lprint :msg "exit program")
					 (return s)))))
     )

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `index
     :headers `()
     :header-preamble `(do0
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

				(include "SokolApp.h"))
     :code `(do0
	     (do0
	      "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"
	      "std::mutex g_stdout_mutex;")

	     ))



    (write-cmake
     (format nil "~a/CMakeLists.txt" *source*)
     :code
     ;;  -s SAFE_HEAP=1 ;; maybe breaks debugging https://github.com/emscripten-core/emscripten/issues/8584
     (let ((dbg  "-O0" ; "-O0 -g4 -s ASSERTIONS=1 -s STACK_OVERFLOW_CHECK=1 -s DEMANGLE_SUPPORT=1"
	     ) ;; -gsource-map
	   (asan "" ; "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
	     )
	   (show-err ""; " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow  -Wswitch-default -Wundef   -Wunused -Wunused-parameter  -Wold-style-cast -Wsign-conversion "
	     ;;
	     ;; -Werror ;; i rather see the warnings
	     ;; "-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  " ;; not supported by emcc
	     ))
       (macrolet ((out (fmt &rest rest)
		    `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	 (out "cmake_minimum_required( VERSION 3.0 )")

	 (out "project( example LANGUAGES CXX )")

	 (out "set( CMAKE_CXX_STANDARD 17 )")
	 (out "set( CMAKE_CXX_STANDARD_REQUIRED True )")
					;(out "set( OpenCV_DIR /home/martin/src/opencv/build_wasm/ )")
					;(out "set( OpenCV_STATIC ON )")
					;(out "find_package( OpenCV REQUIRED )")
					;(out "include_directories( ${OpenCV_INCLUDE_DIRS} /home/martin/src/opencv_contrib/modules/aruco/include )")
	 (out "include_directories( ~{~a~^ ~} )" `(/home/martin/src/sokol
						   /home/martin/src/imgui))
	 (out "option( BUILD_WASM \"Build Webassembly\" ON )")
	 (progn
	   (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
					;(out "set( USE_FLAGS \"-s USE_SDL=2\" )")
					;(out "set( CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${USE_FLAGS}\" )")
	   (out "set( CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG}  ~a ~a ~a \")"
		dbg asan show-err)
	   )

	 (out "set( CMAKE_EXECUTABLE_SUFFIX \".html\" )")

	 (out "set( SRCS ~{~a~^~%~} )"
	      (append
	       (directory (format nil "~a/*.cpp" *source*))
					;(directory "/home/martin/src/opencv_contrib/modules/aruco/src/*.cpp")
	       ;; git clone -b docking --single-branch https://github.com/ocornut/imgui
	       (directory "/home/martin/src/imgui/imgui*.cpp")
	       ))

	 (out "add_executable( index ${SRCS} )")

	 #+nil (loop for e in `(imgui ;implot GLEW
				)
		     do
		     (out "find_package( ~a CONFIG REQUIRED )" e))

	 #+nil (out "target_link_libraries( index PRIVATE ~{~a~^ ~} )"
		    `("imgui::imgui"
					;"implot::implot"
					;"GLEW::GLEW"
					;"${OpenCV_LIBS}"
		      ))
	 )))
    ))

