(eval-when (:compile-toplevel :execute :load-toplevel)
					;(setf (readtable-case *readtable*) :upcase)
  (ql:quickload "spinneret")
  (ql:quickload "cl-js-generator")
  (ql:quickload "cl-cpp-generator2")
  )

(in-package :cl-js-generator)

(defun lprint (&key (msg "") (vars `()))
  `(console.log (string-backtick ,(format nil "~a ~{~a~^, ~}"
					  msg
					  (loop for v in vars
						collect
						(format nil "~a=${~a}" v v))))))

(progn
  (defparameter *source* "08source")
  (defparameter *source-dir* (format nil "example/72_emsdk/~a/" *source*))
  (write-source (format nil "~a/get_video" (asdf:system-relative-pathname
					    'cl-cpp-generator2
					    *source-dir*))
		`(let ((W 640)
		       (H 480))

		   (let ((WASM_FILE (string "./index.wasm"))
			 (mem_in null)
			 (mem_out null)
			 (instance null)
			 (memoryStates (new (WeakMap)))
			 )
		     (defun syscall (instance n args)
		       (case n
			 (45 (comments "brk")
			     (return 0))
			 (146 (comments "writev")
			      (return (instance.exports.writev_c
				       (aref args 0)
				       (aref args 1)
				       (aref args 2))))
			 (192 (comments "mmap2")
			      (do0
			       debugger
			       (let ((memory instance.exports.memory)
				     (memoryState (memoryStates.get instance))
				     (requested (aref args 1)))
				 (unless (memoryState)
				   (setf memoryState (dictionary
						      :object memory
						      :currentPosition memory.buffer.byteLength))
				   (memoryStates.set instance memoryState))
				 (let ((cur memoryState.currentPosition))
				   (when (< memory.buffer.byteLength
					    (+ cur requested))
				     (let ((need (Math.ceil
						  (/ (+ cur
							requested
							-memory.buffer.byteLength)
						     65536))))
				       (memory.grow need)))
				   (incf memoryState.currentPosition
					 requested)
				   (return cur)))))
			 (t
			  break))
		       )

		     (let ((initWASM
			    (lambda ()
			      (dot (fetch WASM_FILE)
				   (then (lambda (response)
					   (response.arrayBuffer)))
				   (then (lambda (bytes)
					   (WebAssembly.instantiate
					    bytes
					    (dictionary
					     :env (dictionary
						   ,@ (loop for i from 0 upto 6
							    appending
							    (let ((args (loop for e below i
									      and c in `(a b c d e f g)
									      collect c)))
							      `(,(make-keyword (string-upcase (format nil "_syscall~a" i)))
								 (defun ,(format nil "__syscall~a" i) (n ,@args)
								   (return (syscall instance n (list ,@args))))))))))))
				   (then (lambda (results)
					   (let ((instance results.instance)
						 (in_p (instance.exports.init_in W H))
						 (out_p (instance.exports.init_out))
						 )
					     (setf mem_in (new (Uint8Array
								instance.exports.memory.buffer
								in_p
								(/ (* W H 3)
								   2))
							       ))
					     (setf mem_out (new (Uint8Array
								 instance.exports.memory.buffer
								 in_p
								 (* W H 4)))))))
				   (catch console.error)))))))

		   #+nil (defun sat (val)
			   (when (< 255 val)
			     (return 255))
			   (when (< val 0)
			     (return 0))
			   (return (logior val
					   "0x00")))

		   (let ((bufferYUV (new (Uint8Array (/ (* 3 W H)
							2))))
			 (bufferRGBA (new (Uint8Array (* W H 4))))
			 (transformer
			  (new (TransformStream
				(curly
				 (space async transform (paren videoFrame
							       controller)
					(curly
					 (let ((copyResult (await (dot videoFrame
								       (copyTo bufferYUV))))
					       ((curly
						 stride
						 "offset:Voffset")
						(aref copyResult 1))
					       ((dictionary
						 :offset Uoffset)
						(aref copyResult 2)))

					   (do0
					    (comments "'instance' object refers to wasm module. call into wasm with allocated buffer.")
					    (instance.exports.process
					     stride
					     Voffset
					     Uoffset))

					   (do0
					    (let ((init (dictionary :timestamp videoFrame.timestamp
								    :codedWidth W
								    :codedHeight H
								    :format (string "RGBA")))


						  (newFrame (new (VideoFrame bufferRGBA
									     init)))))

					    (videoFrame.close)

					    (controller.enqueue newFrame))

					   )))




				 ))))


			 )

		     )
		   (space async
			  (defun start ()
			    (console.log (string "hello"))
			    (let ((cam (aref
					(dot (paren (space await
							   (dot navigator
								mediaDevices
								(enumerateDevices))))
					     (filter (lambda (device)
						       (return
							 (=== device.kind
							      (string "videoinput"))))))
					0))
				  ((curly deviceId)
				   cam))
			      ,(lprint :vars `(deviceId))
			      (let ((constraints (dictionary
						  :audio false
						  :video
						  (dictionary
						   :width W
						   :owheight H
						   :deviceId deviceId)))
				    (camStream (await
						(dot navigator
						     mediaDevices
						     (getUserMedia constraints))))
				    (camVideoTag (document.getElementById (string "cam")))
				    )
				,(lprint :vars `(camStream))
				(setf camVideoTag.srcObject camStream)
				,(lprint :vars `(camVideoTag))
				(let (((list videoTrack) (dot camStream
							      (getVideoTracks)))
				      (trackProcessor (new (MediaStreamTrackProcessor
							    (dictionary :track videoTrack))))
				      (trackGenerator (new (MediaStreamTrackGenerator
							    (dictionary :kind (string "video"))))))
				  ,(lprint :vars `(videoTrack))
				  (dot trackProcessor
				       readable
				       (pipeThrough transformer)
				       (pipeTo trackGenerator.writable))
				  (let ((streamAfter (new (MediaStream (list trackGenerator))))
					(modVideoTag (dot document
							  (getElementById (string "mod")))))
				    (setf modVideoTag.srcObject
					  streamAfter)))))))
		   )))

(in-package :cl-cpp-generator2)

(progn
  (progn (defparameter *source* "08source")
	 (defparameter *source-dir* (format nil "example/72_emsdk/~a/" *source*)))

					;(setf (readtable-case *readtable*) :upcase)
  (load "util.lisp")
  (write-html
   (format nil  "~a/index.html" *source*)
   :str
   (spinneret:with-html-string
       (:doctype)
     (:html
      (:head
       (:script :type "text/javascript"
		:src "get_video.js")
       (:meta :charset "utf-8")
       (:meta :http-equiv "Content-Type"
	      :content "text/html; charset=utf-8")
       (:title "test")
       )
      (:body
       (:video :id "cam")
       (:video :id "mod")
       (:canvas :id "canvas"
		:oncontextmenu "event.preventDefault()"
		:width 800
		:height 600)
       (:script :type "text/javascript"
		(:raw "var Module = { canvas: (function() { return document.getElementById('canvas'); } )() };")
		)
       (:script :src "index.js")))))

  (setf (readtable-case *readtable*) :invert)

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
					;"#define SOKOL_IMGUI_IMPL"

					;"#define SOKOL_GL_IMPL"
			   (include<> sokol_app.h
				      sokol_gfx.h
				      sokol_glue.h)


			   (include<> imgui.h
				      implot.h
				      util/sokol_gl.h
				      util/sokol_imgui.h))
			  "class State;"
			  "#define WASM_EXPORT extern \"C\" __attribute__((visibility(\"default\")))"
	       		  )
       :implementation-preamble `(do0
				  (include<> iostream
					     iomanip
					     chrono
					     thread
					     mutex
					     functional)
				  (include<> cstdlib
					     cstdint))
       :code `(do0
	       (do0
		"std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"
		"std::mutex g_stdout_mutex;")




	       (do0
		(comments "https://medium.com/docler-engineering/video-manipulation-with-webassembly-3477a0c8524d")

		"uint16_t g_frame_width, g_frame_height;"
		"uint8_t*g_input_buffer,*g_output_buffer;"
		;;"uint8_t*g_background_buffer;"
		(defun sat (val)
		  (declare (type float val)
			   (inline)
			   (values uint8_t))
		  (when (< 255 val)
		    (return 255))
		  (when (< val 0)
		    (return 0))
		  (return (static_cast<uint8_t> val)))
		,@(loop for e in `((defun init_in (w h)
				     (declare (type uint16_t w h)
					      (values "WASM_EXPORT uint8_t*"))
				     (setf g_frame_width w
					   g_frame_height h
					   g_input_buffer (static_cast<uint8_t*>
							   (malloc (/ (* w h 3)
								      2))))
				     (return g_input_buffer))
				   (defun init_out ()
				     (declare (values uint8_t*))
				     (setf
				      g_output_buffer (static_cast<uint8_t*>
						       (malloc (* g_frame_width g_frame_height 4))))
				     (return g_input_buffer))
				   (defun process (stride offset1 offset2)
				     (declare (type uint32_t stride offset1  offset2))
				     (let ((xUV 0)
					   (yUV 0))
				       (dotimes (y g_frame_height)
					 (let ((yUV (* stride (>> y 1))))
					   (dotimes (x g_frame_width)
					     (let ((xUV (>> x 1))
						   (Y (static_cast<float> (aref g_input_buffer (+ x (* y g_frame_width)))))
						   (V (static_cast<float> (- (aref g_input_buffer (+ offset1 xUV yUV))
									     128)))
						   (U (static_cast<float> (- (aref g_input_buffer (+ offset2 xUV yUV))
									     128)))
						   (R (sat (+ Y  (* 1.370705 V))))
						   (G (sat (+ Y (* -.698001 V) (* -.337633 U))))
						   (B (sat (+ Y (* 1.732446 U))))
						   (p (+ (* 4 x)
							 (* 4 y g_frame_width)))
						   )
					       (when (< (* .6 (+ R B))
							G)
						 (setf R Y
						       G Y
						       B Y))
					       ,@(loop for c in `(R G B 255) and c-i from 0
						       collect
						       `(setf (aref g_output_buffer (+ ,c-i p))
							      ,c))))))))
				   )
			collect
			e
			))



	       (defclass+ State ()
		 "public:"
		 "sg_pass_action pass_action;"
		 "sg_image img;"
		 "sgl_pipeline pip_3d;")
	       ,@(loop for e in `(init frame cleanup draw_quad)
		       collect
		       `(do0
			 ,(format nil "std::function<void()> ~a;" e)
			 (defun
			     ,(format nil "~a_cfun" e)
			     ()
			   ( ,(format nil "~a" e)))))



	       (defun draw_triangle ()
					; ,(lprint :msg "draw_triangle")
		 (sgl_defaults)
		 (do0
		  (sgl_begin_triangles)
		  (do0
		   ,@(loop for e in `((0 .5 255 0 0)
				      (-.5 -.5 0 0 255)
				      (.5 -.5 0 255 0))
			   collect
			   `(sgl_v2f_c3b ,@e)))
		  (sgl_end)))


	       #+nil (defun draw_cube ()
		       (sgl_begin_quads)
		       ,@(loop for color in
			       `((1 0 0)
				 (0 1 0)
				 (0 0 1)
				 (1 .5 0)
				 (0 .5 1)
				 (1 0 .5))
			       and  faces in
			       `(((- + - - +)
				  (+ + - + +)
				  (+ - - + -)
				  (- - - - -))

				 ((- - + - +)
				  (+ - + + +)
				  (+ + + + -)
				  (- + + - -))

				 ((- - + - +)
				  (- + + + +)
				  (- + - + -)
				  (- - - - -))

				 ((+ - + - +)
				  (+ - - + +)
				  (+ + - + -)
				  (+ + + - -))

				 ((+ - - - +)
				  (+ - + + +)
				  (- - + + -)
				  (- - - - -))

				 ((- + - - +)
				  (- + + + +)
				  (+ + + + -)
				  (+ + - - -)))

			       collect
			       `(do0
				 (sgl_c3f ,@color)
				 ,@(loop for face in faces
					 collect
					 `(sgl_v3f_t2f
					   ,@(loop for sign in face
						   collect
						   (format nil "~a1.0f" sign)))))
			       )
		       (sgl_end)
		       )

	       #+nil (defun draw_tex_cube (state)
		       (declare (type State& state))
		       (do0
			"static float frame_count = 0.0f;"
			(incf frame_count)
			(let ((a (sgl_rad frame_count))
			      (tex_rot (* .5s0 a))
			      (tex_scale  (+ 1s0 (* .5s0 (sinf a)))))
			  (sgl_matrix_mode_texture)
			  (sgl_rotate tex_rot 0s0 0s0 1s0)
			  (sgl_scale tex_scale   tex_scale 1s0)))
		       (let ((eye_x (* 6s0 (sinf a)))
			     (eye_z (* 6s0 (cosf a)))
			     (eye_y (* 3s0 (sinf a))))
			 (sgl_defaults)
			 (sgl_load_pipeline state.pip_3d)
			 (sgl_enable_texture)
			 (sgl_texture state.img)
			 (sgl_matrix_mode_projection)
			 (sgl_perspective (sgl_rad 45s0)
					  1s0 .1s0 100s0)
			 (sgl_matrix_mode_modelview)
			 (sgl_lookat eye_x
				     eye_y
				     eye_z
				     0s0 0s0 0s0
				     0s0 1s0 0s0)
			 (sgl_matrix_mode_texture)
			 (sgl_rotate tex_rot 0s0 0s0 1s0)
			 (sgl_scale tex_scale tex_scale 1s0)
			 (draw_cube)
			 )
		       )

	       (defun make_checkerboard_texture ()
		 (declare (values sg_image))
		 ,(let ((tex-w 8)
			(tex-h 8))
		    `(do0
		      ,(format nil "static uint32_t pixels[~a][~a];" tex-h tex-w)
		      (dotimes (y ,tex-h)
			(dotimes (x ,tex-w)
			  (setf (aref pixels y x)
				(? (& (logxor y x)
				      1)
				   (hex #xFFDDAAff)
				   (hex #xff112233)))))
		      (let ((smi_param (space sg_image_desc
					      (designated-initializer
					       :width ,tex-w
					       :height ,tex-h
					       (dot "" data (aref (aref subimage 0 ) 0))
					       (SG_RANGE pixels)))))
			(return (sg_make_image &smi_param )))))
		 )

	       (defun init_imgui ()
		 (let ((s (simgui_desc_t)))
                   (simgui_setup &s)
                   (setf (dot (ImGui--GetIO)
                              ConfigFlags)
			 (logior (dot (ImGui--GetIO)
                                      ConfigFlags)
				 ImGuiConfigFlags_DockingEnable)))
		 (ImPlot--CreateContext)

		 )

	       (defun sokol_main (argc argv)
		 (declare (type int argc)
			  (type char** argv)
			  (values sapp_desc))
		 (do0
		  "static State state;")

		 (setf
		  init
		  (lambda ()
		    (declare (capture &))
		    ,(lprint :msg "init")
		    (do0
		     (comments "designated initializers require c++20")
		     (let ((sg_setup_param (space sg_desc
						  (designated-initializer
						   :context (sapp_sgcontext)))))
		       (sg_setup (ref sg_setup_param))

		       (init_imgui)

		       (let ((sgl_setup_param (space sgl_desc_t
						     (curly 0))))
			 (sgl_setup
			  (ref sgl_setup_param))

			 (setf (dot state img) (make_checkerboard_texture))
			 (do0
			  (comments "sokol_gl creates shaders, pixel formats")
			  (let ((smp (space sg_pipeline_desc
					    (designated-initializer
					     :depth (designated-initializer
						     :compare SG_COMPAREFUNC_LESS_EQUAL
						     :write_enabled true
						     )
					     :cull_mode SG_CULLMODE_BACK
					     ))))
			    (setf state.pip_3d
				  (sgl_make_pipeline &smp)))
			  (setf   state.pass_action (space sg_pass_action
							   (designated-initializer
							    (aref .colors  0)
							    (designated-initializer
							     :action SG_ACTION_CLEAR
							     :value (curly .5s0 .3s0 .2s0 1s0)))))))))))

		 (setf draw_quad
		       (lambda ()
			 (declare (capture &))
					;,(lprint :msg "draw_quad")



			 (do0 (sgl_defaults)
			      (sgl_load_pipeline state.pip_3d)
			      (sgl_enable_texture)
			      (sgl_texture state.img)
			      )

			 #+nil
			 (do0
			  "static float frame_count = 0.0f;"
			  (incf frame_count)
			  (let ((a (sgl_rad frame_count))
				(tex_rot (* .5s0 a))
				(tex_scale  (+ 1s0 (* .5s0 (sinf a)))))
			    (sgl_matrix_mode_texture)
			    (sgl_rotate tex_rot 0s0 0s0 1s0)
			    (sgl_scale tex_scale   tex_scale 1s0)
			    ))

			 (do0
			  (sgl_begin_quads)
			  (do0
			   ,@(loop for e in `((-1 -1 -1 -1)
					      (1 -1 1 -1)
					      (1 1  1 1)
					      (-1 1  -1 1)
					      )
				   collect
				   `(sgl_v2f_t2f ,@(loop for q in e
							 collect (* 1s0 q)))))
			  (sgl_end))))

		 (setf frame
		       (lambda ()
			 (declare (capture &))
					;,(lprint :msg "frame")
			 (let ((ti (static_cast<float> (* 60s0 (sapp_frame_duration))))
			       (dw (sapp_width))
			       (dh (sapp_height))
			       (ww (/ dh 2))
			       (hh (/ dh 2))
			       (x0 (- (/ dw 2)
				      hh))
			       (x1 (/ dw 2))
			       (y0 0)
			       (y1 (/ dh 2)))

			   (do0 (sgl_viewport x0 y0 ww hh true)
				(draw_triangle))
			   (do0 (sgl_viewport x1 y0 ww hh true)
				(draw_quad))
			   #+nil (do0 (sgl_viewport x1 y1 ww hh true)
				      (draw_tex_cube state))
			   (sgl_viewport 0 0 dw dh true)

			   (progn
			     "static float f = .0f;"
			     "static bool show_test_window=false;"
			     (simgui_new_frame (curly dw dh
						      (sapp_frame_duration)
						      (sapp_dpi_scale)))
			     (ImGui--Text (string "drag windows"))
			     (ImGui--SliderFloat (string "float")
						 &f 0s0 1s0)

			     (do0
			      (ImGui--Image (reinterpret_cast<void*>
					     state.img.id)
					    (ImVec2 128 128))
			      )
			     (when (ImGui--Button (string "window"))
			       (setf show_test_window
				     !show_test_window))

			     (when show_test_window
			       (ImGui--ShowDemoWindow &show_test_window)
			       (ImPlot--ShowDemoWindow)
			       ))

			   (comments "sokol_gl default pass .. sgl_draw renders all commands that were submitted so far.")
			   (sg_begin_default_pass &state.pass_action
						  dw dh)
			   (sgl_draw)

			   (simgui_render)


			   (sg_end_pass)
			   (sg_commit))))



		 (setf cleanup
		       (lambda ()
			 ,(lprint :msg "cleanup")
			 (ImPlot--DestroyContext)
			 (simgui_shutdown)
			 (sgl_shutdown)
			 (sg_shutdown)
			 ))

		 (let ((sap (space sapp_desc
				   (designated-initializer
				    :init_cb init_cfun
				    :frame_cb frame_cfun
				    :cleanup_cb cleanup_cfun
				    :event_cb (lambda (e)
						(declare (type "const sapp_event*" e))
						(simgui_handle_event e))
				    :width 512
				    :height 512
				    :sample_count 1
				    :window_title (string "sokol_gl app")
				    :icon.sokol_default true
				    :gl_force_gles2 true
				    )))))
		 (return sap))

	       ))



      (write-cmake
       (format nil "~a/CMakeLists.txt" *source*)
       :code
       ;;  -s SAFE_HEAP=1 ;; maybe breaks debugging https://github.com/emscripten-core/emscripten/issues/8584
       (let ((dbg  "-O0" ; "-O0 -g4 -s ASSERTIONS=1 -s STACK_OVERFLOW_CHECK=1 -s DEMANGLE_SUPPORT=1"
	       ) ;; -gsource-map
	     (asan   "";  "-fno-omit-frame-pointer -fsanitize=address "
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
						     /home/martin/src/imgui
						     /home/martin/src/implot))

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

		 (directory "/home/martin/src/imgui/imgui*.cpp")
		 (directory "/home/martin/src/implot/implot*.cpp")
		 ))
	   (out "add_executable( index ${SRCS} )")

	   (out "if( EMSCRIPTEN )")
	   (out "set( CMAKE_EXECUTABLE_SUFFIX \".html\" )")
					;(out "option( BUILD_WASM \"Build Webassembly\" ON )")
	   (out "set_target_properties( index PROPERTIES LINK_FLAGS \"-s WASM=1 -s ALLOW_MEMORY_GROWTH=1\" ) ")
	   (out "else()")
	   (out "set( CMAKE_C_COMPILER clang )")
	   (out "set( CMAKE_CXX_COMPILER clang++ )")
	   (out "target_link_libraries( index PRIVATE ~{~a~^ ~} )"
		;; don't forget pthread, without we can get weird bugs
		`(X11 GL Xext Xi Xcursor dl pthread))
	   (out "endif()")




	   #+nil (loop for e in `(imgui ;implot GLEW
				  )
		       do
		       (out "find_package( ~a CONFIG REQUIRED )" e))


	   )))
      )))

