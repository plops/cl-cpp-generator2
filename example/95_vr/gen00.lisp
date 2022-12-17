(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-change-case")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

(let ((log-preamble
       `(do0
	 (include			;<iostream>
					;<iomanip>
					;<chrono>
					;<thread>
	  <spdlog/spdlog.h>
	  ))))

  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file
    (defparameter *source-dir* #P"example/95_vr/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))
    (ensure-directories-exist *full-source-dir*)
    (load "util.lisp")

    (defparameter *includes* `(do0
					;(include <spdlog/spdlog.h>)
			       (include
					;<tuple>
					;<mutex>
					;<thread>
				<iostream>
					;<iomanip>
					;<chrono>
					;<cassert>
					;  <memory>
				)

			       #+nil
			       (include
				"App.h"
				"AttribPointer.h"
				"core.h"
				"Cube.h"
				"Egl.h"
				"format.h"
				"format-inl.h"
				"Framebuffer.h"
				"Geometry.h"
				"Program.h"
				"Renderer.h"
				"Vertex.h"


				)

			       (include "VrApi.h"
					"VrApi_Helpers.h"
					"VrApi_Input.h"
					"VrApi_SystemUtils.h"
					"android_native_app_glue.h"
					<EGL/egl.h>
					<EGL/eglext.h>
					<GLES3/gl3.h>
					<android/log.h>
					<android/window.h>
					;<cstdin>
					<vector>
					<cstdlib>
					<unistd.h>)))

    (let ((name `App))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include "Egl.h"
				   "Renderer.h")
			  #+nil(include
				"AttribPointer.h"
				"core.h"
				"Cube.h"
				"Egl.h"
				"format.h"
				"format-inl.h"
				"Framebuffer.h"
				"Geometry.h"
				"Program.h"
				"Renderer.h"
				"Vertex.h")
			  ,*includes*)
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" name))
				  ,*includes*)
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "ovrJava* java;"
		 "bool resumed;"
		 "Egl egl;"
		 "Renderer renderer;"
		 "ANativeWindow* window;"
		 "ovrMobile* ovr;"
		 "bool back_button_down_previous_frame;"
		 "uint64_t frame_index;"

		 (defmethod ,name (java)
		   (declare
					;  (explicit)
		    (type ovrJava* java)
		    (construct
		     (java java)
		     (resumed false)
		     (egl (Egl))
		     (renderer
		      (Renderer
		       (vrapi_GetSystemPropertyInt
			java
			VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_WIDTH)
		       (vrapi_GetSystemPropertyInt
			java
			VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_HEIGHT)))
		     (window nullptr)
		     (ovr nullptr)
		     (back_button_down_previous_frame false)
		     (frame_index 0))
		    (values :constructor)))
		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
			  (values :constructor)))))))


    (let ((name `Renderer))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include ;"Framebuffer.h"
			   "Program.h"
			   "Geometry.h"
			   )
			  ,*includes*)
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" name))
				  ,*includes*)
       :code `(do0
	       (defclass ,name ()
		 "public:"
					;"std::vector<Framebuffer> framebuffers;"
		 "Program program;"
		 "Geometry geometry;"
		 (defmethod ,name (width height)
		   (declare
					;  (explicit)
		    (type GLsizei width height)
		    (construct
		     (program (Program))
		     (geometry (Geometry))
		     )
		    (values :constructor))
		   #+nil   (dotimes (i VRAPI_FRAME_LAYER_EYE_MAX)
			     (framebuffers.push_back
			      (Framebuffer width height))))
		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
			  (construct)
			  (values :constructor)))))))
    #+nil
    (let ((name `Cube))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  ,*includes*)
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" name))
				  ,*includes*
				  )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "std::vector<Vertex> vertices;"
		 "std::vector<unsigned short> indices;"
		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    (construct
		     (vertices
		      (curly
		       ,@(loop for e in `((-1 1 -1) (1 0 1)
					  (1 1 -1)  (0 1 0)
					  (1 1 1)   (0 0 1)
					  (-1 1 1)  (1 0 0)
					  (-1 -1 -1)(0 0 1)
					  (-1 -1 1) (0 1 0)
					  (1 -1 1)  (1 0 1)
					  (1 -1 -1) (1 0 0))
			       collect
			       `(curly ,@e))))
		     (indices
		      (curly 0 1 2 2 0 3
			     4 6 5 6 4 7
			     2 6 7 7 1 2
			     0 4 5 5 3 0
			     3 5 6 6 2 3
			     0 1 7 7 4 0)))
		    (values :constructor))))
	       )))

    (let ((name `Vertex))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include <array>)
			  ,*includes*)
       :implementation-preamble `(do0
				  ,*includes*
				  (include ,(format nil "~a.h" name))
				  )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "std::array<float,4> position;"
		 "std::array<float,4> color;"
		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    (construct )
		    (values :constructor)))
		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
					;  (explicit)
		    	  (construct)
			  (values :constructor)))))))


    (let ((name `AttribPointer))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  ,*includes*)
       :implementation-preamble `(do0
				  ,*includes*
				  (include ,(format nil "~a.h" name))
				  )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "GLint size;"
		 "GLenum type;"
		 "GLboolean normalized;"
		 "GLsizei stride;"
		 "const GLvoid* pointer;"
		 (defmethod ,name (size type normalized stride pointer)

		   (declare
		    (type GLint size)
		    (type GLenum type)
		    (type GLboolean normalized)
		    (type GLsizei stride)
		    (type "const GLvoid*" pointer)
					;  (explicit)
		    (construct (size size)
			       (type type)
			       (normalized normalized)
			       (stride stride)
			       (pointer pointer))
		    (values :constructor)))
		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
					;  (explicit)

			  (construct
			   )
			  (values :constructor))

			 )
		 )
	       )))

    (let ((name `Geometry))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  ,*includes*)
       :implementation-preamble `(do0
				  ,*includes*
				  (include ,(format nil "~a.h" name))

				  (include "DataExtern.h")
				  )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "GLuint vertex_array, vertex_buffer, index_buffer;"
					;"Cube cube;"
		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    (construct)
		    (values :constructor))
		   (glGenVertexArrays 1
				      &vertex_array)
		   (glBindVertexArray vertex_array)
		   (glGenBuffers 1 &vertex_buffer)
		   #+nil (glBufferData GL_ARRAY_BUFFER
				       cube.vertices.size
				       (cube.vertices.data)
				       GL_STATIC_DRAW)
		   #+nl (do0 (let ((i 0))
			       (for-range (attrib ATTRIB_POINTERS)
					  (glEnableVertexAttribArray i)
					  (glVertexAttribPointer
					   i attrib.size attrib.type
					   attrib.normalized attrib.stride
					   attrib.pointer)
					  (incf i)))
			     (glGenBuffers 1 &index_buffer)
			     (glBindBuffer GL_ELEMENT_ARRAY_BUFFER
					   index_buffer)
			     #+nil (do0 (glBufferData GL_ELEMENT_ARRAY_BUFFER
						      (dot cube
							   indices
							   (size))
						      cube.indices
						      GL_STATIC_DRAW)
					(glBindVertexArray 0))))
		 (defmethod ,(format nil "~~~a" name) ()
		   (declare
					;  (explicit)
		    (construct)
		    (values :constructor))
		   (glDeleteBuffers 1 &index_buffer)
		   (glDeleteBuffers 1 &vertex_buffer)
		   (glDeleteVertexArrays 1 &vertex_array))))))


    (let ((name `Egl))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  ,*includes*)
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" name))
				  ,*includes*)
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "EGLDisplay display;"
		 "EGLContext context;"
		 "EGLSurface surface;"
		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    (construct
		     (display (eglGetDisplay EGL_DEFAULT_DISPLAY)))
		    (values :constructor))
		   (when (== EGL_NO_DISPLAY
			     display)
		     ,(lprint :msg "can't get egl display"))
		   (when (== EGL_FALSE
			     (eglInitialize display
					    nullptr
					    nullptr))
		     ,(lprint :msg "can't initialize egl display"))
		   (do0
		    ,(lprint :msg "get number of egl configs ..")
		    (let ((numConfigs ((lambda ()
					 (declare (capture "&"))
					 (let ((n (EGLint 0)))
					   (when (== EGL_FALSE
						     (eglGetConfigs display nullptr
								    0 &n))
					     ,(lprint :msg "cant get number of egl configs"))
					   (return n)))))
			  (configs (std--vector<EGLConfig> numConfigs)))
		      (when (== EGL_FALSE
				(eglGetConfigs display (configs.data) numConfigs &numConfigs))
			,(lprint :msg "cant get egl configs"))
		      ))
		   (do0
		    ,(lprint :msg "choose egl config")
		    (let ((foundConfig (EGLConfig nullptr)))
		      (for-range (config configs)
				 (let ((renderable_type ((lambda (renderable_type)
							   (declare (type auto renderable_type)
								    (capture "&"))
							   (when (== EGL_FALSE
								     (eglGetConfigAttrib display
											 config
											 EGL_RENDERABLE_TYPE
											 &renderable_type))
							     ,(lprint :msg "cant get EGL config renderable type"))
							   (return renderable_type))
							 (EGLint 0))))
				   (when (or ,@(loop for e in `(EGL_OPENGL_ES3_BIT_KHR
								EGL_PBUFFER_BIT
								EGL_WINDOW_BIT)
						     collect
						     `(== 0
							  (logand renderable_type
								  ,e))
						     ))
				     continue)
				   (let ((surface_type ((lambda (i)
							  (declare (type auto i)
								   (capture "&"))
							  (when (== EGL_FALSE
								    (eglGetConfigAttrib
								     display
								     config
								     EGL_SURFACE_TYPE
								     &i))
							    ,(lprint :msg "cant get surface config type"))
							  (return i))
							(EGLint 0))))
				     ,(let ((l-attrib `((red-size 8)
							(green-size 8)
							(blue-size 8)
							(alpha-size 8)
							(depth-size 0)
							(stencil-size 0)
							(samples 0))))
					`(progn
					   (let ((check (lambda (attrib)
							  (declare (type auto attrib)
								   (values auto)
								   (capture "&"))
							  (let ((value (EGLint 0)))
							    (when (== EGL_FALSE
								      (eglGetConfigAttrib display config attrib &value))
							      ,(lprint :msg "cant get config attrib")))
							  (return value))))
					     (when (and ,@(remove-if
							   #'null
							   (loop for (e f) in l-attrib and i from 0
								 collect
								 (let ((attrib (cl-change-case:constant-case (format nil "egl-~a" e))))
								   (unless (eq 0 f)
								     `(<= ,f (check ,attrib)))))))
					       (setf foundConfig config)))))))))))
		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
					;  (explicit)
			  (construct)
			  (values :constructor)))))))
    #+nil
    (let ((name `Framebuffer))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  ,*includes*)
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" name))
				  ,*includes*)
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    (construct)
		    (values :constructor))
		   )
		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
					;  (explicit)

			  (construct
			   )
			  (values :constructor))

			 )
		 )
	       )))
    #-nil
    (let ((name `Program))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  "#pragma once"

			  ,*includes*
			  (include "DataTypes.h"
				   "DataExtern.h"
				   <array>))

       :implementation-preamble `(do0
				  (do0
				   "#define FMT_HEADER_ONLY"
				   (include "core.h"
					    "format.h"
					    <android/log.h>))

				  (include ,(format nil "~a.h" name))
				  ,*includes*)
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "GLuint program;"
		 "std::array<GLint,UNIFORM_END> uniform_locations;"
		 (defmethod compileShader (type str)
		   (declare (type GLenum type)
			    (type "std::string" str)
			    (values GLuint))
		   (let ((shader (glCreateShader type)))
		     (let ((c_str (str.c_str)))
		       (glShaderSource shader
				       1
				       (ref c_str)
				       nullptr))
		     (glCompileShader shader)
		     (progn
		       (let ((status (GLint 0)))
			 (glGetShaderiv shader
					GL_COMPILE_STATUS
					&status)
			 (when (== GL_FALSE
				   status)
			   (let ((length (GLint 0)))
			     (glGetShaderiv shader
					    GL_INFO_LOG_LENGTH
					    &length)
			     (let ((log (std--vector<char> length)))
			       (glGetShaderInfoLog shader
						   length
						   nullptr
						   (log.data))
			       (let ((logstr (std--string (std--begin log)
							  (std--end log))))
				 ,(lprint :msg "cant compile shader"
					  :vars `(logstr))
				 (std--exit -1)))))
			 (return shader)))))
		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    (construct (program (glCreateProgram)))
		    (values :constructor))
		   (let ((FRAGMENT_SHADER
			  (std--string
			   (string-r
			    ,(cl-cpp-generator2::emit-c
			      :code
			      `(do0
				"#version 300 es"
				"in lowp vec3 vColor;"
				"out lowp vec4 outColor;"
				(defun main ()
				  (setf outColor
					(vec4 vColor 1s0))))))))
			 (VERTEX_SHADER
			  (std--string
			   (string-r
			    ,(cl-cpp-generator2::emit-c
			      :code
			      `(do0
				"#version 300 es"
				"in vec3 aPosition;"
				"in vec3 aColor;"
				"uniform mat4 uModelMatrix;"
				"uniform mat4 uViewMatrix;"
				"uniform mat4 uProjectionMatrix;"
				"out vec3 vColor;"
				(defun main ()
				  (setf gl_Position
					(* uProjectionMatrix
					   (* uViewMatrix
					      (* uModelMatrix
						 (vec4 (* aPosition 1s0)
						       1s0))))
					vColor aColor)))))))
			 (vertexShader (compileShader GL_VERTEX_SHADER
						      VERTEX_SHADER))
			 (fragmentShader (compileShader GL_FRAGMENT_SHADER
							FRAGMENT_SHADER)))
		     (glAttachShader program
				     vertexShader)
		     (glAttachShader program
				     fragmentShader)
		     (let ((i 0))
		       (for-range (name ATTRIB_NAMES)
				  (glBindAttribLocation program
							i
							(name.c_str))
				  (incf i)))
		     (glLinkProgram program)
		     (progn
		       (let ((status (GLint 0)))
			 (glGetProgramiv program
					 GL_LINK_STATUS
					 &status)
			 (when (== GL_FALSE
				   status)
			   (let ((length (GLint 0)))
			     (glGetProgramiv program
					     GL_INFO_LOG_LENGTH
					     &length)
			     (let ((log (std--vector<char> length)))
			       (glGetProgramInfoLog program
						    length
						    nullptr
						    (log.data))
			       (let ((logstr (std--string (std--begin log)
							  (std--end log))))
				 ,(lprint :msg "cant compile shader"
					  :vars `(logstr))
				 (std--exit -1)))))
			 ))
		     (dotimes (i UNIFORM_END)
		       (setf (aref uniform_locations i)
			     (glGetUniformLocation program
						   (dot (aref UNIFORM_NAMES i)
							(c_str)))))))
		 (defmethod ,(format nil "~~~a" name) ()
	       	   (declare
		    (values :constructor))
		   (glDeleteProgram program))))))

    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"DataExtern.h"
		       *source-dir*))
     `(do0
       "#pragma once"
       (include <vector>
		<string>
		"AttribPointer.h")
       "extern const std::vector<std::string> ATTRIB_NAMES;"
       "extern const std::vector<std::string> UNIFORM_NAMES;"
       "extern const std::array<AttribPointer,2> ATTRIB_POINTERS;"
       ))
    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"DataTypes.h"
		       *source-dir*))
     `(do0
       (include <vector>
		<string>)

       (space enum attrib
	      (curly
	       ATTRIB_BEGIN
	       (= ATTRIB_POSITION ATTRIB_BEGIN)
	       ATTRIB_COLOR
	       ATTRIB_END))
       (space enum uniform
	      (curly
	       UNIFORM_BEGIN
	       (= UNIFORM_MODEL_MATRIX UNIFORM_BEGIN)
	       UNIFORM_VIEW_MATRIX
	       UNIFORM_PROJECTION_MATRIX
	       UNIFORM_END))))
    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"main.cpp"
		       *source-dir*))
     `(do0



       ,*includes*
       #+nil (do0
					;(include <spdlog/spdlog.h>)
	      (include
					;<tuple>
					;<mutex>
					;<thread>
	       <iostream>
					;<iomanip>
					;<chrono>
					;<cassert>
					;  <memory>
	       )


					; (include "AttribPointer.h")
	      (include "VrApi.h"
		       "VrApi_Helpers.h"
		       "VrApi_Input.h"
		       "VrApi_SystemUtils.h"
		       "android_native_app_glue.h"
		       <EGL/egl.h>
		       <EGL/eglext.h>
		       <GLES3/gl3.h>
		       <android/log.h>
		       <android/window.h>
					;<cstdin>
		       <cstdlib>
		       <unistd.h>

		       ))
       (include "App.h"
		"Vertex.h")

       (do0
	"#define FMT_HEADER_ONLY"

	(include "core.h"))

       "const std::vector<std::string> ATTRIB_NAMES = {\"aPosition\",\"aColor\"};"
       "const std::vector<std::string> UNIFORM_NAMES = {\"uModelMatrix\",\"uViewMatrix\",\"uProjectionMatrix\"};"

       (let ((ATTRIB_POINTERS
	      (curly (AttribPointer 3 GL_FLOAT GL_FALSE (sizeof Vertex)
				    (reinterpret_cast<GLvoid*> (offsetof Vertex position)))
		     (AttribPointer 3 GL_FLOAT GL_FALSE (sizeof Vertex)
				    (reinterpret_cast<GLvoid*> (offsetof Vertex color))))))
	 (declare (type "const std::array<AttribPointer,2>" ATTRIB_POINTERS)))

       (defun android_main (android_app)
	 (declare (type android_app* android_app))
	 (ANativeActivity_setWindowFlags
	  android_app->activity
	  AWINDOW_FLAG_KEEP_SCREEN_ON
	  0)
	 (do0
	  ,(lprint :msg "attach current thread"
		   :level "info")
	  (let ((java (ovrJava)))
	    (setf java.Vm android_app->activity->vm)
	    (-> java.Vm			;"(*java.Vm)"
		(AttachCurrentThread	;java.Vm
		 &java.Env
		 nullptr))))
	 (do0
	  ,(lprint :msg "initialize vr api")
	  (let ((init_params (vrapi_DefaultInitParms
			      &java)))
	    (unless (== VRAPI_INITIALIZE_SUCCESS
			(vrapi_Initialize &init_params))
	      ,(lprint :msg "can't initialize vr api")
	      (std--exit 1))))
	 (do0
	  (let ((app (App
		      &java))))))))

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
	    (short-file "" ;"-ffile-prefix-map=/home/martin/stage/cl-cpp-generator2/example/86_glbinding_av/source01/="
	      )
	    (show-err "-Wall -Wextra -std=c++20"	;  -march=arm-v8a
					;" -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused"
					;"-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "

	      ))
	(let ((program-name "main"))
	  (macrolet ((out (fmt &rest rest)
		       `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	    (out "cmake_minimum_required( VERSION 3.0 FATAL_ERROR )")
	    (out "project( ~a LANGUAGES CXX )" program-name)

            (out "set( CMAKE_C_COMPILER /home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang )")
	    (out "set( CMAKE_CXX_COMPILER /home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ )")

					;(out "set( CMAKE_CXX_COMPILER g++ )")
	    (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	    (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a ~a \")" dbg asan show-err short-file)
	    (out "set (CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ~a ~a ~a ~a \")" dbg asan show-err short-file)
	    (out "set (CMAKE_C_FLAGS \"${CMAKE_C_FLAGS} ~a ~a ~a ~a \")" dbg asan show-err short-file)
	    (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 23 )")



	    (out "set( SRCS ~{~a~^~%~} )"
		 (append
		  (directory (format nil "~a/*.cpp" *full-source-dir*))
		  (directory (format nil "~a/*.c" *full-source-dir*))
		  ))

					;(out "add_executable( mytest ${SRCS} )")
	    (out "add_library (~a SHARED ${SRCS} )" program-name)




	    (out "set_property( TARGET ~a PROPERTY CXX_STANDARD 20 )" program-name)

					;(out "target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )")

					;(out "find_package( PkgConfig REQUIRED )")
					;(out "pkg_check_modules( spdlog REQUIRED spdlog )")


	    (out "target_include_directories( ~a PRIVATE
/home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/
/home/martin/quest2/ovr/VrApi/Include
)" program-name)
	    ;;/home/martin/quest2/ovr/VrApi/Libs/Android/arm64-v8a/Debug
					; /platforms/android-26/arch-arm64/usr/lib
	    (progn
	      (out "add_library( vrapi SHARED IMPORTED )")
	      (out "set_target_properties( vrapi PROPERTIES IMPORTED_LOCATION /home/martin/quest2/ovr/VrApi/Libs/Android/arm64-v8a/Debug/libvrapi.so
 )")
	      )
	    #+nil
	    (progn
	      (out "add_library( c++shared SHARED IMPORTED )")
	      (out "set_target_properties( c++shared PROPERTIES IMPORTED_LOCATION /home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so
 )")
	      )

	    (out "target_link_libraries( ~a PRIVATE ~{~a~^ ~} )"
		 program-name
		 `(android
		   log
		   vrapi
					;c++shared
		   ))

	    #+nil
	    (out "target_compile_options( ~a PRIVATE ~{~a~^ ~} )"
		 program-name `())

					;(out "target_precompile_headers( ~a PRIVATE fatheader.hpp )" program-name)
	    )))
      )))

