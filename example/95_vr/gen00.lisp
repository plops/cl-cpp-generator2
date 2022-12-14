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
    (let ((name `App))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include "bla.h"))
       :implementation-preamble `(do0
				  (include "bah.h"))
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
			     (vrapi_GetystemPropertyInit
			      java
			      VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_WIDTH)
			     (vrapi_GetystemPropertyInit
			      java
			      VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_HEIGHT)))
		     (window nullptr)
		     (ovr nullptr)
		     (back_button_down_previous_frame false)
		     (frame_index 0))
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

    (let ((name `Renderer))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include "bla.h"))
       :implementation-preamble `(do0
				  (include "bah.h"))
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "std::vector<Framebuffer> framebuffers;"
		 "Program program;"
		 "Geometry geometry;"
		 (defmethod ,name (width height)
		   (declare
					;  (explicit)
		    (type GLsizei width height)
		    (construct
		     (program (Program))
		     (geometry (Geometry)))
		    (values :constructor))
		   (dotimes (i VRAPI_FRAME_LAYER_EYE_MAX)
		     (framebuffers.push_back
			   (Framebuffer width height))))
		#+nil (defmethod ,(format nil "~~~a" name) ()
		   (declare
					;  (explicit)
		    
		    (construct
		     )
		    (values :constructor))
			)))))

    (let ((name `Cube))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include "bla.h"))
       :implementation-preamble `(do0
				  (include "bah.h"))
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
			  (include "bla.h"))
       :implementation-preamble `(do0
				  (include "bah.h"))
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "std::array<float,4> position;"
		 "std::array<float,4> color;"
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
			  (include "bla.h"))
       :implementation-preamble `(do0
				  (include "bah.h"))
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
			  (include "bla.h"))
       :implementation-preamble `(do0
				  (include "bah.h")
				  "extern static const std::array<AttribPointer,2> ATTRIB_POINTERS;")
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "GLuint vertex_array, vertex_buffer, index_buffer;"
		 "Cube cube;"
		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    (construct)
		    (values :constructor))
		   (glGenVertexArrays 1
				      &vertex_array)
		   (glBindVertexArrays vertex_array)
		   (glGenBuffers 1 &vertex_buffer)
		   (glBufferData GL_ARRAY_BUFFER
				 cube.vertices.size
				 (cube.vertices.data)
				 GL_STATIC_DRAW)
		   (let ((i 0))
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
		   (glBufferData GL_ELEMENT_ARRAY_BUFFER
				 (dot cube
				      indices
				      (size))
				 cube.indices
				 GL_STATIC_DRAW)
		   (glBindVertexArray 0))
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
			  (include "bla.h"))
       :implementation-preamble `(do0
				  (include "bah.h"))
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
		     (diplay (eglGetDisplay EGL_DEFAULT_DISPLAY)))
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
							  (declare (type aute renderable_type))
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
							 (declare (type auto i))
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
					  (let (
						(check (lambda (attrib)
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
		    
			  (construct
			   )
			  (values :constructor))			 )
		 )
	       )))
    
    (let ((name `Framebuffer))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include "bla.h"))
       :implementation-preamble `(do0
				  (include "bah.h"))
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

    (let ((name `Program))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include "bla.h"))
       :implementation-preamble `(do0
				  (include "bah.h"))
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
		    
			  (construct)
			  (values :constructor)))))))
    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"main.cpp"
		       *source-dir*))
     `(do0


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

       (do0
	(include <spdlog/spdlog.h>)
					


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
		<cstdin>
		<cstdlib>
		<unistd.h>)

       (do0
	"#define FMT_HEADER_ONLY"

	(include "core.h"))

       (let ((ATTRIB_POINTERS
	       (curly (AttribPointer 3 GL_FLOAT GL_FALSE (sizeof Vertex)
				     (reinterpret_cast<GLvoid*> (offsetof Vertex position)))
		      (AttribPointer 3 GL_FLOAT GL_FALSE (sizeof Vertex)
				     (reinterpret_cast<GLvoid*> (offsetof Vertex color))))))
	 (declare (type "static const std::array<AttribPointer,2>" ATTRIB_POINTERS)))

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
	    (-> (deref java.Vm)
		(AttachCurrentThread java.Vm
				     &java.Env
				     nullptr))))

	 (do0
	  ,(lprint :msg "initialize vr api")
	  (let ((init_params (vrapi_DefaultInitParams
			      &java)))
	    (unless (== VRAPI_INITIALIZE_SUCCESS
			(vrapi_Initialize &init_params))
	      ,(lprint :msg "can't initialize vr api")
	      (std--exit 1))))

	 (do0
	  (let ((app (App &app
			  &java)))))
	 )

       ))

    #+nil
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
	    (show-err "-Wall -Wextra"	;
					;" -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused"
					;"-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "

	      ))
	(macrolet ((out (fmt &rest rest)
		     `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	  (out "cmake_minimum_required( VERSION 3.0 FATAL_ERROR )")
	  (out "project( mytest LANGUAGES CXX )")

	  ;;(out "set( CMAKE_CXX_COMPILER clang++ )")
					;(out "set( CMAKE_CXX_COMPILER g++ )")
	  (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	  (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a ~a \")" dbg asan show-err short-file)
	  (out "set (CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ~a ~a ~a ~a \")" dbg asan show-err short-file)
	  (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 23 )")



	  (out "set( SRCS ~{~a~^~%~} )"
	       (append
		(directory (format nil "~a/*.cpp" *full-source-dir*))
		))

	  (out "add_executable( mytest ${SRCS} )")




	  (out "set_property( TARGET mytest PROPERTY CXX_STANDARD 20 )")

	  (out "target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )")

	  (out "find_package( PkgConfig REQUIRED )")
	  (out "pkg_check_modules( spdlog REQUIRED spdlog )")


	  (out "target_include_directories( mytest PRIVATE
/usr/local/include/
/home/martin/src/popl/include/
)")
	  #+nil (progn
		  (out "add_library( libnc SHARED IMPORTED )")
		  (out "set_target_properties( libnc PROPERTIES IMPORTED_LOCATION /home/martin/stage/cl-cpp-generator2/example/88_libnc/dep/libnc-2021-04-24/libnc.so
 )")
		  )

	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `(spdlog
		 pthread
		 X11
		 Xext
		 ))

	  #+nil
	  (out "target_compile_options( mytest PRIVATE ~{~a~^ ~} )"
	       `())

					;(out "target_precompile_headers( mytest PRIVATE fatheader.hpp )")
	  ))
      )))

