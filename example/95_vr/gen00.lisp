(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
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
		  (dotimes (i VRAPI_FRAME_LAYER_EYE_MAX)
		     (setf (aref framebuffers i)
			   (Framebuffer width height)))
		   )
		 )
	       )))

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
				  (include "bah.h"))
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
				      vertex_array)
		   (glBindVertexArrays vertex_array)
		   (glGenBuffers 1 vertex_buffer)
		   (glBufferData GL_ARRAY_BUFFER
				 cube.vertices.size
				 (cube.vertices.data)
				 GL_STATIC_DRAW)
		   )
		 #+nil (defmethod ,(format nil "~~~a" name) ()
		   (declare
					;  (explicit)
		    
		    (construct
		     )
		    (values :constructor))
		  (dotimes (i VRAPI_FRAME_LAYER_EYE_MAX)
		     (setf (aref framebuffers i)
			   (Framebuffer width height)))
		   )
		 )
	       )))

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
		 "GLuint vertex_array, vertex_buffer, index_buffer;"
		 "Cube cube;"
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
		  (dotimes (i VRAPI_FRAME_LAYER_EYE_MAX)
		     (setf (aref framebuffers i)
			   (Framebuffer width height)))
		   )
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
		 "GLuint vertex_array, vertex_buffer, index_buffer;"
		 "Cube cube;"
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
		  (dotimes (i VRAPI_FRAME_LAYER_EYE_MAX)
		     (setf (aref framebuffers i)
			   (Framebuffer width height)))
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
		 "GLuint vertex_array, vertex_buffer, index_buffer;"
		 "Cube cube;"
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
			 (dotimes (i VRAPI_FRAME_LAYER_EYE_MAX)
			   (setf (aref framebuffers i)
				 (Framebuffer width height)))
			 )
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

