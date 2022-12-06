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
    (defparameter *source-dir* #P"example/92_pipewire/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))
    (ensure-directories-exist *full-source-dir*)
    (load "util.lisp")
    #+nil (let ((name `AGameCharacter)
		)
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
			  (construct
			   (Camera
			    3))
			  (values :constructor))
			 )
		       )
		     )))


    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"main.cpp"
		       *source-dir*))
     `(do0
       "#define _REENTRANT"

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
					;(include <popl.hpp>)


	)
       (space "extern \"C\""
	      (progn
		(include <pipewire/pipewire.h>
			 <gio/gio.h>)
		" "))

       (defun main (argc argv)
	 (declare (type int argc)
		  (type char** argv)
		  (values int))
	 "(void)argv;"
	 ,(lprint :msg "start" :vars `(argc))
	 (pw_init nullptr nullptr)
	 ;; https://github.com/obsproject/obs-studio/blob/6fb83abaeb711d1e12054d2ef539da5c43237c58/plugins/linux-pipewire/screencast-portal.c
	 (progn
	   "g_autoptr(GError) error = nullptr;"
	   (let ((*connection (g_bus_get_sync
			       G_BUS_TYPE_SESSION
			       nullptr
			       &error)))
	     (when error
	       ,(lprint :msg "d-bus connection failed"
			:vars `(error->message))
	       (return -1))
	     (progn

	       (let ((*screencast_proxy
		      (g_dbus_proxy_new_sync
		       connection
		       G_DBUS_PROXY_FLAGS_NONE
		       nullptr
		       (string "org.freedesktop.portal.Desktop")
		       (string "/org/freedesktop/portal/desktop")
		       (string "org.freedesktop.portal.ScreenCast")
		       nullptr &error)))
		 (when error
		   ,(lprint :msg "d-bus proxy retrieval failed"
			    :vars `(error->message))
		   (return -1))
		 (let ((cachedSourceTypes
			(g_dbus_proxy_get_cached_property
			 screencast_proxy
			 (string "AvailableSourceTypes")))
		       )
		   ;; https://github.com/obsproject/obs-studio/blob/7c493ea03572ddf6e34459b49291904e195ba8ea/plugins/linux-pipewire/screencast-portal.c
		   (when cachedSourceTypes
		     (let ((availableCaptureTypes (g_variant_get_uint32
						   cachedSourceTypes))
			   (CAPTURE_TYPE_MONITOR "1<<0")
			   (CAPTURE_TYPE_WINDOW "1<<1")
			   (CAPTURE_TYPE_VIRTUAL "1<<2")
			   (desktopCaptureAvailable
			    (!= 0 (logand
				   availableCaptureTypes
				   CAPTURE_TYPE_MONITOR
				   )))
			   (windowCaptureAvailable
			    (!= 0 (logand
				   availableCaptureTypes
				   CAPTURE_TYPE_WINDOW
				   ))))
		       ,(lprint :vars `(availableCaptureTypes
					desktopCaptureAvailable
					windowCaptureAvailable))
		       (do0
			(comments "init capture")
			(let ((cancellable (g_cancellable_new))
			      (connection (portal_get_dbus_connection)))
			  (unless connection
			    ,(lprint :msg "can't get connection")
			    (return -1))
			  ;; (get_screencast_portal_proxy)
			  ;; (ensure_screencast_portal_proxy)
			  ;; create_session
			  ;;  portal_create_request_path
			  ;;  portal_create_session
			  "char*request_path,*request_token;"
			  (portal_create_request_path &request_path &request_token)
			  ,(lprint :vars `(request_path
					   request_token))
			  )))
		     )))))
	   )
	 ;; remotedesktyeyop.py https://gitlab.gnome.org/-/snippets/39


	 #+nil (let ((availableCaptureTypes
		      (get_available_capture_types))))
	 #+nil
	 (let (
	       (context (pw_context_new pw nullptr 0))
	       (display (pw_wayland_context_get_display context))
	       (stream (pw_stream_new_with_listener
			context
			(string "screen-capture")
			nullptr
			0))
	       )
	   (pw_stream_set_format stream
				 PW_FORMAT_RGB
				 1920 1080 0
				 ))
	 )

       ))


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
	  (out "pkg_check_modules( pipewire REQUIRED libpipewire-0.3 )")

	  (out "target_include_directories( mytest PRIVATE
/usr/local/include/
/home/martin/src/popl/include/
/usr/include/pipewire-0.3
/usr/include/spa-0.2
/usr/include/glib-2.0
/usr/lib64/glib-2.0/include
/usr/include/sysprof-4  )")
	  #+nil (progn
		  (out "add_library( libnc SHARED IMPORTED )")
		  (out "set_target_properties( libnc PROPERTIES IMPORTED_LOCATION /home/martin/stage/cl-cpp-generator2/example/88_libnc/dep/libnc-2021-04-24/libnc.so
 )")
		  )

	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `(spdlog
		 pthread
		 glib-2.0
		 gio-2.0
		 pipewire-0.3))

	  #+nil
	  (out "target_compile_options( mytest PRIVATE ~{~a~^ ~} )"
	       `())

					;(out "target_precompile_headers( mytest PRIVATE fatheader.hpp )")
	  ))
      )))

