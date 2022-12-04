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
    (defparameter *source-dir* #P"example/91_wayland/source00/")
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
       #+nil
       (do0
	(include <glbinding/glbinding.h>
					;<glbinding/gl/gl.h>
		 <glbinding/gl32core/gl.h>
		 )
					;"using namespace gl;"
	"using namespace gl32core;"
	"using namespace glbinding;")
       (space "extern \"C\""
	      (progn
		(include <wayland-client.h>
			 <sys/mman.h>
			 <sys/stat.h>
			 <fcntl.h>
			 <errno.h>)
		" "))

       ,(let ((l `(wl_compositor wl_shm wl_output)))
	  `(do0
	    ,@(loop for e in l
		    collect
		    (format nil "struct ~a* ~a =nullptr;" e e))
	    (defun registry_handle_global (data registry id interface version)
	      (declare (type void* data)
		       (type "struct wl_registry*" registry)
		       (type uint32_t id)
		       (type "const char*" interface)
		       (type uint32_t version))
	      ,(lprint :vars `(id version interface))
	      ,@(loop for e in l
		      collect
		      `(when (== (string ,e)
				 (std--string_view interface))
			 ,(lprint :msg (format nil "~a" e) :vars `(id version interface))
			 (setf ,e (,(format nil "static_cast<struct ~a*>" e)
				    (wl_registry_bind registry
						      id
						      ,(format nil "&~a_interface" e)
						      version)))
			 (return))))
	    (defun registry_handle_global_remove (data registry id)
	      (declare (type void* data)
		       (type "struct wl_registry*" registry)
		       (type uint32_t id))

	      ,@(loop for e in l
		      collect
		      `(when (and ,e
				  (== id
				      (wl_proxy_get_id ("reinterpret_cast<struct wl_proxy*>" ,e))))
			 ,(lprint :msg (format nil "~a" e) :vars `(id))
			 (setf ,e nullptr))))
	    (do0
	     (setf "static const struct wl_registry_listener registry_listener"
		   (curly registry_handle_global
			  registry_handle_global_remove))

	     (defun main (argc argv)
	       (declare (type int argc)
			(type char** argv)
			(values int))
	       "(void)argv;"
	       ,(lprint :msg "start" :vars `(argc))
	       (progn
		 (let ((*display (wl_display_connect (string "wayland-0"))))
		   (when (== nullptr
			     display)
		     ,(lprint :msg "can't connect to display")
		     (return -1))


		   (let ((*registry (wl_display_get_registry display)))
		     ,(lprint :msg "add listener..")
		     (wl_registry_add_listener registry
					       &registry_listener
					       nullptr))
		   ,(lprint :msg "roundtrip..")
		   (wl_display_roundtrip display)
		   ,(lprint :msg "dispatch..")
		   (wl_display_dispatch display)

		   ,@(loop for e in l
			   collect
			   `(unless ,e
			      ,(lprint :msg (format nil "missing ~a" e))
			      (return -1)))

		   (progn
		     (let ((fd (shm_open (string "/my-wayland-pool")
					 (logior O_RDWR O_CREAT)
					 (logior S_IRUSR S_IWUSR)))
			   (width 1920)
			   (height 1080)
			   (stride 4)
			   (format WL_SHM_FORMAT_ARGB8888)
			   (size (* width height stride)))
		       (when (< fd 0)
			 ,(lprint :msg "shm_open failed."
				  :vars `(errno (strerror errno)))
			 (return -1))
		       (when (< (ftruncate fd size) 0)
			 ,(lprint :msg "ftruncate failed")
			 (return -1))
		       #+nil    (let ((*data (mmap nullptr size (logior PROT_READ
									PROT_WRITE)
						   MAP_SHARED
						   fd 0))
				      (*buffer (wl_shm_create_buffer wl_shm
								     fd
								     width
								     height
								     stride
								     format))
				      #+nil (*pool (wl_shm_create_pool
						    wl_shm
						    fd
						    size
						    )))
				  (do0
				   ,(lprint :msg "capture screen..")
				   (wl_output_damage_buffer wl_output
							    0 0 width height)
				   (let ((cap_stride (wl_buffer_get_stride buffer))
					 (*cap_data (wl_buffer_get_data buffer))
					 (local_buffer ("std::array<uint8_t,size>"))
					 (out_fn (string "screen.raw"))
					 (file (std--ofstream out_fn
							      std--ios--binary)))
				     (std--memcpy (local_buffer.data)
						  data
						  size)
				     ,(lprint :msg "store to file" :vars `(out_fn))
				     (file.write (reinterpret_cast<char*>
						  (local_buffer.data)
						  size))
				     (file.close))))))

		   (do0
		    ,(lprint :msg "disconnect..")
		    (wl_display_disconnect display))
		   ))
	       ))))

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
					; glbinding--glbinding
		 wayland-client))

	  #+nil
	  (out "target_compile_options( mytest PRIVATE ~{~a~^ ~} )"
	       `())

					;(out "target_precompile_headers( mytest PRIVATE fatheader.hpp )")
	  ))
      )))

