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
    (defparameter *source-dir* #P"example/94_xshm_cap/source00/")
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
       (include <X11/Xlib.h>
		<X11/Xutil.h>
		<X11/extensions/XShm.h>
		<cassert>
		<sys/shm.h>
		<fstream>)

       (defun main (argc argv)
	 (declare (type int argc)
		  (type char** argv)
		  (values int))
	 "(void)argv;"
	 ,(lprint :msg "start" :vars `(argc))
	 (let ((*display (XOpenDisplay nullptr)))
	   (assert display)
	   (let ((screenNum (DefaultScreen display))
		 (rootWindow (RootWindow display
					 screenNum))
		 (info (XShmSegmentInfo))
		 (w (DisplayWidth display screenNum))
		 (h (DisplayHeight display screenNum)))
	     (setf info.shmid -1)
	     (let ((*image (XShmCreateImage display
					    nullptr
					    24
					    ZPixmap
					    nullptr
					    &info
					    w h)))
	       (setf info.shmid (shmget IPC_PRIVATE
					(* image->bytes_per_line
					   image->height)
					(logior IPC_CREAT "0777")))
	       (assert (<= 0 info.shmid) )
	       (setf image->data (reinterpret_cast<char*>
				  (shmat info.shmid 0 0))
		     info.shmaddr image->data)
	       (setf info.readOnly False)
	       (XShmAttach display &info)
	       (XShmGetImage display rootWindow image 0 0 AllPlanes)


	       (let ((file (std--ofstream (string "screenshot.pgm")
					  std--ios--binary)))
		 (<< file (string "P5") std--endl)
		 (<< file image->width
		     (string " ")
		     image->height
		     std--endl)
		 (<< file "255" std--endl)

		 (dotimes (y image->height)
		   (dotimes (x image->width)
		     (let ((pixel (XGetPixel image x y)))
		       (file.put ("static_cast<unsigned char>" pixel)))))
		 (file.close))

	       (shmdt info.shmaddr
		      )
	       (shmctl info.shmid IPC_RMID 0)
	       (XDestroyImage image)
	       (XCloseDisplay display)
	       (return 0)))))

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

