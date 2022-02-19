(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(let ((log-preamble `(do0 (include <iostream>
				   <iomanip>
				   <chrono>
				   <thread>
				   <mutex>)
			  "extern std::mutex g_stdout_mutex;"
			  "extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;")))
  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file
    (defparameter *source* "01source")
    (defparameter *source-dir* (format nil "example/72_emsdk/~a/" *source*))
    (load "util.lisp")
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `index
     :headers `()
     :header-preamble `(do0
			(include<>
			 SDL/SDL.h
			 emscripten.h
					;iostream
			 ))
     :implementation-preamble `(do0
				,log-preamble)
     :code `(do0
	     (do0 "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"
		  "std::mutex g_stdout_mutex;")
	     (defun main (argc argv)
	       (declare (type int argc)
			(type char** argv)
			(values "extern \"C\" int"))
	       (setf g_start_time ("std::chrono::high_resolution_clock::now"))
	       (progn
		 ,(lprint :msg "enter program" :vars `(argc (aref argv)))
		 (SDL_Init SDL_INIT_VIDEO)
		 (let ((screen (SDL_SetVideoMode 256 256 32 SDL_SWSURFACE)))
		   #+nil (EM_ASM
			  (string "SDL.defaults.copyOnLock = false; SDL.defaults.discardOnLock = true; SDL.defaults.opaqueFrontBuffer = false;"))
		   (when (SDL_MUSTLOCK screen)
		     ,(lprint :msg "lock screen")
		     (SDL_LockSurface screen))
		   ,(lprint :msg "draw")
		   (dotimes (i 256)
		     (dotimes (j 256)
		       (let ((alpha 255 #+nil (% (+ i j)
						 255)))
			 (setf (aref (static_cast<Uint32*>
				      screen->pixels)
				     (+ i (* 256 j)))
			       (SDL_MapRGBA screen->format i j (- 255 i) alpha)))))
		   (when (SDL_MUSTLOCK screen)
		     ,(lprint :msg "unlock screen")
		     (SDL_UnlockSurface screen))
		   ,(lprint :msg "flip screen")
		   (SDL_Flip screen)
		   ,(lprint :msg "quit sdl")
		   (SDL_Quit))
		 ,(lprint :msg "exit program")
		 (return 0)))
	     ))

    (with-open-file (s (format nil "~a/CMakeLists.txt" *source*)
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (let ((dbg "-ggdb -O0 ")
	    (asan "" ; "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
	      )
	    (show-err " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow  -Wswitch-default -Wundef -Werror  -Wno-unused -Wno-unused-parameter"
	      ;;
	      ;; -Wold-style-cast -Wsign-conversion
	      ;; "-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "
	      ))
	(macrolet ((out (fmt &rest rest)
		     `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	  (out "cmake_minimum_required( VERSION 3.5.1 )")

	  (out "project( example LANGUAGES CXX )")

	  (out "set( CMAKE_CXX_STANDARD 17 )")
	  (out "set( CMAKE_CXX_STANDARD_REQUIRED True )")

	  (progn
	    (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	    (out "set( USE_FLAGS \"-s USE_SDL=2\" )")
	    (out "set( CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${USE_FLAGS}\" )")
	    (out "set( CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG}  ~a ~a ~a \")" dbg asan show-err)

	    )

	  (out "set( CMAKE_EXECUTABLE_SUFFIX \".html\" )")

	  (out "set( SRCS ~{~a~^~%~} )"
	       (append
		(directory (format nil "~a/*.cpp" *source*))))

	  (out "add_executable( index ${SRCS} )"))))))

