(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

(let ((log-preamble
       `(do0
	 (include ;<iostream>
					;<iomanip>
					;<chrono>
					;<thread>
	  <spdlog/spdlog.h>
	  ))))

  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file
    (defparameter *source-dir* #P"example/86_libtorch/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))

    (ensure-directories-exist *full-source-dir*)
    (load "util.lisp")

    (let ((name `Net))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
					;(include <torch/torch.h>)

			  ,@(loop for e in `(autograd
					     cuda
					;data
					;enum
					;fft
					;jit
					;linalg
					;nested
					     nn
					;optim
					;serialize
					;sparse
					; special
					     types
					;utils
					;version
					     )
				  collect
				  `(include ,(format nil "<torch/~a.h>" e)))
			  )
       :implementation-preamble `(do0
				  ,log-preamble
				  ,@(loop for e in `(autograd
						     cuda
					;	     data
					;	     enum
					;	     fft
					;	     jit
					;	     linalg
					;	     nested
					;	     nn
					;	     optim
					;	     serialize
					;	     sparse
					;	     special
					;	     types
					;utils
					; version
						     )
					  collect
					  `(include ,(format nil "<torch/~a.h>" e)))
				  #+nil
				  (include <torch/torch.h>)
				  )
       :code `(do0
	       (defclass ,name torch--nn--Module
		 "public:"
		 "torch::Tensor W, b;"
		 (defmethod ,name (N M)
		   (declare
		    (type int64_t N M)
					;  (explicit)
		    (construct
		     (W (register_parameter (string "W")
					    (torch--randn (curly N M))))
		     (b (register_parameter (string "b")
					    (torch--randn M))))
		    (values :constructor))
		   ,(lprint :msg "Net constructor")
		   )
		 (defmethod forward (input)
		   (declare (type "torch::Tensor" input)
			    (values "torch::Tensor"))
		   (return (torch--addmm b input W)))
		 #+nil
		 (defmethod ,(format nil "~~~a" name) ()
		   (declare
		    (values :constructor))
		   )))))

    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"main.cpp"
		       *source-dir*))
     `(do0


       (include
					;<tuple>
					;<mutex>
	<thread>
	<iostream>
	<iomanip>
	<chrono>
	<cassert>
					;  <memory>
	)

       (do0
	(include <spdlog/spdlog.h>)
					;(include <torch/torch.h>)
	(include "Net.h")
	)
       #+nil
       (do0
	,@(loop for e in `(autograd
			   cuda
			   data
			   enum
			   fft
			   jit
			   linalg
			   nested
			   nn
			   optim
			   serialize
			   sparse
			   special
			   types
			   utils
			   version
			   )
		collect
		`(include ,(format nil "<torch/~a.h>" e))))


       (defun main (argc argv)
	 (declare (type int argc)
		  (type char** argv)
		  (values int))
	 ,(lprint :msg "start" :vars `(argc))
	 "torch::Tensor tensor = torch::eye(3);"
	 (<< std--cout
	     tensor
	     std--endl)
	 (let ((net (Net 2 3))))
	 )))

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
	    (show-err "-Wall -Wextra";
					;" -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused"
					;"-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "

	      ))
	(macrolet ((out (fmt &rest rest)
		     `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	  (out "cmake_minimum_required( VERSION 3.0 FATAL_ERROR )")
	  (out "project( mytest LANGUAGES CXX )")
	  #+nil(loop for e in `(xtl xsimd xtensor)
		     do
		     (format s "find_package( ~a REQUIRED )~%" e))
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

	  (out "target_include_directories( mytest PRIVATE
/usr/local/include/
 )")

	  #+nil (/home/martin/stage/cl-cpp-generator2/example/86_libtorch/dep/libtorch/include/torch/csrc/api/include/
		 /home/martin/stage/cl-cpp-generator2/example/86_libtorch/dep/libtorch/include/torch/csrc/
		 /home/martin/stage/cl-cpp-generator2/example/86_libtorch/dep/libtorch/include/
		 /home/martin/src/popl/include/
		 )

					;(out "target_compile_features( mytest PUBLIC cxx_std_20 )")
	  (out "set_property( TARGET mytest PROPERTY CXX_STANDARD 20 )")

	  (out "target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )")
	  (loop for e in `(Torch)
		do
		(out "find_package( ~a REQUIRED )" e))

	  #+nil
	  (progn
	    (out "add_library( fmt_static STATIC IMPORTED )")
	    (out "set_target_properties( fmt_static PROPERTIES IMPORTED_LOCATION /usr/local/lib64/libfmt.a )")
	    )

	  #+nil (progn
		  (out "add_library( spdlog_static STATIC IMPORTED )")
		  (out "set_target_properties( spdlog_static PROPERTIES IMPORTED_LOCATION /usr/local/lib64/libspdlog.a )")
		  )

	  (out "find_package( PkgConfig REQUIRED )")
	  (out "pkg_check_modules( spdlog REQUIRED spdlog )")


	  (let ((libs (directory
		       "/home/martin/stage/cl-cpp-generator2/example/86_libtorch/dep/libtorch/lib/lib*.a")))
	    (defparameter *bla* libs)
	    #+nil  (loop for e in libs
			 do
			 (let* ((lib (cl-ppcre:split "(/)" (format nil "~a" e)))
				(liba (elt lib (1- (length lib))))
				#+nil (stem (cl-ppcre:all-matches-as-strings
					     "lib(.*)\\.a" liba)))
			   (register-groups-bind (stem)
						 ("lib(.*)\\.a" liba)
						 (progn
						   (out "add_library( ~a STATIC IMPORTED )" stem)
						   (out "set_target_properties( ~a PROPERTIES IMPORTED_LOCATION ~a )" stem e)
						   ))))

	    (out "target_link_libraries( mytest )")
	    (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
		 `("\"${TORCH_LIBRARIES}\""
					;,@libs

		   spdlog
		   )))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))

