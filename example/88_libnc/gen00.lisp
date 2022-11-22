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
    (defparameter *source-dir* #P"example/88_libnc/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))

    (ensure-directories-exist *full-source-dir*)
    (load "util.lisp")
    #+nil
    (let ((name `DCGANGeneratorImpl)
	  )
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  )
       :implementation-preamble `(do0
				  ,log-preamble
				  
				  )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    (construct)
		    (values :constructor))
		   ,(lprint :msg (format nil "~a constructor" name))
		   )
		 (defmethod ,(format nil "~~~a" name) ()
		   (declare
		    (values :constructor))
		   ))
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
	(include <popl.hpp>)
					;(include <torch/torch.h>)
					;(include "DCGANGeneratorImpl.h")
	)
       (do0
	(include <libnc.h>))

       (defun main (argc argv)
	 (declare (type int argc)
		  (type char** argv)
		  (values int))
	 ,(lprint :msg "start" :vars `(argc))
	 ,(let ((l `(;(:name kNoiseSize :default 100 :short n)
		     )))
	    `(let ((op (popl--OptionParser (string "allowed opitons")))
		   ,@(loop for e in l collect
			   (destructuring-bind (&key name default short) e
			     `(,name (int ,default))))
		   ,@(loop for e in `((:long help :short h :type Switch :msg "produce help message")
				      (:long verbose :short v :type Switch :msg "produce verbose output")
				      ,@(loop for f in l
					      collect
					      (destructuring-bind (&key name default short) f
						`(:long ,name
							:short ,short
							:type int :msg "parameter"
							:default ,default :out ,(format nil "&~a" name))))

				      )
			   appending
			   (destructuring-bind (&key long short type msg default out) e
			     `((,(format nil "~aOption" long)
				 ,(let ((cmd `(,(format nil "add<~a>"
							(if (eq type 'Switch)
							    "popl::Switch"
							    (format nil "popl::Value<~a>" type)))
						(string ,short)
						(string ,long)
						(string ,msg))))
				    (when default
				      (setf cmd (append cmd `(,default)))
				      )
				    (when out
				      (setf cmd (append cmd `(,out)))
				      )
				    `(dot op ,cmd)
				    ))))
			   ))
	       (op.parse argc argv)
	       (when (helpOption->count)
		 (<< std--cout
		     op
		     std--endl)
		 (exit 0))
	       
	       ))
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
	  (out "pkg_check_modules( pcappp REQUIRED PcapPlusPlus )")

	  (out "target_include_directories( mytest PRIVATE
/usr/local/include/
/home/martin/src/popl/include/
/home/martin/stage/cl-cpp-generator2/example/87_pcap/dep/installed/include/pcapplusplus
 )")


	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `("\"${pcappp_LIBRARIES}\""

		 spdlog
		 ))

	  (out "target_compile_options( mytest PRIVATE ~{~a~^ ~} )"
	       `("${pcappp_CFLAGS_OTHER}"))

					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))

