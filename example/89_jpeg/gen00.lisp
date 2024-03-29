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
    (defparameter *source-dir* #P"example/89_jpeg/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))

    (ensure-directories-exist *full-source-dir*)
    (load "util.lisp")

    (let ((name `JxlEncode)
	  (types `(;(:var enc :name JxlEncoderPtr :member t :ptr nil)
		   #+nil
		   (:var runner :name JxlThreadParallelRunnerPtr
			 :member t :ptr nil)
					;(:name JxlEncoderFrameSettings :member t)
					;  (:var options :name JxlEncoderOptions :member t)
					;(:name JxlEncoderError :member t)
		   )))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include <jxl/encode_cxx.h>
				   <vector>)
			  #+nil
			  ,@(loop for e in types
				  collect
				  (destructuring-bind (&key var name member) e
				    (format nil "struct ~a;" name)))
			  )
       :implementation-preamble `(do0
				  ,log-preamble
				  " "

				  (include <jxl/encode.h>
					   <jxl/encode_cxx.h>
					   <jxl/thread_parallel_runner.h>
					   <jxl/thread_parallel_runner_cxx.h>

					   <vector>)
				  " "
				  )
       :code `(do0
	       (defclass ,name ()
		 "public:"
					;"std::vector<uint8_t> compressed;"
					;"size_t numWorkerThreads;"

		 "JxlThreadParallelRunnerPtr runner;"
		 ,@(loop for e in types
			 collect
			 (destructuring-bind (&key var name member (ptr t)) e
			   (when member
			     (format nil "~a ~:[~;*~]~a;"
				     name
				     ptr
				     (if var
					 var
					 (string-downcase name))))))

		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    #+nil (construct
			   (enc (JxlEncoderMake nullptr))
			   (numWorkerThreads (JxlThreadParallelRunnerDefaultNumWorkerThreads))
			   (runner
			    (JxlThreadParallelRunnerMake
			     nullptr
			     numWorkerThreads
			     )))
		    (values :constructor))
		   ,(lprint :msg (format nil "~a constructor" name))
		   (let ((enc (JxlEncoderMake nullptr))

			 (runner (JxlThreadParallelRunnerMake nullptr
							      4)))
		     #+nil (setf runner (JxlThreadParallelRunnerMake nullptr
								     4))
		     (unless (== JXL_ENC_SUCCESS
				 (JxlEncoderSetParallelRunner
				  (enc.get)
				  JxlThreadParallelRunner
				  (runner.get)))
		       ,(lprint :msg "parallel runner setting failed"))
		     (progn
		       "JxlBasicInfo basic_info;"
		       (JxlEncoderInitBasicInfo
			&basic_info)
		       ,@(loop for (e f) in `((xsize 512)
					      (ysize 256)
					      (bits_per_sample 32)
					      (exponent_bits_per_sample 8)
					      (uses_original_profile
					       JXL_FALSE))
			       collect
			       `(setf (dot basic_info ,e)
				      ,f))
		       (unless (== JXL_ENC_SUCCESS
				   (JxlEncoderSetBasicInfo
				    (enc.get)
				    &basic_info))
			 ,(lprint :msg "basic info failed")
			 )
		       (let ((*frame_settings (JxlEncoderFrameSettingsCreate
					       (enc.get)
					       nullptr)))
			 (let ((pixel_format
				(JxlPixelFormat (curly 3
						       JXL_TYPE_FLOAT
						       JXL_NATIVE_ENDIAN
						       0)))))
			 (unless (== JXL_ENC_SUCCESS
				     (JxlEncoderAddImageFrame
				      frame_settings
				      &pixel_format
				      nullptr 0))
			   ,(lprint :msg "adding image frame failed")
			   ))
		       (JxlEncoderCloseInput (enc.get))
		       ))
		   )
		 (defmethod Encode (;pixels
					; width height
				    )
		   (declare (type "std::vector<float>" pixels)
			    (type int width height))

		   )
		 #+nil (defmethod Reset ()
			 (JxlEncoderReset (jxlencoderptr.get)))
		 (defmethod ,(format nil "~~~a" name) ()
		   (declare
		    (values :constructor))

		   #+nil (JxlEncoderDestroy jxlencoder)))
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
	(include "JxlEncode.h"))

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

	       (let ((jxl (JxlEncode))))

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
	  (out "pkg_check_modules( jxl REQUIRED libjxl )")
	  (out "pkg_check_modules( jxlt REQUIRED libjxl_threads )")

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
		 jxl
		 jxl_threads
		 ))

	  #+nil
	  (out "target_compile_options( mytest PRIVATE ~{~a~^ ~} )"
	       `())

					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))

