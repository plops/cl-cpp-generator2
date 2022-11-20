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

    (let ((name `DCGANGeneratorImpl)
	  (l `((:name conv1 :init (torch--nn--ConvTranspose2dOptions kNoiseSize 256 4)
		      :type "torch::nn::ConvTranspose2d"
		      :options ((bias false)))
	       (:name conv2 :init (torch--nn--ConvTranspose2dOptions 256 128 3)
		      :type "torch::nn::ConvTranspose2d"
		      :options ((stride 2)
				(padding 1) (bias false)))
	       (:name conv3 :init (torch--nn--ConvTranspose2dOptions 128 64 4)
		      :type "torch::nn::ConvTranspose2d"
		      :options ((stride 2)
				(padding 1) (bias false)))
	       (:name conv4 :init (torch--nn--ConvTranspose2dOptions 64 1 4)
		      :type "torch::nn::ConvTranspose2d"
		      :options ((stride 2)
				(padding 1) (bias false)))
	       (:name batch_norm1 :type "torch::nn::BatchNorm2d" :init 256)
	       (:name batch_norm2 :type "torch::nn::BatchNorm2d" :init 128)
	       (:name batch_norm3 :type "torch::nn::BatchNorm2d" :init 64))))
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
	       (defclass ,name "public torch::nn::Module"
		 "public:"
		 ,@(loop for e in l
			 collect
			 (destructuring-bind (&key name init type options) e
			   (format nil "~a ~a;"
				   type
				   name)))
		 (defmethod ,name (kNoiseSize)
		   (declare
		    (type int kNoiseSize)
					;  (explicit)
		    (construct
		     ,@(loop for e in l
			     collect
			     (destructuring-bind (&key name init type options) e
			       `(,name (dot ,init ,@options)))))
		    (values :constructor))
		   ,(lprint :msg "Net constructor")
		   ,@(loop for e in l
			   collect
			   (destructuring-bind (&key name init type options) e
			     `(register_module (string ,name) ,name))))
		 (defmethod forward (x)
		   (declare (type "torch::Tensor" x)
			    (values "torch::Tensor"))
		   ,@(loop for e in `((torch--relu batch_norm1 conv1)
				      (torch--relu batch_norm2 conv2)
				      (torch--relu batch_norm2 conv3)
				      (torch--tanh conv4))
			   collect
			   `(setf x ,(let ((q `x))
				       (loop for f in (reverse e) do
					     (setf q (append `(,f ,q))))
				       q)))
		   (return x))
		 #+nil
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
	(include "DCGANGeneratorImpl.h")
	)
       (do0
	,@(loop for e in `(;autograd
					;cuda
			   data
					;enum
					;fft
					;jit
					;linalg
					;nested
					;nn
			   optim
					;serialize
					;sparse
					;special
					;types
					;utils
					;version
			   )
		collect
		`(include ,(format nil "<torch/~a.h>" e))))

       (TORCH_MODULE DCGANGenerator)

       (defun main (argc argv)
	 (declare (type int argc)
		  (type char** argv)
		  (values int))
	 ,(lprint :msg "start" :vars `(argc))
	 ,(let ((l `((:name kNoiseSize :default 12 :short n)
		     (:name kBatchSize :default 32 :short b)
		     (:name kNumberOfEpochs :default 10 :short e))))
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
		 (exit 0))))

	 "torch::Tensor tensor = torch::eye(3);"
	 (<< std--cout
	     tensor
	     std--endl)
	 ,(let ((ld `((:name layer1 :type "torch::nn::Conv2d"
			     :init (torch--nn--Conv2dOptions 1 64 4)
			     :options ((stride 2)
				       (padding 1)
				       (bias false)))
		      (:name layer1nl
			     :type "torch::nn::LeakyReLU"
			     :init (torch--nn--LeakyReLUOptions)
			     :options ((negative_slope .2)))

		      (:name layer2 :type "torch::nn::Conv2d"
			     :init (torch--nn--Conv2dOptions 64 128 4)
			     :options ((stride 2)
				       (padding 1)
				       (bias false)))
		      (:name layer2bn
			     :type "torch::nn::BatchNorm2d"
			     :init 128)
		      (:name layer2nl
			     :type "torch::nn::LeakyReLU"
			     :init (torch--nn--LeakyReLUOptions)
			     :options ((negative_slope .2)))

		      (:name layer3 :type "torch::nn::Conv2d"
			     :init (torch--nn--Conv2dOptions 128 256 4)
			     :options ((stride 2)
				       (padding 1)
				       (bias false)))
		      (:name layer3bn
			     :type "torch::nn::BatchNorm2d"
			     :init 256)
		      (:name layer3nl
			     :type "torch::nn::LeakyReLU"
			     :init (torch--nn--LeakyReLUOptions)
			     :options ((negative_slope .2)))


		      (:name layer4 :type "torch::nn::Conv2d"
			     :init (torch--nn--Conv2dOptions 256 1 3)
			     :options ((stride 1)
				       (padding 0)
				       (bias false)))

		      (:name layer4nl
			     :type "torch::nn::Sigmoid")

		      )))
	    `(let (;(kNoiseSize 12)
					;(kBatchSize 32)
		   (generator (DCGANGenerator kNoiseSize))
		   (discriminator
		    (torch--nn--Sequential
		     ,@(loop for e in ld
			     collect
			     (destructuring-bind (&key name init type options) e
			       `(,type (dot ,init ,@options))))))
		   (dataset (dot
			     (torch--data--datasets--MNIST (string "./mnist"))
			     (map (torch--data--transforms--Normalize<> .5 .5))
			     (map (torch--data--transforms--Stack<>))))
		   (batches_per_epoch (/ (std--ceil (dot dataset
							 (size)
							 (value)))
					 (static_cast<double> kBatchSize)))
		   (data_loader (torch--data--make_data_loader
				 (std--move dataset)
				 (dot (torch--data--DataLoaderOptions)
				      (batch_size kBatchSize)
				      (workers 2))))
		   )
	       (for-range (&batch *data_loader)
			  ,(lprint :vars `((batch.data.size 0)))
			  (dotimes (i (batch.data.size 0))
			    ,(lprint :vars `((dot batch
						  (aref target i)
						  (item<int64_t>))))))
	       (let ((generator_optimizer
		      (torch--optim--Adam
		       (generator->parameters)
		       (dot
			(torch--optim--AdamOptions 2e-4)
			(betas (curly .9 .5)))))
		     (discriminator_optimizer
		      (torch--optim--Adam
		       (generator->parameters)
		       (dot
			(torch--optim--AdamOptions 5e-4)
			(betas (curly .9 .5))))))
		 (dotimes (epoch kNumberOfEpochs)
		   (let ((batch_index (int64_t 0)))
		     (for-range
		      (&batch *data_loader)
		      (let ((noise (torch--randn (curly (batch.data.size 0)
							kNoiseSize 1 1))))
			,@(loop for e in `((:selec real
						   :img batch.data
						   :lab (dot (torch--empty (batch.data.size 0))
							     (uniform_ .8 1.0)) ; high output
						   :out (discriminator->forward real_images))
					   (:selec fake
						   :img (generator->forward noise)
						   :lab (torch--zeros (batch.data.size 0)) ;low output
						   :out (discriminator->forward (dot fake_images (detach)))))
				collect
				(destructuring-bind (&key selec img lab out) e
				  (flet ((n (var)
					   (format nil "~a_~a" selec var)))
				    (let ((images (n "images"))
					  (labels (n "labels"))
					  (output (n "output"))
					  (d_loss (n "d_loss")))
				      `(do0
					(comments ,(format nil "train discriminator with ~a images" selec))
					(let ((,images ,img)
					      (,labels ,lab)
					      (,output ,out)
					      (,d_loss (torch--binary_cross_entropy
							,output ,labels)))
					  (dot ,d_loss (backward))))))))
			(let ((d_loss (+ fake_d_loss
					 real_d_loss)))
			  (discriminator_optimizer.step)
			  (progn
			    (comments "train generator")
			    (generator->zero_grad)
			    (fake_labels.fill_ 1)
			    (setf fake_output (discriminator->forward fake_images))
			    (let ((g_loss (torch--binary_cross_entropy
					   fake_output fake_labels)))
			      (g_loss.backward)
			      (generator_optimizer.step)
			      ,(lprint :vars `(epoch kNumberOfEpochs batch_index
						     batches_per_epoch
						     (d_loss.item<float>)
						     (g_loss.item<float>)))
			      (incf batch_index)))))
		      ))))))
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
/home/martin/src/popl/include/
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

