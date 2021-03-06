(declaim (optimize 
	  (safety 3)
	  (speed 0)
	  (debug 3)))

(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)



(setf *features* (union *features* `()))

(setf *features* (set-difference *features*
				 '()))
(defvar *header-file-hashes* (make-hash-table))

(progn
  (defparameter *source-dir* #P"example/57_torch/source/")
  
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  
  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " ")
    (defparameter *global-code* nil)
    (defun emit-global (&key code)
      (push code *global-code*)
      " "))
  (progn
    
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)
    (defun logprint (msg &optional rest)
      `(progn ;do0
	" "
	#-nolog
	(let (;(lock (std--unique_lock<std--mutex> ,(g `_stdout_mutex)))
	      )
	 
	 (do0
					;("std::setprecision" 3)
	  (<< "std::cout"
	      ;;"std::endl"
	      ("std::setw" 10)
	      (dot ("std::chrono::high_resolution_clock::now")
		   (time_since_epoch)
		   (count))
					;,(g `_start_time)
	     
	      (string " ")
	      ("std::this_thread::get_id")
	      (string " ")
	      __FILE__
	      (string ":")
	      __LINE__
	      (string " ")
	      __func__
	      (string " ")
	      (string ,msg)
	      (string " ")
	      ,@(loop for e in rest appending
		     `(("std::setw" 8)
					;("std::width" 8)
		       (string ,(format nil " ~a='" (emit-c :code e)))
		       ,e
		       (string "'")))
	      "std::endl"
	      "std::flush")))))
    (defun emit-globals (&key init)
      (let ((l `((_start_time ,(emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
							   (time_since_epoch)
							   (count)))))
		 ,@(loop for e in *module-global-parameters* collect
			(destructuring-bind (&key name type default)
			    e
			  (declare (ignorable default))
			  `(,name ,type))))))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(declare (ignorable type))
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      (include <chrono>)
	      (defstruct0 State
		  ,@(loop for e in l collect
 			 (destructuring-bind (name type &optional value) e
			   (declare (ignorable value))
			   `(,name ,type))))))))
    (defun define-module (args)
      "each module will be written into a c file with module-name. the global-parameters the module will write to will be specified with their type in global-parameters. a file global.h will be written that contains the parameters that were defined in all modules. global parameters that are accessed read-only or have already been specified in another module need not occur in this list (but can). the prototypes of functions that are specified in a module are collected in functions.h. i think i can (ab)use gcc's warnings -Wmissing-declarations to generate this header. i split the code this way to reduce the amount of code that needs to be recompiled during iterative/interactive development. if the module-name contains vulkan, include vulkan headers. if it contains glfw, include glfw headers."
      (destructuring-bind (module-name global-parameters module-code) args
	(let ((header ()))
	  #+nil (format t "generate ~a~%" module-name)
	  (push `(do0
		  " "
		  (include "utils.h")
		  " "
		  (include "globals.h")
		  " "
		  ;(include "proto2.h")
		  " ")
		header)
	  (unless (or (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
		      (cl-ppcre:scan "base" (string-downcase (format nil "~a" module-name)))
		      (cl-ppcre:scan "mesher_module" (string-downcase (format nil "~a" module-name))))
	    (push `(do0 "extern State state;")
		  header))
	  (push `(:name ,module-name :code (do0 ,@(reverse header) ,module-code))
		*module*))
	(loop for par in global-parameters do
	     (destructuring-bind (parameter-name
				  &key (direction 'in)
				  (type 'int)
				  (default nil)) par
	       (declare (ignorable direction))
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))

  (let*  ()
    (define-module
       `(base ((_main_version :type "std::string")
	       (_code_repository :type "std::string")
	       (_code_generation_time :type "std::string")
	       (_stdout_mutex :type "std::mutex")
	       )
	      (do0
	       (include <iostream>
			<chrono>
			<thread>
			
					;<future>
					; <experimental/future>
		
			)
	       " "

	       
	       " "
	       ;(include <torch/torch.h>)
	       " "
	       (split-header-and-code
		(do0 (comments "header"))
		(do0 (comments "implementation")
		     (include "vis_00_base.hpp")))
	       ;"using namespace torch;"

	    	       "using namespace std::chrono_literals; "
	       " "
	       
	      
	       
	       (let ((state ,(emit-globals :init t)))
		 (declare (type "State" state)))

	       ,(let ((c `((c256 256)
			   (c128 128)
			   (c64 64)
			  ; (imageSize 64)
			 ;  (ngf 64)
			   (kNoiseSize 100)
			   (kBatchSize 256)
			   (kNumberOfEpochs 28)
			   (kCheckpointEvery 200)))
		      (l `((conv1 ConvTranspose2d
				  k_noise_size c256 4 ;; input channels, output channels, kernel size
				  :bias false)
			   (batch_norm1 BatchNorm2d c256)
			   (conv2 ConvTranspose2d
				  c256 c128 3
				  :stride 2
				  :padding 1
				  :bias false)
			   (batch_norm2 BatchNorm2d c128)
			   (conv3 ConvTranspose2d
				  c128 c64 4
				  :stride 2
				  :padding 1
				  :bias false)
			   (batch_norm3 BatchNorm2d c64)
			   (conv4 ConvTranspose2d
				  c64 1 4
				  :stride 2
				  :padding 1
				  :bias false)))
		      (l-discriminator 
			`(;; layer 1
			  (Conv2d :opt (1 c64 4) ;; input channels, output channels, kernel size
				  :stride 2
				  :padding 1
				  :bias false)
			  (LeakyReLU :opt () :negative-slope .2)
			  ;; layer 2
			  (Conv2d :opt (c64 c128 4)
				  :stride 2
				  :padding 1
				  :bias false)
			  (BatchNorm2d :param c128)
			  (LeakyReLU :opt () :negative-slope .2)
			  ;; layer 3
			  (Conv2d :opt (c128 c256 4)
				  :stride 2
				  :padding 1
				  :bias false)
			  (BatchNorm2d :param c256)
			  (LeakyReLU :opt () :negative-slope .2)
			  ;; layer 4
			  (Conv2d :opt (c256 1 3)
				  :stride 1
				  :padding 0
				  :bias false)
			  (Sigmoid :param nil))))
		  
		  `(do0
		    ,@(loop for (name val) in c
			     collect
			     (format nil "static constexpr int ~a=~a;" name val))
		    (defclass DCGANGeneratorImpl "public torch::nn::Module"
		     "public:"
		     
		     (defmethod DCGANGeneratorImpl (k_noise_size)
		       (declare (type int k_noise_size)
				(values :constructor)
				(construct
				 ,@(loop for e in l
					 collect
					 (destructuring-bind (name type
							      x
							      &optional y z
							      &key
								stride
								padding
								bias) e
					   (if (eq type 'ConvTranspose2d)
					       `(,name (dot
							(,(format nil "torch::nn::~aOptions" type)
							 ,x ,y ,z)
							,(when stride `(stride ,stride))
							,(when padding `(padding ,padding))
							,(when bias `(bias ,bias))
							)
						       )
					       `(,name ,x))
					   ))))
		       (comments "k_noise_size is the size of the input noise vector")
		       ,@(loop for e in l
			       collect
			       (destructuring-bind (name type x &optional y z
						    &key stride padding bias)
				   e
				 `(register_module (string ,name) ,name))))

		     (defmethod forward (x)
		       (declare (type "torch::Tensor" x)
				(values "torch::Tensor"))
		       (setf x (torch--relu (batch_norm1 (conv1 x))))
		       (setf x (torch--relu (batch_norm2 (conv2 x))))
		       (setf x (torch--relu (batch_norm3 (conv3 x))))
		       (setf x (torch--tanh (conv4 x)))
		       (return x))

		     ,@(loop for e in l
			       collect
			       (destructuring-bind (name type x &optional y z
						    &key stride padding bias)
				   e
				 (format nil "torch::nn::~a ~a;" type name)))
		     
		     )
		    (TORCH_MODULE DCGANGenerator)

		    (defun main (argc argv)
		 (declare (type int argc)
			  (type char** argv)
			  (values int))
		 (do0
		  (setf ,(g `_main_version)
			(string ,(let ((str (with-output-to-string (s)
					      (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
				   (subseq str 0 (1- (length str))))))

		  (setf
		   ,(g `_code_repository) (string ,(format nil "https://github.com/plops/cl-cpp-generator2/tree/master/~a"
							   *source-dir*))
		   ,(g `_code_generation_time) 
		   (string ,(multiple-value-bind
				  (second minute hour date month year day-of-week dst-p tz)
				(get-decoded-time)
			      (declare (ignorable dst-p))
			      (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
				      hour
				      minute
				      second
				      (nth day-of-week *day-names*)
				      year
				      month
				      date
				      (- tz)))))

		  (setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
					       (time_since_epoch)
					       (count)))
		 
		  ,(logprint "start main" `(,(g `_main_version)
					    ,(g `_code_repository)
					    ,(g `_code_generation_time)))
		 
	
		  )

	
		 (do0
		  (torch--manual_seed 1)
		  (let ((device (torch--Device torch--kCPU)))
		    (when (torch--cuda--is_available)
		      (setf device (torch--Device torch--kCUDA))
		      ,(logprint "we have cuda" `(device))
		      )
		    
		    (let (
			  (generator (DCGANGenerator kNoiseSize)))
		      (generator->to device)
		      (let ((discriminator
			      (torch--nn--Sequential
			       ,@(loop for e in l-discriminator
				       collect
				       (destructuring-bind (type  &key (param nil param-p) opt stride padding bias negative-slope) e
					 `(,(format nil "torch::nn::~a" type)
					   ,(if param-p
					       param
					       `(dot (,(format nil "torch::nn::~aOptions" type)
						      ,@opt)
						     ,(when stride `(stride ,stride))
						     ,(when padding `(padding ,padding))
						     ,(when bias `(bias ,bias))
						     ,(when negative-slope `(negative_slope ,negative-slope))))))))))
			(discriminator->to device)
			(let ((dataset (dot (torch--data--datasets--MNIST (string "./mnist"))
					    ;(map (torch--data--transforms--Resize<> imageSize))
					    (map (torch--data--transforms--Normalize<> .5 .5))
					    (map (torch--data--transforms--Stack<>))
					    )
				       )
			      ;(kBatchSize 64)
			      (data_loader (torch--data--make_data_loader
					    (std--move dataset)
					    (dot (torch--data--DataLoaderOptions)
						 (batch_size kBatchSize)
						 (workers 12)))))
			  #+nil
			  (do0
			   (comments "print data")
			   (for-range (&batch *data_loader)
				      ,(logprint "" `((batch.data.size 0)
						      (dot batch (aref target 0)
							   (item<int64_t>))))))

			  (let ((generator_optimizer
				  (torch--optim--Adam (generator->parameters)
						      (dot (torch--optim--AdamOptions 2e-4)
							   (betas (std--make_tuple .5 .999)))))
				(discriminator_optimizer
				  (torch--optim--Adam (discriminator->parameters)
						      (dot (torch--optim--AdamOptions 2e-4)
							   (betas (std--make_tuple .5 .999)))))
				(checkpoint_counter 0))
			    (when true ;; restore from checkpoint
			      ,@(loop for e in `(generator
						 generator_optimizer
						 discriminator
						 discriminator_optimizer)
				      collect
				      `(torch--load ,e (string
							,(format nil "~a.pt" e))))
				)
			    (dotimes (epoch kNumberOfEpochs)
			      (let ((batch_index 0)
				    )
				(for-range
				 (&batch *data_loader)
				 (do0
				  (comments "train discriminator with real images")
				  (discriminator->zero_grad)
				  (let ((real_images (dot batch.data
							  (to device)))
					(real_labels (dot (torch--empty (batch.data.size 0)
									device)
							  (uniform_ .8 1.0)))
					(real_output (discriminator->forward real_images))
					(real_d_loss (torch--binary_cross_entropy
						      real_output
						      real_labels)))
				    (dot real_d_loss (backward))
				    (do0
				     (comments "train discriminator with fake images"
					       )
				     (let ((noise (torch--randn (curly (batch.data.size 0)
								       kNoiseSize
								       1 1)
								device))
					   (fake_images (generator->forward noise))
					   (fake_labels (torch--zeros (batch.data.size 0)
								      device))
					   (fake_output (discriminator->forward
							 (fake_images.detach)))
					   (fake_d_loss (torch--binary_cross_entropy
							 fake_output
							 fake_labels))
					   )
				       (dot fake_d_loss (backward))
				       (let ((d_loss (+ real_d_loss
							fake_d_loss)))
					 (discriminator_optimizer.step))
				       (do0
					(comments "train generator"
						  "discriminator should assign probabilities close to 1")
					(generator->zero_grad)
					(fake_labels.fill_ 1)
					(setf fake_output (discriminator->forward
							   fake_images))
					(let ((g_loss (torch--binary_cross_entropy
						       fake_output
						       fake_labels)))
					  (dot g_loss (backward))
					  (generator_optimizer.step)
					  ,(logprint ""
						     `(epoch (incf batch_index)
							     (real_d_loss.item<float>)
							     (fake_d_loss.item<float>)
							      (d_loss.item<float>)
							      (g_loss.item<float>)))
					  (when (== 0 (% batch_index kCheckpointEvery))
					    ,@(loop for e in `(generator
							       generator_optimizer
							       discriminator
							       discriminator_optimizer)
						    collect
						    `(torch--save ,e (string
								      ,(format nil "~a.pt" e))))
					    (let ((samples (generator->forward
							    (torch--randn (curly 8
										 kNoiseSize
										 1
										 1)
									  device))))
					      (torch--save (* .5 (+ samples 1.0))
							   (torch--str
							    (string "dcgan-sample-")
							    (incf checkpoint_counter)
							    (string ".pt")))
					      ,(logprint "" `(checkpoint_counter))
					      )))))))))))
			    )))))
		  )

		 (return 0))
		    ))
	       
	    
	       )))
    (define-module
       `(demangle ()
	      (do0
	       (include <iostream>
			<chrono>
			<thread>
			
					;<future>
					; <experimental/future>
			
			)

	       " "

	       (include <cxxabi.h>)
	       " "


	       	       
	       "using namespace std::chrono_literals;"

	      
	       " "
	       
	       (defun demangle (name)
		 (declare (type ;"const char*"
			   "const std::string"
			   name)
			  (values "std::string"))
		 (let ((status -4))
		   "std::unique_ptr<char,void(*)(void*)> res {abi::__cxa_demangle(name.c_str(), nullptr,nullptr,&status),std::free};"
		   (if (== 0 status)
		       (return (res.get))
		       (return name))))
	       (defun type_name ()
		 (declare (values "template<class T> std::string"))
		 "typedef typename std::remove_reference<T>::type TR;"
		 "std::unique_ptr<char,void(*)(void*)> own(nullptr,std::free);"
		 "std::string r = (own != nullptr) ? own.get() : typeid(TR).name();"
		 (setf r (demangle r))
		 ,@(loop for (e f) in `(
					(" const" std--is_const<TR>--value)
					(" volatile" std--is_volatile<TR>--value)
					("&" std--is_lvalue_reference<TR>--value)
					("&&" std--is_rvalue_reference<TR>--value))
			 collect
			 `(when ,f
			    (incf r (string ,e))))
		 (return r))
	       )))
    
    
  )
  
  (progn
    
    
    (progn ;with-open-file
      #+nil (s (asdf:system-relative-pathname 'cl-cpp-generator2
					(merge-pathnames #P"proto2.h"
							 *source-dir*))..
	 :direction :output
	 :if-exists :supersede
	 :if-does-not-exist :create)
      #+nil (format s "#ifndef PROTO2_H~%#define PROTO2_H~%~a~%"
		    (emit-c :code `(include <cuda_runtime.h>
					    <cuda.h>
					    <nvrtc.h>)))

      ;; include file
      ;; http://www.cplusplus.com/forum/articles/10627/
      
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       
	       (unless cuda
		 #+nil (progn (format t "emit function declarations for ~a~%" name)
			      (emit-c :code code
				      :hook-defun #'(lambda (str)
						      (format t "~a~%" str))
				      :header-only t))
		 #+nil (emit-c :code code
			 :hook-defun #'(lambda (str)
					 (format s "~a~%" str)
					 )
			 :hook-defclass #'(lambda (str)
					    (format s "~a;~%" str)
					    )
			 :header-only t
			 )
		 (let* ((file (format nil
				      "vis_~2,'0d_~a"
				      i name
				      ))
			(file-h (string-upcase (format nil "~a_H" file)))
			(fn-h (asdf:system-relative-pathname 'cl-cpp-generator2
								      (format nil "~a/~a.hpp"
									      *source-dir* file)))
			
			(code-str (with-output-to-string (sh)
				    (format sh "#ifndef ~a~%" file-h)
				    (format sh "#define ~a~%" file-h)
			 
				    (emit-c :code code
								    :hook-defun #'(lambda (str)
										    (format sh "~a~%" str))
								    :hook-defclass #'(lambda (str)
										       (format sh "~a;~%" str))
					    :header-only t)
				    (format sh "#endif")))
			(fn-hash (sxhash fn-h))
			(code-hash (sxhash code-str)))
		   (multiple-value-bind (old-code-hash exists) (gethash fn-hash *header-file-hashes*)
		     (when (or (not exists)
			       (/= code-hash old-code-hash)
			       (not (probe-file fn-h)))
		       ;; store the sxhash of the header source in the hash table
		       ;; *header-file-hashes* with the key formed by the sxhash of the full
		       ;; pathname
		       (setf (gethash fn-hash *header-file-hashes*) code-hash)
		       (format t "~&write header: ~a fn-hash=~a ~a old=~a~%" fn-h fn-hash code-hash old-code-hash
			       )
		       (with-open-file (sh fn-h
					   :direction :output
					   :if-exists :supersede
					   :if-does-not-exist :create)
			 (format sh "#ifndef ~a~%" file-h)
			 (format sh "#define ~a~%" file-h)
			 
			 (emit-c :code code
				 :hook-defun #'(lambda (str)
						 (format sh "~a~%" str))
				 :hook-defclass #'(lambda (str)
						    (format sh "~a;~%" str))
				 :header-only t)
			 (format sh "#endif"))
		       (sb-ext:run-program "/usr/bin/clang-format"
					   (list "-i"  (namestring fn-h)))))))
	       (progn
		#+nil (format t "emit cpp file for ~a~%" name)
		(write-source (asdf:system-relative-pathname
			       'cl-cpp-generator2
			       (format nil
				       "~a/vis_~2,'0d_~a.~a"
				       *source-dir* i name
				       (if cuda
					   "cu"
					   "cpp")))
			      code)))))
      #+nil (format s "#endif"))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"utils.h"
				    *source-dir*))
		  `(do0
		    "#ifndef UTILS_H"
		    " "
		    "#define UTILS_H"
		    " "
		    (include <vector>
			     ;<array>
			     <iostream>
			     <iomanip>)
		    
		    " "
		    (do0
		     
		     " "
		     ,@(loop for e in (reverse *utils-code*) collect
			  e))
		    " "
		    "#endif"
		    " "))
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames
								     #P"globals.h"
								     *source-dir*))
		  `(do0
		    "#ifndef GLOBALS_H"
		    " "
		    "#define GLOBALS_H"
		    " "
		    (include <torch/torch.h>)
		    #+nil (include <complex>)
		    #+nil (include <deque>
			     <map>
			     <string>)
		     (include ;<thread>
			      <mutex>
			     ;<queue>
			     ;<condition_variable>
			     )
		    " "

		    " "
		    ;(include "proto2.h")
		    " "
		    ,@(loop for e in (reverse *global-code*) collect
			 e)

		    " "
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))
    (with-open-file (s "source/CMakeLists.txt" :direction :output
					       :if-exists :supersede
					       :if-does-not-exist :create)
      ;; run emcmake cmake .. && make
      (macrolet ((out (fmt &rest rest)
		   `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	
	
	(out "cmake_minimum_required( VERSION 3.0 FATAL_ERROR )")
	(out "project( mytest LANGUAGES CXX )")
	(out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	(out "set( CMAKE_PREFIX_PATH /home/martin/stage/cl-cpp-generator2/example/57_torch/source/libtorch/share/cmake/Torch/ )")
	;(out "set( CMAKE_CXX_STANDARD 14 )")

	(loop for e in `(C CXX) do
     (loop for (f flags) in `((DEBUG (-ggdb3 -gdwarf-4 -g3 -Og -fvar-tracking-assignments -fno-eliminate-unused-debug-symbols ))
			      (MINSIZEREL  (-g0 -Os))
			      (RELWITHDEBINFO  (-g2 -ggdb -O2))
			      (RELEASE (-g0 -O2)))
	   do
	      (out "set( CMAKE_~a_FLAGS_~a \"~{~a~^ ~}\" )"
		   e f (if (eq e 'CXX)
			   (append flags )
			   (append flags )))
	   ))
	
	(out "find_package( Torch REQUIRED )")
	(out "set( Torch_DIR /home/martin/.local/lib/python3.8/site-packages/torch/share/cmake/Torch/ )")
	(out "set( SRCS ~{~a~^~%~} )"	;(directory "source/*.cpp")
	     `(vis_00_base.cpp
	       vis_01_demangle.cpp))
	
	(out "add_executable( mytest ${SRCS} )")
	(out "target_link_libraries( mytest ${TORCH_LIBRARIES} )")
	(out "target_precompile_headers( mytest PUBLIC globals.h )")
	(out "set_property( TARGET mytest PROPERTY CXX_STANDARD 14 )")
	) 
      )))



