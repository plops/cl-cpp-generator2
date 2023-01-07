(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  ;; for classes with templates use write-source and defclass+
  ;; for cpp files without header use write-source
  ;; for class definitions and implementation in separate h and cpp file
  (defparameter *source-dir* #P"example/100_cuda1_1_box/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  #+nil
  (let ((name `AGameCharacter))
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
		  (construct
		   (Camera 3))
		  (values :constructor)))))))

  


  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (include <spdlog/spdlog.h>)
     (include
      ,@(loop for e in `(iostream
			 fstream
			 string
			 cassert
			 vector
			 cuda_runtime.h)
	      collect
	      (format nil "<~a>" e)))
     ,(let ((image-width 512)
	    (image-height 512)
	    (kernel-width 3)
	    (kernel-height 3))
	`(do0
	  (defstruct0 Pixel
	    (red "unsigned char")
	    (green "unsigned char")
	    (blue "unsigned char"))
	  (defun LoadImage (fileName image)
	    (declare (type "const std::string&" fileName)
		     (type "std::vector<Pixel>&" image))
	    (let ((in (std--ifstream fileName
				     std--ios--binary))
		  (header (std--string))
		  (width (int 0))
		  (height (int 0))
		  (maxValue (int 0)))
	      (assert (in.is_open))
	      (>> in header)
	      (assert (== (string "P6") header))
	      (>> in width height maxValue)
	      (assert (== ,image-width width))
	      (assert (== ,image-height height))
	      (assert (== 255 maxValue))
	      (in.ignore 256 (char "\\n"))
	      (image.reserve (* width height))
	      (in.read
	       (reinterpret_cast<char*>
		(image.data)
		(* width height 3)))
	      ))
	  (defun SaveImage (fileName image)
	    (declare (type "const std::string&" fileName)
		     (type "const std::vector<Pixel>&" image))
	    (let ((out (std--ofstream fileName
				      std--ios--binary)))
	      (assert (out.is_open))
	      (<< out
		  
		  (string ,(format nil "P6\\n~a ~a\\n255\\n" image-width image-height))
		  )
	      (out.write
	       ("reinterpret_cast<const char*>"
		(image.data)
		(* ,image-width
		   ,image-height
		   3)))))
	  (defun BoxFilterKernel (input output width height
					;kernelWidth kernelHeight
				  )
	    (declare (type "const Pixel*" input)
		     (type "Pixel restrict*" output)
		     (type int width height kernelWidth kernelHeight)
		     (values "__global__ void"))
	    (let ((x (+ threadIdx.x (* blockIdx.x blockDim.x)))
		  (y (+ threadIdx.y (* blockIdx.y blockDim.y)))
		  (r 0)
		  (g 0)
		  (b 0))
	      (declare (type int x y r g b))
	      (when (<= width x)
		return)
	      (when (<= height y)
		return)
	      (for ((= "int i" ,(floor kernel-width -2))
		    (<= i ,(floor kernel-width 2))
		    (incf i))
		   (for ((= "int j" ,(floor kernel-height -2))
		    (<= j ,(floor kernel-height 2))
		    (incf j))
			(let ((posX (+ x i))
			      (posY (+ y j)))
			  (declare (type int posX posY))
			  (when (and (<= 0 posX)
				     (< posX width)
				     (<= 0 posY)
				     (< posY height))
			    ,@(loop for (e f) in `((r red)
						   (g green)
						   (b blue))
				    collect
				    `(incf ,e (dot (aref input (+ posX (* width posY)))
						  ,f)))))))
	      ,@(loop for e in `(r g b)
		      collect
		      `(setf ,e (/ ,e ,(* kernel-width kernel-height))))
	      (setf (aref output (+ x (* width y)))
		    (curly ,@(loop for e in `(r g b)
				   collect
				   `(cast "unsigned char" ,e))))
	      ))
	  (defun main (argc argv)
	    (declare (type int argc)
		     (type char** argv)
		     (values int))
	    "(void)argv;"
	    ,(lprint :msg "start" :vars `(argc))
	    (let ((input (std--vector<Pixel>)))
	      (LoadImage (string "input.ppm")
			 input))
	    (do0
	     ,@(loop for e in `(d_in d_out)
		     collect
		     `(do0
		       ,(format nil "Pixel *~a;" e)
		       (cudaMalloc (ref ,e)
				   (* (sizeof Pixel)
				      ,image-width
				      ,image-height))))
	     (cudaMemcpy d_in (input.data)
			 (* (sizeof Pixel)
			    ,image-width
			    ,image-height)
			 cudaMemcpyHostToDevice)
	     (let ((blockSize (dim3 16 16)))))

	    
	    "
// Launch the kernel
dim3 blockSize(16, 16);
dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
BoxFilterKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, IMAGE_WIDTH, IMAGE_HEIGHT, KERNEL_WIDTH, KERNEL_HEIGHT);

// Copy the output image back to the host
std::vector<Pixel> outputImage(IMAGE_WIDTH * IMAGE_HEIGHT);
cudaMemcpy(outputImage.data(), d_outputImage, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Pixel), cudaMemcpyDeviceToHost);

// Save the output image
SaveImage("output.ppm", outputImage);

// Clean up
cudaFree(d_inputImage);
cudaFree(d_outputImage);

return 0;
"

	    )))))

  (with-open-file (s (format nil "~a/CMakeLists.txt" *full-source-dir*)
		     :direction :output
		     :if-exists :supersede
		     :if-does-not-exist :create)
		  (let ((l-dep `(spdlog )))
		    (macrolet ((out (fmt &rest rest)
				    `(format s ,(format nil "~&~a~%" fmt) ,@rest))
			       )
			      (out "cmake_minimum_required( VERSION 3.16 FATAL_ERROR )")
			      (out "project( mytest )")

			      ;;(out "set( CMAKE_CXX_COMPILER clang++ )")
			      ;;(out "set( CMAKE_CXX_COMPILER g++ )")
			      (out "set( CMAKE_VERBOSE_MAKEFILE ON )")

			      (out "set( SRCS ~{~a~^~%~} )"
				   (append
				    (directory (format nil "~a/*.cpp" *full-source-dir*))
				    ))

			      (out "add_executable( mytest ${SRCS} )")
			      (out "set_property( TARGET mytest PROPERTY CXX_STANDARD 20 )")

		      ;;(out "target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )")

			      (out "find_package( PkgConfig REQUIRED )")
			      (loop for e in l-dep
				    do
				    (out "pkg_check_modules( ~a REQUIRED ~a )" e e))

			      (out "target_include_directories( mytest PUBLIC ~{${~a_INCLUDE_DIRS}~^ ~} )" l-dep)
			      (out "target_compile_options( mytest PUBLIC ~{${~a_CFLAGS_OTHER}~^ ~} )" l-dep)

		      ;; (out "set_property( TARGET mytest PROPERTY POSITION_INDEPENDENT_CODE ON )")
		      ;(out "set( CMAKE_POSITION_INDEPENDENT_CODE ON )")
		      #+nil
			      (progn
				(out "add_library( libnc SHARED IMPORTED )")
				(out "set_target_properties( libnc PROPERTIES IMPORTED_LOCATION /home/martin/stage/cl-cpp-generator2/example/88_libnc/dep/libnc-2021-04-24/libnc.so )"))
			      (out "target_link_libraries( mytest PRIVATE ~{${~a_LIBRARIES}~^ ~} )"
				   l-dep)
			      (out "target_precompile_headers( mytest PRIVATE fatheader.hpp )")))))

