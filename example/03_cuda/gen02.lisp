(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; https://www.youtube.com/watch?v=uUEHuF5i_qI
;; https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/vectorAdd/vector_add/vector_add/vector_add.cu

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/vector_add.cu"))
  (let* ((code
	  `(do0
	    (include <cuda_runtime.h>
		     <device_launch_parameters.h>)
	    (defun vector_add ()
	      (declare (values "__global__ void")))
	    (defun main ()
	      (declare (values int)))
	    
	    )))
    (write-source *code-file* code)))
