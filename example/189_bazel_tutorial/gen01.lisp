(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(defpackage #:my-cpp-project
  (:use #:cl #:cl-cpp-generator2)) 

(in-package #:my-cpp-project)

;; based on
;; https://blog.devgenius.io/getting-started-with-bazel-for-c-cb3944c673f



(let ()
  (defparameter *source-dir* #P"example/189_bazel_tutorial/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))
  (cl:ensure-directories-exist "bazel_tutorial/cc/my_lib" )
  (write-source
   "bazel_tutorial/cc/main.cpp"
   `(do0
     (include<> iostream)
     (include cc/my_lib/my_lib.hpp)
     (defun main ()
       (let ((obj (MyClass)))
	 (obj.setValue 5)
	 (<< std--cout
	     (string "Value: ")
	     (obj.getValue)
	     std--endl))
       (return 0)))
   :dir *full-source-dir*))
