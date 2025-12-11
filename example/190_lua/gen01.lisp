(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-change-case")
  (ql:quickload "cl-cpp-generator2")
  )

(defpackage #:my-cpp-project
  (:use #:cl #:cl-cpp-generator2)) 

(in-package #:my-cpp-project)




(let ()
  (defparameter *source-dir* #P"example/190_lua/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))
   (write-source
   "src/main.cpp"
   `(do0
     (include<> cstdio vector array)
     (space extern "\"C\"" (progn
			     (include lua.h lualib.h lauxlib.h)))
     (defun main ()
       (declare (values int))
       (return 0)))
   :omit-parens t
   :dir *full-source-dir*)
  #+nil (write-class ))
