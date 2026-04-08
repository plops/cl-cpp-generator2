(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(progn
  (defparameter *source-dir* #P"example/195_gccjit/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname 'cl-cpp-generator2 *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  
  
  (write-source
   (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames "main.cpp" *source-dir*))
   `(do0
     (include<> glad/glad.h)
  

     (defun main (argc argv)
       (declare (type int argc) (type char** argv) (values int))
       (return 0))))))))))
  :omit-parens t
  :format t
  :tidy nil))
