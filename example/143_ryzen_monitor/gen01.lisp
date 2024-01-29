(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(let ()
  (defparameter *source-dir* #P"example/143_ryzen_monitor/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include "imgui.h"
	      "imgui_impl_glfw.h"
	      "imgui_impl_opengl3.h")
     (include<> GLFW/glfw3.h
		format
		iostream)
     (defun glfw_error_callback (err description)
       (declare (type int err)
		(type "const char*" description))
       ,(lprint :vars `(err description)))
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       ))
   :omit-parens t
   :format t
   :tidy nil))

