(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
 (defparameter *server-dir*
   (asdf:system-relative-pathname 'cl-cpp-generator2
                                  "example/198_vermellcc/server/src/"))
 (defparameter *main-file* (merge-pathnames "main.cpp" *server-dir*))
 (ensure-directories-exist *server-dir*)

 ;; -------------------------------------------------------------------------
 ;; 4a. common.glsl -- quaternion algebra, marker table, camera tour
 ;; -------------------------------------------------------------------------
 (write-source
  *main-file*
  `(do0
    (include<> vermell/vermell.h)
    
    (defun main ()
      
      (declare (values int))
      (let ((router (Router)))
	(router.setPort 8080)
	(router.get (string "/")
		    (curly (lambda (http)
			     (declare (type Query& http))
			     (http.send (string "Hello from Vermell")))))
	(router.listen))))
  :format t :tidy nil :omit-parens t))
