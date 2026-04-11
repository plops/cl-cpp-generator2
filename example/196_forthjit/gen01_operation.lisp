(in-package :cl-cpp-generator2)

(let* ((class-name 'Operation))
  (write-class
   :dir *full-source-dir*
   :name class-name
   :headers '()
   :header-preamble `(do0 (include<> string vector variant))
   :implementation-preamble '()
   :code `(do0
	   (defclass ,class-name ()
	     "public:"
	     (defmethod ,class-name ()
	       (declare (construct))
	       (progn))))))
