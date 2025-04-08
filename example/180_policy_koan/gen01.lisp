(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(progn
  (progn
    (defparameter *source-dir* #P"example/180_policy_koan/src/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "../163_value_based_oop/util.lisp")
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      iostream
      )
     "using namespace std;"
     ,@(loop for e in `((:name OpNewCreator :create (return (new T)))
			(:name MallocCreator :create (let ((buf (malloc (sizeof T))))
						       (unless buf
							 (return nullptr))
						       (return (space new (paren buf)
								      "T"))))
			(:name PrototypeCreator
			 :create (return (? prototype (prototype->clone) nullptr))
			 :extra (do0
				 (defmethod PrototypeCreator (&key (obj nullptr))
				   (declare (type T* obj)
					    (values "explicit")
					    (construct (prototype obj))))
				 (defmethod getPrototype () (declare (values T*)) (return prototype))
				 (defmethod setPrototype (obj)
				   (declare (type T* obj))
				   (setf prototype obj))
				 "private:"
				 "T* prototype;")))
	     collect
	     (destructuring-bind (&key name create extra) e
	       `(space template (angle "class T")
		       (defclass+ ,name ()
			 "public:"
			 (defmethod create ()
			   (declare (values "T*")
				    (static))
			   ,create)
			 ,(if extra extra "")))))

     (defclass+ Widget ()
	      "int a;"
       "float f;")
     (space template (angle "class CreationPolicy")
	    (defclass+ WidgetManager "public CreationPolicy"
	      ))
     "using MyWidgetMgr = WidgetManager<OpNewCreator<Widget>>;"
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))

       (let ((wm (MyWidgetMgr))))
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil)
  (sb-ext:run-program "/usr/bin/clang-format"
		      `("-i"
			,@(loop for e in (directory (format nil "~a/*.*" *full-source-dir*))
				collect (format nil "~a" e))
			"-style=file")))

