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
								      "T")))
			       )
			(:name PrototypeCreator
			 :create (return (? prototype (prototype->clone) nullptr))
			 :extra (do0
				 (defmethod PrototypeCreator (obj=nullptr)
				   (declare (type T* obj=nullptr)
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
			   ,(lprint :msg (format nil  "~a<T>::create " name)
				    :vars `((sizeof T)
					    (sizeof *this)
					    (sizeof (decltype *this))
					    (sizeof ,(format nil "~a<T>" name))))
			   ,create)
			 ,(if extra extra "")
			 (do0
			  "protected:"
			  ,(format nil "~~~a(){}" name))))))

     (defclass+ Widget ()
	      "int a;"
       "float f;"
       "array<char,20> c;"
       "public:"
       (defmethod clone ()
	 (declare (values Widget*)
		  )
	 (return (new Widget))
	 ))
     (space ;template (angle "class CreationPolicy")
     "template<template<class Created> class CreationPolicy = OpNewCreator>"
      (defclass+ WidgetManager "public CreationPolicy<Widget>"
	"public:"

	(defmethod WidgetManager ()
	  (declare (values :constructor))
	  )
	
	(defmethod switchPrototype (newPrototype)
	  (declare (type Widget* newPrototype))
	  "CreationPolicy<Widget>& myPolicy = *this;"
	  (delete (myPolicy.getPrototype))
	  (myPolicy.setPrototype newPrototype)
	  )
	))
     "using MyWidgetMgr = WidgetManager<OpNewCreator>;"
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))

       (let ((wm0 (MyWidgetMgr))
	     (e0 (wm0.create)))
	 )
       (let ((wm1 (WidgetManager<MallocCreator>))
	     (e1 (wm1.create))))

       (let ((wm2 (WidgetManager<PrototypeCreator>))
	      )
	 (wm2.setPrototype e1)
	 (let ((e2 (wm2.create)))))
       (wm2.switchPrototype e2)
       ,(lprint :vars `((sizeof wm0)
			(sizeof wm1)
			(sizeof wm2)))
       
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil)
  (sb-ext:run-program "/usr/bin/clang-format"
		      `("-i"
			,@(loop for e in (directory (format nil "~a/*.*" *full-source-dir*))
				collect (format nil "~a" e))
			"-style=file")))
