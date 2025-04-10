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
    (defparameter *source-dir* #P"example/181_reference_koan/src/")
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
      array
      deque
      memory
      atomic
      condition_variable
      mutex
      )
     "using namespace std;"

     (space "template<typename T>"
      (defclass+ Ref ()
	"public:"
	;; ctor
	(defmethod Ref (r)
	  (declare (type T& r)
		   (construct (ref r)
			      (sp (make_shared<Priv>))
			      (q (new Q)))
		   (explicit)
		   (values :constructor))
	  ,(lprint :msg "Ref::ctor"))
	;; dtor
	(defmethod ~Ref ()
	  (declare (values :constructor))
	  (delete q)
	  ,(lprint :msg "~Ref"))

	;; copy ctor
	(defmethod Ref (rhs)
	  (declare (type "const T&" rhs)
		   (construct (ref rhs.ref)
			      (sp (make_shared<Priv> )))
		   (values :constructor))
	  ,(lprint :msg "Ref::copy-ctor"))

	(defmethod use_count ()
	  (declare (values int))
	  (return (dot sp (load ) (use_count))))
	"private:"
	(defclass+ Priv ())
	(defclass+ Q ()
	  "private:"
	  "mutex m;"
	  "condition_variable c;")
	"T& ref;"
	"atomic<shared_ptr<Priv>> sp;"
	"Q* q;"))
          
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       (defclass+ Widget ()
	 "public:"
	 "private:"
	 "int i{3};"
	 "float f{4.5F};")
       "constexpr int N{17};"
       (let ((as (space array (angle Widget N)
		       (paren)))))
       (let ((ar (space deque (angle (space Ref (angle Widget)))
			(paren))))
	 (for-range (e as)
		    (ar.emplace_back e)))
       
       ,(lprint :vars `((sizeof as)))
       ,(lprint :vars `((sizeof ar)))
       ,(lprint :vars `((dot (aref ar 0) (use_count))))
            
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil)
  (sb-ext:run-program "/usr/bin/clang-format"
		      `("-i"
			,@(loop for e in (directory (format nil "~a/*.*" *full-source-dir*))
				collect (format nil "~a" e))
			"-style=file")))

