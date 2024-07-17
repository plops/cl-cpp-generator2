(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ()
  (defparameter *source-dir* #P"example/155_hana_liballocs/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  ;(load "util.lisp")
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      ;boost/hana/define_struct.hpp
      boost/hana.hpp
      string
      iostream
      )
     "namespace hana = boost::hana;"
     ,(let ((name 'Person)
	    (members `((:var name :type std--string)
		       (:var age :type "unsigned short"))))
       `(space struct
	      ,name
	      (progn
		(BOOST_HANA_DEFINE_STRUCT
		 ,name
		 ,@(loop for e in members
			 collect
			 (destructuring-bind (&key var type) e
			   `(paren ,type ,var)))))))
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (let ((john (Person (curly (string "John")
				  30)))))
       (hana--for_each john
		       (hana--fuse
			(lambda (name member)
			  (<< std--cout ("hana::to<char const*>" name)
			      (string ": ")
			      member (string "\\n")))))
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))
