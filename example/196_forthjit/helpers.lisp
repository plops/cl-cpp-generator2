(in-package :cl-cpp-generator2)

(write-source
 (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames "helpers.h" *source-dir*))
 `(do0
   "#pragma once"
   (space enum class "Error : int" (curly (comma (= kOk 0) (= Unknown_Word 1)
						 (= Stack_Error 2)
						 (= Compile_Error 3)
                                                 (= Dictionary_Full 4)
                                                 (= Invalid_Fuel 5)
						 )))
   
   (defun error_name (error)
     (declare (type "Error" error)
              (values "inline const char*"))
     (case error
       ,@(loop for e in `(Unknown_Word Stack_Error Compile_Error Dictionary_Full Invalid_Fuel kOk)
               collect
               `(,(format nil "Error::~a" e)
                 (return (string ,e )))))
     (return (string "Error")))
   (space enum class Primitive
	  (curly
	   ,@(mapcar #'second *l-prim*)))
   ,@(loop for e in `((:name OperationKind :values (Literal
						    Primitive
						    CallWord
						    If))
		      (:name ParseMode :values (Immediate Definition))
		      (:name SequenceStop :values (End Else Then)))
	   collect
	   (destructuring-bind (&key name values) e
	     `(space enum class ,name
		     (curly ,@values)))))
 :format t
 :omit-parens t)
