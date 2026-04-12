(in-package :cl-cpp-generator2)

(write-source
 (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames "helpers.h" *source-dir*))
 `(do0
   "#pragma once"
   "constexpr auto kOk = 0;"
   (space enum class "Error : int" (curly (comma (= Unknown_Word 1)
						 (= Stack_Error 2)
						 (= Compile_Error 3)
						 )))
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
