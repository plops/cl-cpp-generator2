(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/106_clang_ast/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     
     (include<> iostream
		clang-c/Index.h)
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       (unless (== 2 argc)
	 (<< std--cerr
	     (string "usage: ")
	     (aref argv 0)
	     (string " <source file>")
	     std--endl)
	 (return 1))
       (let ((index (clang_createIndex 0 0))
	     (tu (clang_parseTranslationUnit index
					     (aref argv 1)
					     nullptr
					     0
					     nullptr
					     0
					     CXTranslationUnit_None)))
	 (unless tu
	   (<< std--cerr
	     (string "failed to parse ")
	     (aref argv 1)
	     std--endl)
	   )

	 (let ((code (clang_getTranslationUnitSpelling tu)))
	   (<< std--cout
	       (clang_getCString code)
	       std--endl))
	 (clang_disposeString code)
	 (clang_disposeTranslationUnit tu)
	 (clang_disposeIndex index)
	 (return 0))
       ))))

