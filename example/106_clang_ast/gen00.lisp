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
		clang-c/Index.h
		vector)
     
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
       (let (#+nil (unsaved_file (CXUnsavedFile)))
	 #+nil (setf unsaved_file.Filename (string "-I/usr/lib/llvm-11/include/"))
	 (let ((cmdline (string "-I/usr/lib/llvm-11/include/"))
	       (cmdlineArgs ("std::vector<const char*>"))
	       (index (clang_createIndex 0 0))
	     
	       #+nil (unsaved_files (std--vector<CXUnsavedFile>
			       (curly unsaved_file)))
	       
	       
	       )
	   (cmdlineArgs.push_back cmdline)
	   ;; clang_parseTranslationUnit (CXIndex CIdx, const char *source_filename, const char *const *command_line_args, int num_command_line_args, struct CXUnsavedFile *unsaved_files, unsigned num_unsaved_files, unsigned options)
	   (let ((tu (clang_parseTranslationUnit index ;; idx
						 (aref argv 1) ;; source_filename
						 (cmdlineArgs.data)
						 1
						 nullptr
						 0
						 CXTranslationUnit_None)))
	     (unless tu
	       (<< std--cerr
		   (string "failed to parse ")
		   (aref argv 1)
		   std--endl)
	       ))

	   (let ((code (clang_getTranslationUnitSpelling tu)))
	     (<< std--cout
		 (clang_getCString code)
		 std--endl))
	   #+nil(clang_visitChildren
	    (clang_getTranslationUnitCursor tu)
	    (lambda (c parent data)
	      (declare (type CXCursor c parent)
		       (type CXClientData data)
		       )
	      (when (== CXCursor_ParenExpr
			(clang_getCursorKind c))
		(let ((child (clang_getChild c)))
		  (when (clang_isExpression
			 (clang_getCursorKind child))
		    (clang_replaceChildRange parent
					     c
					     child
					     (clang_getNextSibling child)
					     data))
		  ))
	      (return CXChildVisit_Recurse))
	    nullptr)
	   
	   (clang_disposeString code)
	   (clang_disposeTranslationUnit tu)
	   (clang_disposeIndex index)
	   (return 0)))
       ))))

