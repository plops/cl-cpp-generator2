(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/106_clang_ast/source01/")
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
		
		
		clang/Frontend/CompilerInstance.h 
		clang/AST/ASTContext.h 
		clang/AST/ASTConsumer.h 
		clang/AST/RecursiveASTVisitor.h 
		clang/Frontend/TextDiagnosticPrinter.h 
		clang/Frontend/FrontendActions.h 
		clang/Tooling/CommonOptionsParser.h 
		clang/Tooling/Tooling.h 
		)

     (defclass+ ASTPrinter "public clang::ASTConsumer"
       "public:"
       (defmethod HandleTopLevelDecl (dg)
	 (declare (type "clang::DeclGroupRef" dg)
		  (values bool)
		  (virtual))
	 (for-range (&d dg)
		    (d->dump))
	 (return true)))
     
     (defclass+ ASTAction "public clang::ASTFrontendAction"
       "public:"
       (defmethod CreateASTConsumer (ci file)
	 (declare (type "clang::CompilerInstance&" ci)
		  (type "llvm::StringRef" file)
		  (values "std::unique_ptr<clang::ASTConsumer>")
		  (virtual))
	 (return (std--unique_ptr<clang--ASTConsumer> (new (ASTPrinter))))
	 ))
     
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type "const char**" argv))
       (let ((op (clang--tooling--CommonOptionsParser  argc argv
						       (string "ast dump tool")
						       #+nil (llvm--cl--OptionCategory
							      (string "ast dump tool"))
		  ))
	     (tool (clang--tooling--ClangTool (op.getCompilations)
					      (op.getSourcePathList))))
	 (return (tool.run
		  (dot (clang--tooling--newFrontendActionFactory<ASTAction>)
		       (get)))))
))))

