(in-package :cl-cpp-generator2)

(write-source
 (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames "main.cpp" *source-dir*))
 `(do0
   (include<> algorithm
	      array
	      charconv
	      cctype
	      cstdlib
	      functional
	      iostream
	      libgccjit++.h
	      memory
	      optional
	      sstream
	      string
	      string_view
	      unordered_map
	      utility
	      vector)
   (include "helpers.h"
            "ForthVM.h")
   "using namespace gccjit;"
   (space namespace
	  (progn

	    "using CompiledWord = int (*)(ForthVM*);"
	    
	    
	    

	    (defun main ()
	      (declare (values int))
	      (let ((vm (space ForthVM (curly)))
		    (line (space std--string (curly))))
		(while (std--getline std--cin line)
		       (case (vm.process_line line)
			 (Error (error)
			  (when (== error Error--Compile_Error)
			    (vm.abort_pending_definition))
			  (let ((error_name_str (error_name error)))
			    ,(lprint :vars `(error_name_str)))))
		       )
		(return 0)))
	    :omit-parens t
	    :format t
	    :tidy t))))
