(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "130_torch_optim")
  (defparameter *idx* "01")
  (defparameter *path* (format nil "/home/martin/stage/cl-cpp-generator2/train_llm/" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key msg vars)
    `(do0 ;when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}"
				    msg
				    (mapcar (lambda (x)
					      (emit-py :code x))
					    vars)))
                   (format  (- (time.time) start_time)
                            ,@vars)))))

  
  
  (let* ((notebook-name "collect_examples_as_json")
	 (cli-args `(#+nil (:short "c" :long "chunk_size" :type int :default 500
		      :help "Approximate number of words per chunk")
		     #+nil (:short "p" :long "prompt" :type str
		      :default (string "Summarize the following video transcript as a bullet list.")
		      :help "The prompt to be prepended to the output file(s).")))
	 )
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "python -m venv ~/pytorch_env"
		 ". ~/pytorch_env/bin/activate"
		 "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu11"
		 "pip install lmfit")
       (imports (os
		 time
		 json
		 pathlib
		 re
		 sys
		 ;torch
		 ;(pd pandas)
		 ;lmfit
		 ))

       #+nil(do0
	(setf start_time (time.time)
	      debug True)
	(setf
	 _code_git_version
	 (string ,(let ((str (with-output-to-string (s)
			       (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
		    (subseq str 0 (1- (length str)))))
	 _code_repository (string ,(format nil
					   "https://github.com/plops/cl-py-generator/tree/master/example/~a/source/"
					   *project*))
	 _code_generation_time
	 (string ,(multiple-value-bind
			(second minute hour date month year day-of-week dst-p tz)
		      (get-decoded-time)
		    (declare (ignorable dst-p))
		    (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
			    hour
			    minute
			    second
			    (nth day-of-week *day-names*)
			    year
			    month
			    date
			    (- tz))))))

       (do0
	(setf directory (dot pathlib (Path (string "/home/martin/stage/cl-cpp-generator2"))))


	(setf training_data (list))
	,(let ((l-ext `("cpp" "c" "h" "hpp" "cu" "cl")))
	   `(for (f (dot (/ directory (string "example")) (rglob (string "gen*.lisp"))))

		 (do0
		  (comments "exclude python generating files")
		  (setf content (f.read_text))
		  (when (re.search (rstring3 "\\(ql:quickload \"cl-py-generator\"\\)"
					     )
				   content)
		    continue))
		 (comments "genXX.lisp -> sourceXX")
	       (setf output_dir (/ f.parent (dot (string "source{}")
						 (format (aref f.stem (slice 3 5))))))
	       (if (output_dir.exists)
		   (do0
		    (setf output_files (+ ,@(loop for e in l-ext
						  collect
						  `("list" (dot output_dir (glob (string ,(format nil "*.~a" e))))))))
		    (if (< 0 (len output_files))
			(print (fstring "Info 1: Found match {f} {len(output_files)}."))
			(do0
			 (print (fstring "Warning 1: No matches in output directory for {f}."))
			 continue)))
		   (do0
		    (setf content (f.read_text))
		    (setf match (re.search (rstring3 "\\(defparameter \\*source-dir\\* .*\\\"(.*)\\\"\\)")
					   content))
		    (if match
			(do0
		       4
			 (setf output_dir (/ directory
					     (match.group 1))
			       output_files (+ ,@(loop for e in l-ext
						       collect
						       `("list" (dot output_dir (glob (string ,(format nil "*.~a" e))))))))
			 (if (< 0 (len output_files))
			     (print (fstring "Info 2: Found match {f} {len(output_files)}."))
			     (do0
			      (print (fstring "Warning 2: Not enough files for {f} in {output_dir} gp1={match.group(1)}."))
			      (print (fstring "Warning 4: match={match} ls {output_dir}={output_files}."))
			      
			      continue)))
			(do0
			 (print (fstring "Warning 3: Could not determine output directory for {f}."))
			 continue)))))))
       
       
       ))))

