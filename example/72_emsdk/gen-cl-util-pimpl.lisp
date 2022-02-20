(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-commonlisp-generator")
  (ql:quickload "alexandria"))
(in-package :cl-commonlisp-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-commonlisp-generator/example/00_test")
  (defparameter *code-file* "run_00_test")
  (defparameter *source* (format nil "~a/source/" *path*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  #+nil (defun lprint (cmd &optional rest)
    `(when debug
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
		   (format (- (time.time) start_time)
			   ,@rest)))))
  (let* (
	 (code
	   `(toplevel
	      (defun test1 (a)
		(+ a 3))
	      (defun test2 (a &optional b (c 4))
		(let* ((q (* b c))
		      (p  (* q q))) (+ a 3 b p c)))
	      (test2 3 4)
	      (defun test3 (a &key b (c 4))
		(let ((q (* 2 (string "b")))
		      (p (+ b c)))
		  (+ a 3 (* q p))))

	      (test3 1 :b 3)
	      (setf
	       _code_git_version
	       (string ,(let ((str (with-output-to-string (s)
				     (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			  (subseq str 0 (1- (length str)))))
	       _code_repository (string ,(format nil "https://github.com/plops/cl-commonlisp-generator/tree/master/example/00_test/gen00.lisp"))
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
				  (- tz)))))
	     
	     
	     )))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    ))

