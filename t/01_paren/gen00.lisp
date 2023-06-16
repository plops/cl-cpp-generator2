(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
   (progn
    (defparameter *source-dir* #P"t/01_paren/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)

  (with-open-file (s "/home/martin/stage/cl-cpp-generator2/t/01_paren/source00/strings.txt"
			    :if-exists :supersede
			    :if-does-not-exist :create
			     :direction :output)

    ;; the following tests check if paren* avoids redundant parentheses
    (loop for e in `((:name basic1 :code (* 3 (+ 1 2)) :reference "3*(1+2)")
		     (:name basic2 :code (* (+ 3 4) 3 (+ 1 2)) :reference "(3+4)*3*(1+2)")
		     (:name basic3 :code (* (+ 3 4) (/ 13 4) (/ (+ 171 2) 5))
		      :lisp-code (* (+ 3 4) (floor 13 4) (floor (+ 171 2) 5))
			    :reference "(3+4)*3/4*(1+2)/5")
		      (:name basic4 :code (* (+ 3 4) (- 7 3))
			    :reference "(3+4)*(7-3)")
		     (:name basic5 :code (+ (+ 3 4) (- 7 3))
			    :reference "3+4+7-3")
		     (:name basic6 :code (- (+ 3 4) (- 7 3))
			    :reference "3+4-(7-3)")
		     (:name mod1 :code (% (* 3 5) 4) :lisp-code (mod (* 3 5) 4) :reference "(3*5)%4")
		     (:name mod2 :code (% 74 (* 3 5)) :lisp-code (mod 74 (* 3 5)) :reference "74%(3*5)")
		     (:name mod3 :code (% 74 (/ 17 5)) :lisp-code (mod 74 (floor 17 5)) :reference "74%(17/5)")
		     (:name hex1 :code (+ (hex ad) 3) :lisp-code (+ #xad 3) :reference "0xad+3")
		     (:name div0 :code (/ 17 5)  :lisp-code (floor 17 5) :reference "17/5")
		     (:name div1 :code (+ (/ 17 5) 3) :lisp-code (+ (floor 17 5) 3) :reference "17/5+3")
		     (:name div2 :code (+ 3 (/ 17 5)) :lisp-code (+ 3 (floor 17 5)) :reference "3+17/5"))
	  and e-i from 0
	  do
	     (destructuring-bind (&key code name (lisp-code code) reference) e
	       (let ((emit-str (emit-c :code code :diag nil))
		     (emit-str-diag (emit-c :code code :diag t)))
		 (if (string= emit-str reference)
		     (format s "~2,'0d ~a works diag ~a~%" e-i emit-str emit-str-diag)
		     (format s "~2,'0d ~a should be ~a diag ~a~%" e-i
			     emit-str reference emit-str-diag))
		 (write-source
		  (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames (format nil "c~2,'0d_~a.cpp" e-i name)
				    *source-dir*))
		  `(do0
		    (include<> cassert)
		    (defun main (argc argv)
		      (declare (values int)
			       (type int argc)
			       (type char** argv))
		      (comments ,reference)
		      (assert (== ,code
				  ,(eval lisp-code)))
		      (return 0)))))))))




