(declaim (optimize (debug 3)
		   (speed 0)
		   (safety 3)))

(setf sb-ext:*muffled-warnings* nil)

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

#+nil
(trace emit-c)

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
    (loop for e in
	  #+nil `((:name basic1 :code (* 3 (+ 1 2)) :reference "3*(1+2)"))
	  #-nil
	  `((:name basic1 :code (* 3 (+ 1 2)) :reference "3*(1+2)")
	    (:name basic2 :code (* (+ 3 4) 3 (+ 1 2)) :reference "(3+4)*3*(1+2)")
	    (:name basic3 :code (* (+ 3 4) (/ 13 4) (/ (+ 171 2) 5))
	     :lisp-code (* (+ 3 4) (floor 13 4) (floor (+ 171 2) 5))
	     :reference "(3+4)*(13/4)*((171+2)/5)")
	    (:name basic4 :code (* (+ 3 4) (- 7 3))
	     :reference "(3+4)*(7-3)")
	    (:name basic5 :code (+ (+ 3 4) (- 7 3))
	     :reference "(3+4)+(7-3)")
	    (:name basic6 :code (- (+ 3 4) (- 7 3))
	     :reference "(3+4)-(7-3)")
	    (:name basic7 :code (* 2 -1)
	     :reference "2*(-1)")
	    (:name basic8 :code (- 2 -1)
	     :reference "2-(-1)")
	    (:name mod1 :code (% (* 3 5) 4) :lisp-code (mod (* 3 5) 4) :reference "(3*5)%4")
	    (:name mod2 :code (% 74 (* 3 5)) :lisp-code (mod 74 (* 3 5)) :reference "74%(3*5)")
	    (:name mod3 :code (% 74 (/ 17 5)) :lisp-code (mod 74 (floor 17 5)) :reference "74%(17/5)")
	    (:name hex1 :code (+ (hex ad) 3) :lisp-code (+ #xad 3) :reference "0xad+3")
	    (:name div0 :code (/ 17 5)  :lisp-code (floor 17 5) :reference "17/5")
	    (:name div1 :code (+ (/ 17 5) 3) :lisp-code (+ (floor 17 5) 3) :reference "(17/5)+3")
	    (:name div2 :code (+ 3 (/ 17 5)) :lisp-code (+ 3 (floor 17 5)) :reference "3+(17/5)")
	     (:name array0 :code (+ (aref a 0) 3 (/ 17 5)) :lisp-code (+ 1 3 (floor 17 5)) :reference "a[0]+3+(17/5)"
	     :pre "int a[1]={1};")
	     (:name array1 :code (+ (aref a (- (* 12 (+ 3 4))
					      83))
				   3 (/ 17 5))
	     :lisp-code (+ 2 3 (floor 17 5)) :reference "a[(12*(3+4))-83]+3+(17/5)"
	     :pre "int a[2]={1,2};")
	    (:name call0 :code (* 3  (H (+ 3 1)))
		   :lisp-code (* 3 1)
		   :reference "3*H(3+1)"
		   :pre (defun H (n)
			  (declare (type int n)
				   (values int))
			  (return (+ 1 (* 0 n)))))
	    (:name colon0 :code (<< bla--i (+ 3 1))
		   :lisp-code (ash 3 (+ 3 1))
		   :reference "bla::i<<(3+1)"
		   :pre (namespace bla
				   "int i = 3;"))
	    (:name string0
	     :code (+ str
		      (string "hello ")
		      (string "worlds"))
	     :pre "std::string str =\"\";"
	     :lisp-code "hello worlds"
	     :reference "str+\"hello \"+\"worlds\""
	     :supersede-fail (<< std--cout (string "hello world \\033[31mFAIL\\033[0m ")  std--endl))

	    (:name deref0
	     :code (-> car (dot w j))
	     :pre (do0
		   (defclass+ Wheel ()
		     "public:"
		     "int j;"
		     (defmethod Wheel (jj)
		       (declare (type int jj)
				(construct (j jj))
				(values :constructor))))
		   (defclass+ Car ()
		     "public:"
		     "int i;"
		     "Wheel w;"
		     (defmethod Car (ii jj) 
		       (declare (type int ii jj)
				(construct (i ii) (w (Wheel jj)))
				(values :constructor))))
		   "Car car[1]={Car(1,2)};")
	     :lisp-code 2
	     :reference "car->w.j")
	    (:name insertion0
		   :pre (do0
			 (include<> sstream
				    iomanip)
			 "std::ostringstream oss;")
		   :code (paren
			  (comma (<< oss
				     "std::fixed"
				     ("std::setprecision" 3) 3.141590s0)
				 (dot oss (str))))
		   :reference "(oss<<std::fixed<<std::setprecision(3)<<3.141590f, oss.str())"
		   :lisp-code "3.142")
	    (:name singleor0
		   :code (bitwise-not (or 255))
		   :lisp-code (lognot (or 255))
		   :reference "~255"
		   )
	    (:name doubleor0
		   :code (bitwise-not (or #xf0 #x0f))
		   :lisp-code (lognot (logior #xf0 #x0f))
		   :reference ,(format nil "~~(~a | ~a)" #xf0 #x0f)
		   )

	    (:name ternary0
		   :code (? (== 5 3) 1 2)
		   :lisp-code (if (eq 5 3) 1 2)
		   :reference "(5==3) ? 1 : 2")
	    (:name ternary1
		   :code (- 7 (? (== 5 3) 1 2))
		   :lisp-code (- 7 (if (eq 5 3) 1 2))
		   :reference "7-((5==3) ? 1 : 2)")
	    (:name ternary2
		   :code (== (? (== 5 3) 1 2) 7 )
		   :lisp-code (let ((v (eq (if (eq 5 3) 1 2)
					   7)
				       
				       ))
				(if v v 0))
		   :reference "((5==3) ? 1 : 2)==7")
	    )
	  and e-i from 0
	  do
	     (destructuring-bind (&key code name (lisp-code code) reference pre supersede-fail) e
	       (let ((emit-str (emit-c :code code :diag nil))
		     (emit-str-diag (emit-c :code code :diag t)))
		 (if (string= (m-of emit-str) reference)
		     (format s "~2,'0d ~a works~%" e-i emit-str)
		     (format s "~2,'0d ~a should be ~a diag ~a~%" e-i
			     emit-str reference emit-str-diag))
		 (write-source 
		  (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames (format nil "c~2,'0d_~a.cpp" e-i name)
				    *source-dir*))
		  (let ((lisp-var (if (stringp lisp-code)
				    `(string ,lisp-code)
				    (eval lisp-code))))
		   `(do0
		     (include<> cassert
				iostream)
		     ,(if pre
			  pre
			  `(comments "no pre"))
		     (defun main (argc argv)
		       (declare (values int)
				(type int argc)
				(type char** argv))
		       "(void) argc;"
		       "(void) argv;"
		       (comments ,reference)
		      
		       (if (== ,code
			       ,lisp-var)
			   (<< "std::cout" (string ,(format nil "~a OK" (substitute #\' #\" reference)))
			       "std::endl")
			   ,(if supersede-fail
			       supersede-fail
			       `(<< "std::cout" (string ,(format nil "~a \\033[31mFAIL\\033[0m " (substitute #\' #\" reference)))
				   (paren ,code)
				   (string " != ")
				   (paren ,lisp-var)
				   "std::endl")))
		       (return 0))))
		  :format nil
		  :tidy nil))))))


