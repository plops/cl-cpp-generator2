;;;; Unit tests for the parenthesis elision of EMIT-C (:omit-parens t).
;;;;
;;;; Run with:
;;;;   sbcl --noinform --disable-debugger --load t/02_paren_precedence/paren-tests.lisp --quit
;;;; or via t/02_paren_precedence/run.sh
;;;;
;;;; The suite has three independent layers:
;;;;
;;;;   1. STRING TESTS  -- compare the emitted C++ expression against a
;;;;      hand-verified reference string, both for the fully parenthesized mode
;;;;      and for the paren-eliding mode.  This layer also catches bugs that are
;;;;      present in *both* modes (e.g. the missing parentheses around a cast
;;;;      operand).
;;;;
;;;;   2. VALUE TESTS   -- compile one C++ program that evaluates every
;;;;      expression twice (fully parenthesized and paren-eliding) and compares
;;;;      both results with an expected integer.  This is the semantic ground
;;;;      truth: a dropped pair of parentheses changes the value.
;;;;
;;;;   3. HELPER TESTS  -- direct unit tests of EFFECTIVE-OPERATOR and
;;;;      BINDS-LOOSER-P, the two functions the decision is based on.
;;;;
;;;; Layer 2 is skipped (with a warning, not a failure) when no C++ compiler is
;;;; available.

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2" :silent t))

(in-package :cl-cpp-generator2)

(defparameter *failures* 0)
(defparameter *checks* 0)

(defun report (ok name fmt &rest args)
  (incf *checks*)
  (unless ok
    (incf *failures*))
  (format t "~:[FAIL~;ok  ~] ~a~@[ ~a~]~%"
	  ok name (apply #'format nil fmt args)))

(defun emit-str (code &key omit)
  (m-of (emit-c :code code :omit-redundant-parentheses omit)))

;;; ------------------------------------------------------------------
;;; 1+2. expression test table
;;;
;;; :code      s-expression handed to emit-c
;;; :omit      expected C++ text with :omit-parens t
;;; :full      expected C++ text without paren elision (optional)
;;; :value     expected integer value, checked by compiling C++ (optional)
;;; ------------------------------------------------------------------
;;; The integer variables available in the value tests:
;;;   a=7 b=3 c=2 d=5   and the array   arr[4] = {10,20,30,40}
(defparameter *variables*
  "int a = 7, b = 3, c = 2, d = 5;
   int arr[8] = {10, 20, 30, 40, 50, 60, 70, 80};")

(defparameter *expression-tests*
  '(;; --------------------------------------------------------------
    ;; the bug that started this: a negated expression in front of a
    ;; division lost its parentheses, turning -(a-b)/c into -a-b/c
    ;; --------------------------------------------------------------
    (:name unary-minus-over-division
     :code (/ (- (- a b)) c)
     :omit "( -(a-b))/c"
     :full "( -((a)-(b)))/(c)"
     :value -2)			   ; -(7-3)/2 = -4/2 = -2, wrong: -7-3/2 = -8
    (:name unary-minus-over-division-rhs
     :code (/ a (- (- b c)))
     :omit "a/( -(b-c))"
     :value -7)			   ; 7/(-(3-2)) = -7, wrong: 7/-3-2 = -4
    (:name unary-minus-over-multiplication
     :code (* (- (- a b)) c)
     :omit " -(a-b)*c"
     :value -8)			   ; -(7-3)*2 = -8, wrong: -7-3*2 = -13
    (:name unary-minus-over-plus
     :code (+ (- (- a b)) c)
     :omit " -(a-b)+c"
     :value -2)			   ; -(7-3)+2 = -2, wrong: -7-3+2 = -8
    (:name unary-minus-over-modulo
     :code (% (- (- a b)) c)
     :omit "( -(a-b))%c"
     :value 0)			   ; -4 % 2 = 0, wrong: -7-3%2 = -8
    (:name unary-minus-of-sum
     :code (* (- (+ a b)) c)
     :omit " -(a+b)*c"
     :value -20)		   ; -(7+3)*2 = -20, wrong: -7+3*2 = -1
    (:name unary-minus-of-product
     :code (- (* a b))
     ;; unary minus binds tighter than *, the parentheses are redundant but
     ;; harmless -- correctness first
     :omit " -(a*b)"
     :value -21)
    (:name unary-minus-nested
     :code (- (- (- a b)))
     :omit " - -(a-b)"
     :value 4)
    (:name unary-minus-of-symbol
     :code (+ (- a) b)
     :omit " -a+b"
     :value -4)
    (:name unary-minus-shift
     :code (<< a (- (- c b)))
     :omit "a<< -(c-b)"
     :value 14)			   ; 7 << -(2-3) = 7 << 1
    (:name unary-minus-in-ternary-condition
     :code (? (- (- a b)) c d)
     :omit " -(a-b) ? c : d"
     :value 2)
    (:name unary-minus-in-comparison
     :code (== (- (- a b)) c)
     :omit " -(a-b)==c"
     :value 0)
    (:name unary-minus-in-call
     :code (abs (- (- a b)))
     :omit "abs( -(a-b))"
     :value 4)
    (:name unary-minus-of-array-element
     :code (- (aref arr 1))
     :omit " -arr[1]"
     :value -20)
    (:name unary-minus-index
     :code (aref arr (- (- b a)))
     ;; the index is bracketed anyway, but the grouping has to survive
     :omit "arr[( -(b-a))]"
     :value 50)			   ; arr[-(3-7)] = arr[4]
    ;; --------------------------------------------------------------
    ;; C style cast: the cast only binds to the next unary expression
    ;; --------------------------------------------------------------
    (:name cast-of-sum
     :code (cast int (+ a b))
     :omit "(int) (a+b)"
     :full "(int) ((a)+(b))"
     :value 10)
    (:name cast-of-symbol
     :code (cast int a)
     :omit "(int) a"
     :full "(int) a"
     :value 7)
    (:name cast-in-product
     :code (* (cast int a) b)
     :omit "(int) a*b"
     :value 21)
    ;; --------------------------------------------------------------
    ;; member access binds tighter than every operator
    ;; --------------------------------------------------------------
    (:name dot-of-expression
     :code (dot (- (- a b)) c)
     :omit "( -(a-b)).c")
    (:name arrow-of-expression
     :code (-> (- (- a b)) c)
     :omit "( -(a-b))->c")
    ;; --------------------------------------------------------------
    ;; single argument forms
    ;; --------------------------------------------------------------
    (:name reciprocal
     :code (/ a)
     :omit "1.0/a")
    (:name reciprocal-of-sum
     :code (/ (+ a b))
     :omit "1.0/(a+b)")
    (:name single-or-stays-bare
     :code (bitwise-not (or 255))
     :omit "~255"
     :value -256)
    (:name single-logior-in-bitand
     :code (and 5 (logior (== 1 1)))
     :omit "5 & 1==1"
     :value 1)
    ;; --------------------------------------------------------------
    ;; chained comparison (a<b<c expands to a<b && b<c)
    ;; --------------------------------------------------------------
    (:name chained-compare
     :code (<= c b a)
     :omit "c<=b && b<=a"
     :value 1)
    (:name chained-compare-negated
     :code (not (<= c b a))
     :omit "!(c<=b && b<=a)"
     :value 0)
    (:name chained-compare-in-bitor
     :code (or (<= c b a) 0)
     :omit "(c<=b && b<=a) | 0"
     :value 1)
    ;; --------------------------------------------------------------
    ;; equal precedence: the operand on the side that the
    ;; associativity does not favour has to keep its parentheses
    ;; --------------------------------------------------------------
    (:name nested-ternary-in-condition
     ;; ?: is right associative: a?b:c ? d : e would regroup
     :code (? (? a b c) d 1)
     :omit "(a ? b : c) ? d : 1"
     :value 5)
    (:name compare-of-compare
     :code (< a (< b c))
     :omit "a<(b<c)"
     :value 0)
    (:name eq-of-eq
     :code (== a (== b c))
     :omit "a==(b==c)"
     :value 0)
    (:name shift-of-shift
     :code (<< a (<< b c))
     :omit "a<<(b<<c)"
     :value 28672)
    (:name product-of-quotient
     :code (* a (/ b c))
     :omit "a*(b/c)"
     :value 7)
    (:name quotient-of-product
     :code (/ (* a b) c)
     :omit "(a*b)/c"
     :value 10)
    (:name sum-of-sum-stays-flat
     ;; + is associative, no parentheses needed
     :code (+ a (+ b c))
     :omit "a+b+c"
     :value 12)
    ;; --------------------------------------------------------------
    ;; forms that used to signal an error instead of emitting code
    ;; --------------------------------------------------------------
    (:name compound-xor-assign
     ;; ^= had no entry in *precedence* (it was spelled ^-), which made
     ;; paren* compare NIL with a number
     :code (^= a (+ b c))
     :omit "a^=b+c")
    (:name bitand-needs-parens-in-sum
     ;; (and ...) emits a & b without brackets, & binds looser than +
     :code (+ (and a b) c)
     :omit "(a & b)+c"
     :value 5)
    (:name bitand-form-brings-own-parens
     ;; (& ...) always emits its own brackets, no second pair
     :code (+ (& a b) c)
     :omit "(a&b)+c"
     :value 5)
    ;; --------------------------------------------------------------
    ;; regression cases that already worked, kept as a safety net
    ;; (references taken from t/01_paren/gen00.lisp)
    ;; --------------------------------------------------------------
    (:name basic1 :code (* 3 (+ 1 2)) :omit "3*(1+2)" :value 9)
    (:name basic2 :code (+ (* 3 1) 2) :omit "3*1+2" :value 5)
    (:name basic3 :code (* (+ 3 4) 3 (+ 1 2)) :omit "(3+4)*3*(1+2)" :value 63)
    (:name basic4 :code (* (+ 3 4) (/ 13 4) (/ (+ 171 2) 5))
     :omit "(3+4)*(13/4)*((171+2)/5)" :value 714)
    (:name basic5 :code (* (+ 3 4) (- 7 3)) :omit "(3+4)*(7-3)" :value 28)
    (:name basic6 :code (+ (+ 3 4) (- 7 3)) :omit "3+4+(7-3)" :value 11)
    (:name basic7 :code (- (+ 3 4) (- 7 3)) :omit "(3+4)-(7-3)" :value 3)
    (:name basic8 :code (- (- 7 3) (+ 3 4)) :omit "(7-3)-(3+4)" :value -3)
    (:name basic9 :code (+ (- 7 3) (+ 3 4)) :omit "(7-3)+3+4" :value 11)
    (:name basica :code (* 2 -1) :omit "2* -1" :value -2)
    (:name basicb :code (- 2 -1) :omit "2- -1" :value 3)
    (:name mod1 :code (% (* 3 5) 4) :omit "(3*5)%4" :value 3)
    (:name mod2 :code (% 74 (* 3 5)) :omit "74%(3*5)" :value 14)
    (:name mod3 :code (% 74 (/ 17 5)) :omit "74%(17/5)" :value 2)
    (:name hex1 :code (+ (hex ad) 3) :omit "0xad+3" :value 176)
    (:name div0 :code (/ 17 5) :omit "17/5" :value 3)
    (:name div1 :code (+ (/ 17 5) 3) :omit "(17/5)+3" :value 6)
    (:name div2 :code (+ 3 (/ 17 5)) :omit "3+(17/5)" :value 6)
    (:name array0 :code (+ (aref arr 0) 3 (/ 17 5)) :omit "arr[0]+3+(17/5)" :value 16)
    (:name array1 :code (+ (aref arr (- (* 1 (+ 1 1)) 1)) 3 (/ 17 5))
     :omit "arr[((1*(1+1))-1)]+3+(17/5)" :value 26)
    (:name colon0 :code (<< (scope bla i) (+ 3 1)) :omit "bla::i<<3+1")
    (:name ternary0 :code (? (== 5 3) 1 2) :omit "5==3 ? 1 : 2" :value 2)
    (:name ternary1 :code (- 7 (? (== 5 3) 1 2)) :omit "7-(5==3 ? 1 : 2)" :value 5)
    (:name ternary2 :code (== (? (== 5 3) 1 2) 7) :omit "(5==3 ? 1 : 2)==7" :value 0)
    (:name ternary3 :code (== (paren (? (- 5 3) 1 2)) 7) :omit "((5-3) ? 1 : 2)==7" :value 0)
    (:name unary0 :code (== -1 2) :omit " -1==2" :value 0)
    (:name unary1 :code (== 2 -1) :omit "2== -1" :value 0)
    (:name logorand0 :code (logior 1 (logand 0 1)) :omit "1||0&&1" :value 1)
    (:name doubleor0 :code (bitwise-not (or 240 15)) :omit "~(240 | 15)" :value -256)
    (:name assigneq0 :code (= d (== a 7)) :omit "d=a==7" :value 1)
    (:name deref0 :code (dot (-> pcar w) j) :omit "pcar->w.j")
    ;; --------------------------------------------------------------
    ;; non associative operators keep their grouping
    ;; --------------------------------------------------------------
    (:name div-of-div :code (/ (/ a b) c) :omit "(a/b)/c" :value 1)
    (:name div-by-div :code (/ a (/ b c)) :omit "a/(b/c)" :value 7)
    (:name minus-of-minus :code (- a (- b c)) :omit "a-(b-c)" :value 6)
    (:name minus-chain :code (- (- a b) c) :omit "(a-b)-c" :value 2)
    ;; + binds tighter than <<, so no parentheses are needed here
    (:name shift-of-sum :code (<< (+ a b) c) :omit "a+b<<c" :value 40)
    (:name bitand-of-eq :code (and (== a 7) (== b 3)) :omit "a==7 & b==3" :value 1)
    (:name booland-of-eq :code (logand (== a 7) (== b 3)) :omit "a==7&&b==3" :value 1)
    (:name ternary-in-product :code (* (? a b c) d) :omit "(a ? b : c)*d" :value 15)
    (:name not-of-sum :code (not (+ a b)) :omit "!(a+b)" :value 0)
    (:name deref-of-sum :code (deref (+ pa b)) :omit "*(pa+b)")
    (:name bitnot-of-minus :code (bitwise-not (- a b)) :omit "~(a-b)" :value -5)))

;;; ------------------------------------------------------------------
;;; layer 1: string comparison
;;; ------------------------------------------------------------------
(defun run-string-tests ()
  (format t "~&== string tests ==~%")
  (dolist (e *expression-tests*)
    (destructuring-bind (&key name code omit full value) e
      (declare (ignore value))
      (let ((got-omit (handler-case (emit-str code :omit t)
			(condition (c) (format nil "<error ~a>" c)))))
	(report (string= got-omit omit) name
		"omit: got ~s expected ~s" got-omit omit))
      (when full
	(let ((got-full (handler-case (emit-str code)
			  (condition (c) (format nil "<error ~a>" c)))))
	  (report (string= got-full full) name
		  "full: got ~s expected ~s" got-full full))))))

;;; ------------------------------------------------------------------
;;; layer 2: compile and compare values
;;; ------------------------------------------------------------------
(defun cxx-compiler ()
  (loop for c in '("/usr/bin/g++" "/usr/bin/clang++" "/usr/bin/c++")
	when (probe-file c) return c))

(defun run-value-tests ()
  (format t "~&== value tests ==~%")
  (let ((compiler (cxx-compiler))
	(cases (remove-if-not (lambda (e) (getf e :value)) *expression-tests*)))
    (if (not compiler)
	(format t "SKIP no C++ compiler found~%")
	(let* ((dir (asdf:system-relative-pathname
		     'cl-cpp-generator2 "t/02_paren_precedence/build/"))
	       (src (merge-pathnames "value_tests.cpp" dir))
	       (exe (merge-pathnames "value_tests" dir)))
	  (ensure-directories-exist dir)
	  (with-open-file (s src :direction :output :if-exists :supersede
				 :if-does-not-exist :create)
	    (format s "// generated by t/02_paren_precedence/paren-tests.lisp~%")
	    (format s "#include <cstdio>~%#include <cstdlib>~%~%")
	    (format s "int main() {~%  int fails = 0;~%")
	    (dolist (e cases)
	      (destructuring-bind (&key name code omit full value) e
		(declare (ignore omit full))
		(let ((full-str (emit-str code))
		      (omit-str (emit-str code :omit t)))
		  (format s "  {~%    ~a~%" *variables*)
		  (format s "    (void)a;(void)b;(void)c;(void)d;(void)arr;~%")
		  (format s "    long expected = ~a;~%" value)
		  (format s "    long vfull = (~a);~%" full-str)
		  (format s "    long vomit = (~a);~%" omit-str)
		  (format s "    if (vfull != expected) { ~
                               std::printf(\"FAIL ~a full: %ld != %ld  [%s]\\n\", ~
                               vfull, expected, \"~a\"); fails++; }~%"
			  name (substitute #\' #\" full-str))
		  (format s "    if (vomit != expected) { ~
                               std::printf(\"FAIL ~a omit: %ld != %ld  [%s]\\n\", ~
                               vomit, expected, \"~a\"); fails++; }~%"
			  name (substitute #\' #\" omit-str))
		  (format s "    if (vfull == expected && vomit == expected) ~
                               std::printf(\"ok   ~a = %ld  [%s]\\n\", vomit, \"~a\");~%"
			  name (substitute #\' #\" omit-str))
		  (format s "  }~%"))))
	    (format s "  std::printf(\"%d value failures\\n\", fails);~%")
	    (format s "  return fails == 0 ? 0 : 1;~%}~%"))
	  (let ((compile-ok
		  (zerop (sb-ext:process-exit-code
			  (sb-ext:run-program compiler
					      (list "-std=c++17" "-O0" "-w"
						    "-o" (namestring exe)
						    (namestring src))
					      :output *standard-output*
					      :error *standard-output*)))))
	    (report compile-ok "value-tests-compile" "~a" (namestring src))
	    (when compile-ok
	      (let ((code (sb-ext:process-exit-code
			   (sb-ext:run-program (namestring exe) nil
					       :output *standard-output*
					       :error *standard-output*))))
		(report (zerop code) "value-tests-run"
			"~a expression~:p compiled from ~a" (length cases)
			(namestring src)))))))))

;;; ------------------------------------------------------------------
;;; layer 3: helper functions
;;; ------------------------------------------------------------------
(defparameter *effective-operator-tests*
  '(((- a b) -)			  ; binary minus
    ((- a) -unary)		  ; unary minus binds much tighter
    ((- (+ a b)) -unary)
    ((+ a b) +)
    ((+ a) nil)			  ; (+ x) emits just x
    ((+ (- a b)) -)		  ; ... and inherits its precedence
    ((or 255) nil)		  ; single argument chain emits just 255
    ((or a b) or)
    ((paren a) nil)		  ; brings its own brackets
    ((curly a b) nil)
    ((& a b) nil)		  ; (& ...) is always parenthesized
    ((< a b c) logand)		  ; chained comparison joins with &&
    ((< a b) <)
    ((aref a 1) aref)
    ((foo a b) foo)		  ; unknown -> function call
    (a nil)
    (1 nil)))

(defun run-helper-tests ()
  (format t "~&== helper tests ==~%")
  (dolist (e *effective-operator-tests*)
    (destructuring-bind (arg expected) e
      (let ((got (effective-operator arg)))
	(report (eq got expected) (format nil "effective-operator ~s" arg)
		"got ~s expected ~s" got expected))))
  ;; binds-looser-p
  (dolist (e '(((- a b) / t)		 ; a-b binds looser than /
	       ((* a b) / nil)		 ; a*b binds equally tight
	       ((- a) / nil)		 ; -a binds tighter
	       ((- a b) cast t)		 ; (int)(a-b) needs the parens
	       (a cast nil)
	       ((foo a) cast nil)
	       ((- a b) dot t)		 ; (a-b).x
	       ((aref a 1) dot nil)))
    (destructuring-bind (arg parent expected) e
      (let ((got (and (binds-looser-p arg parent) t)))
	(report (eq got expected)
		(format nil "binds-looser-p ~s ~s" arg parent)
		"got ~s expected ~s" got expected)))))

;;; ------------------------------------------------------------------
;;; layer 4: randomized differential test
;;;
;;; The fully parenthesized mode is the oracle: for every randomly generated
;;; expression the paren-eliding output has to evaluate to the same value.
;;; Only operators that are free of undefined behaviour for the chosen operand
;;; range are generated (no shifts, divisors are positive literals).
;;; ------------------------------------------------------------------
(defparameter *random-atoms* '(a b c d 1 2 3 7 (aref arr 0) (aref arr 3)))
(defparameter *random-safe-divisors* '(2 3 5 7))

(defun random-expr (depth state)
  (if (or (<= depth 0)
	  (< (random 1.0 state) 0.3))
      (nth (random (length *random-atoms*) state) *random-atoms*)
      (let ((kind (random 21 state))
	    (l (random-expr (1- depth) state))
	    (r (random-expr (1- depth) state)))
	(case kind
	  (0 `(+ ,l ,r))
	  (1 `(- ,l ,r))
	  (2 `(* ,l ,r))
	  (3 `(/ ,l ,(nth (random (length *random-safe-divisors*) state)
			  *random-safe-divisors*)))
	  (4 `(% ,l ,(nth (random (length *random-safe-divisors*) state)
			  *random-safe-divisors*)))
	  (5 `(- ,l))			; unary minus - the interesting one
	  (6 `(not ,l))
	  (7 `(bitwise-not ,l))
	  (8 `(== ,l ,r))
	  (9 `(< ,l ,r))
	  (10 `(and ,l ,r))		; bitwise &
	  (11 `(logand ,l ,r))		; &&
	  (12 `(? ,l ,r ,(random-expr (1- depth) state)))
	  (13 `(cast int ,l))
	  (14 `(or ,l ,r))		; bitwise |
	  (15 `(^ ,l ,r))
	  (16 `(logior ,l ,r))		; ||
	  (17 `(!= ,l ,r))
	  (18 `(<= ,l ,r))
	  ;; mask the operand so that the shift stays defined
	  (19 `(<< (and ,l 7) ,(1+ (random 3 state))))
	  (20 `(>> (and ,l 255) ,(1+ (random 3 state))))))))

(defun run-random-tests (&key (count 400) (depth 3) (seed 20260830))
  (format t "~&== randomized differential tests ==~%")
  (let ((compiler (cxx-compiler)))
    (if (not compiler)
	(format t "SKIP no C++ compiler found~%")
	(let* ((state (sb-ext:seed-random-state seed))
	       (dir (asdf:system-relative-pathname
		     'cl-cpp-generator2 "t/02_paren_precedence/build/"))
	       (src (merge-pathnames "random_tests.cpp" dir))
	       (exe (merge-pathnames "random_tests" dir))
	       (exprs (loop repeat count collect (random-expr depth state))))
	  (ensure-directories-exist dir)
	  (with-open-file (s src :direction :output :if-exists :supersede
				 :if-does-not-exist :create)
	    (format s "// generated by t/02_paren_precedence/paren-tests.lisp~%")
	    (format s "#include <cstdio>~%~%int main() {~%  int fails = 0;~%")
	    (loop for e in exprs
		  and i from 0
		  do (let ((full (emit-str e))
			   (omit (emit-str e :omit t)))
		       (format s "  {~%    ~a~%" *variables*)
		       (format s "    (void)a;(void)b;(void)c;(void)d;(void)arr;~%")
		       (format s "    long vfull = (~a);~%" full)
		       (format s "    long vomit = (~a);~%" omit)
		       (format s "    if (vfull != vomit) { std::printf(\"FAIL random ~a: ~
                                  %ld != %ld\\n      sexp %s\\n      full %s\\n      omit %s\\n\", ~
                                  vfull, vomit, \"~a\", \"~a\", \"~a\"); fails++; }~%"
			       i
			       (substitute #\' #\"
					   (let ((*print-pretty* nil))
					     (format nil "~a" e)))
			       (substitute #\' #\" full)
			       (substitute #\' #\" omit))
		       (format s "  }~%")))
	    (format s "  std::printf(\"%d of ~a random expressions disagree\\n\", fails, ~a);~%"
		    count count)
	    (format s "  return fails == 0 ? 0 : 1;~%}~%"))
	  (let ((compile-ok
		  (zerop (sb-ext:process-exit-code
			  (sb-ext:run-program compiler
					      (list "-std=c++17" "-O0" "-w"
						    "-o" (namestring exe)
						    (namestring src))
					      :output *standard-output*
					      :error *standard-output*)))))
	    (report compile-ok "random-tests-compile" "~a" (namestring src))
	    (when compile-ok
	      (report (zerop (sb-ext:process-exit-code
			      (sb-ext:run-program (namestring exe) nil
						  :output *standard-output*
						  :error *standard-output*)))
		      "random-tests-run"
		      "~a random expressions, depth ~a, seed ~a" count depth seed)))))))

;;; ------------------------------------------------------------------
(run-string-tests)
(run-helper-tests)
(run-value-tests)
(run-random-tests)

(format t "~&~%~a check~:p, ~a failure~:p~%" *checks* *failures*)
(when (< 0 *failures*)
  (sb-ext:exit :code 1 :abort t))
