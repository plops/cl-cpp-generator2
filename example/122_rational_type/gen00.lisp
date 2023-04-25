(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)


(progn
   (progn
    (defparameter *source-dir* #P"example/122_rational_type/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     
     (include<> ratio
		)

     (do0
      "#define FMT_HEADER_ONLY"
      (include "core.h"))

     (defun convert_hp (old_hp old_maxhp new_maxhp)
       (declare (type float new_maxhp old_hp old_maxhp)
		(values float))
       (return (* new_maxhp (/ old_hp old_maxhp))))

     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       ,(lprint :msg (multiple-value-bind
			   (second minute hour date month year day-of-week dst-p tz)
			 (get-decoded-time)
		       (declare (ignorable dst-p))
		       (format nil "generation date ~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
			       hour
			       minute
			       second
			       (nth day-of-week *day-names*)
			       year
			       month
			       date
			       (- tz))))
       (let ((old_hp 1s0)
	     (old_maxhp 55s0)
	     (new_maxhp 55s0)
	     (dold_hp 1d0)
	     (dold_maxhp 55d0)
	     (dnew_maxhp 55d0)
	     
	     ;(r1over55 "std::ratio<1,55>")
	     ;(r55over1  "std::ratio<55,1>")
	     )
	 ,(lprint :msg "when the computation is <1, this is due to a floating point rounding error and leads to a x")
	 ,(lprint :msg "func" :vars `((convert_hp old_hp old_maxhp new_maxhp)))
	 
	 ,@(loop for e in `((:name buggy :code (* new_maxhp (/ old_hp
							       old_maxhp)))
			    (:name buggy_double :code (* dnew_maxhp (/ dold_hp
								       dold_maxhp)))
			    (:name mul_first :code (/ (* new_maxhp old_hp)
						      old_maxhp))
					;(:name cpp_ratio :code "std::ratio_add<r55over1,r1over55>")
			    (:name lisp_ratio :code ,(* 55 (/ 1 55)))
			    (:name lisp_float :code ,(* 55s0 (/ 1s0 55s0))))
		 collect
		 (destructuring-bind (&key name code) e
		   (lprint :msg (format nil "~a" name)
			   :vars `(,code)))))


      
       (return 0)))))



