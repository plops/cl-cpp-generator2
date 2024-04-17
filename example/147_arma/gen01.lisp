(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

;; https://conradsanderson.id.au/pdfs/sanderson_curtin_armadillo_pasc_2017.pdf
;; % is element wise product
;; https://www.youtube.com/watch?v=PRy8DmRRr6c
;; https://www.youtube.com/watch?v=Ppui7qs9drs
;; sci-libs/armadillo arpack blas -doc -examples lapack -mkl superlu -test
(let ()
  (defparameter *source-dir* #P"example/147_arma/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  
  #+nil
  (let* ((name `CpuAffinityManagerWithGui)
	 (members `((selectedCpus :type std--vector<bool> ; std--bitset<12>
				  :initform nil)
		    (pid :type pid_t :param t))))
  
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> sched.h
				   unistd.h
				   bitset
				   cstring
				   string
				   )
			(include CpuAffinityManagerBase.h
				 DiagramWithGui.h))
     :implementation-preamble
     `(do0
       (include imgui.h)
       (include<>			; sched.h 
	stdexcept
		  
	)
       )
     :code `(do0
	     (defclass ,name "public CpuAffinityManagerBase"
	       "public:"
	       "using CpuAffinityManagerBase::CpuAffinityManagerBase;"
	       (defmethod RenderGui ()
		 (ImGui--Begin (string "CPU Affinity"))
		 (ImGui--Text  (string "Select CPUs for process ID: %d") pid_)
		 (let ((affinityChanged false))
		   (dotimes (i threads_)
		     (let ((label (+ (std--string (string "CPU "))
				     (std--to_string i)))))
		     (let ((isSelected  (aref selected_cpus_ i)))
		       (declare (type bool isSelected)))
		     (when (ImGui--Checkbox (label.c_str)
					    &isSelected)
		       (setf (aref selected_cpus_ i) isSelected)
		       (setf affinityChanged true)))
		   (when affinityChanged
		     (ApplyAffinity))
		   (ImGui--End)
		   ))
	       ))))
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<> 
      format
      iostream
					;unistd.h
					;vector deque chrono
					;cmath
      #+more	popl.hpp
      armadillo
      )
     "using namespace arma;"
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       ,(lprint :msg "start")


       #+more ,(let ((l `(
			  (:name maxThreads :default 12 :short t)
			  (:name maxDataPoints :default 1024 :short n)
			  )))
		 `(let ((op (popl--OptionParser (string "allowed options")))
			,@(loop for e in l collect
					   (destructuring-bind (&key name default short) e
					     `(,name (int ,default))))
			,@(loop for e in `((:long help :short h :type Switch :msg "produce help message")
					   (:long verbose :short v :type Switch :msg "produce verbose output")
					   ,@(loop for f in l
						   collect
						   (destructuring-bind (&key name default short) f
						     `(:long ,name
						       :short ,short
						       :type int :msg "parameter"
						       :default ,default :out ,(format nil "&~a" name))))

					   )
				appending
				(destructuring-bind (&key long short type msg default out) e
				  `((,(format nil "~aOption" long)
				     ,(let ((cmd `(,(format nil "add<~a>"
							    (if (eq type 'Switch)
								"popl::Switch"
								(format nil "popl::Value<~a>" type)))
						   (string ,short)
						   (string ,long)
						   (string ,msg))))
					(when default
					  (setf cmd (append cmd `(,default)))
					  )
					(when out
					  (setf cmd (append cmd `(,out)))
					  )
					`(dot op ,cmd)
					))))
				))
		    (op.parse argc argv)
		    (when (helpOption->count)
		      (<< std--cout
			  op
			  std--endl)
		      (exit 0))

		    ))
       (arma_rng--set_seed_random )
       (let ((A (randn 5 5))
	     (B (mat (pinv A)))
	     (C (mat (inv A)))))
       (for-range (b B )
		  ,(lprint :vars `(b)))
       (for-range (c C)
		  ,(lprint :vars `(c)))
       
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))

