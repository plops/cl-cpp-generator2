(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(progn
  (progn
    (defparameter *source-dir* #P"example/182_shared_from_this_koan/src/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "../163_value_based_oop/util.lisp")
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
      iostream
      array
      deque
      memory
      atomic
      ;condition_variable
      ;mutex
      algorithm
					;thread
      cassert
      )
     "using namespace std;"

      "constexpr int N{3};"

     "template<typename T, int N> class Arena;"
     
     (space "template<typename T>"
	    (defclass+ Ref ()
	      "public:"
	      ;; ctor
	      (defmethod Ref (r idx arena)
		(declare (type T& r)
			 (type "Arena<T,N>&" arena)
			 (type int idx)
			 (construct (ref r)
				    (sp (createPriv idx arena)))
			 (explicit)
			 (values :constructor))
		,(lprint :msg "Ref::ctor"
			 :vars `(idx (dot sp (load) (get)) &ref &arena )))
	      ;; dtor
	      (defmethod ~Ref ()
		(declare (values :constructor))
		,(lprint :msg "~Ref" :vars `((use_count)))
		(when (== 3 (use_count))
		  
		  #-nil ,(lprint :msg "#### # tell arena" :vars `((idx) (use_count)))
		  ("sp.load()->arena.setUnused" (idx))
		  )
		)
	      ;; copy ctor
	      (defmethod Ref (rhs)
		(declare (type "const Ref&" rhs)
			 (construct (ref rhs.ref)
				    (sp (rhs.sp.load) #+nil  (createPriv (-> (dot rhs sp (load))
									idx)
								    (-> (dot rhs sp (load))
									arena))))
			 (values :constructor))
		,(lprint :msg "Ref::copy-ctor"
			 :vars `("sp.load()->idx")))
	      (defmethod use_count ()
		(declare (values "long int"))
		(return (dot sp (load)
			     (use_count))))
	      (defmethod idx ()
		(declare (values "long int"))
		(return (-> (dot sp (load)
				 )
			    idx)))
	      "private:"
	      (defclass+ Priv ()
		"public:"
		"int idx;"
		"Arena<T,N>& arena;")
	      (defmethod createPriv (idx arena)
		(declare (type int idx)
			 (type "Arena<T,N>&" arena)
			 (values shared_ptr<Priv>))
		(return (shared_ptr<Priv> (new (Priv idx arena))
					  (lambda (p)
					    (declare (type Priv* p))
					    ,(lprint :msg "~shared_ptr" :vars `(p p->idx ))
					    ;("p->arena.setUnused" p->idx)
					    (delete p)))))
	      
	      "T& ref;"
	      "atomic<shared_ptr<Priv>> sp{nullptr};"
	      ))

     ,(let ((name "Arena"))
	`(space "template<typename T, int N>"
	       (defclass+ ,name ()
		 "public:"
		 (defmethod aquire ()
		   (declare (values Ref<T>))
		   (let ((it (find (used.begin)
				   (used.end)
				   false)))
		     (if (== (used.end)
			     it)
			 (do0
			  (throw (runtime_error (string "no free arena element"))))

			 (do0
			  (setf *it true)
			  (let ((idx (- it (used.begin)))
				(el (dot r (at idx)))))
			  ,(lprint :msg "found unused element"
				   :vars `(idx))
			  (return el)))))

		 (defmethod setUnused (idx)
		   (declare (type int idx))
		   ,(lprint :msg "Arena::setUnused"
			    :vars `(idx))
		   (setf (aref used idx) false))
		 (defmethod use_count (idx)
		   (declare (values "long int")
			    (type int idx))
		   (let ((count (dot (aref r idx) (use_count)))))
		   ,(lprint :msg "Arena::use_count"
			    :vars `(count))
		   (return count))
		 
		 (defmethod ,name ()
		   (declare (values :constructor))
		   "int idx=0;"
		   (for-range (e a)
			      (r.emplace_back e idx *this)
			      (incf idx)))

		 ,@(loop for e in `(,(format nil "~a(const T&)" name)
				    ,(format nil "~a(T&&)" name)
				    "const T& operator=(const T&)"
				    "T& operator=(T&&)")
			 collect
			 (format nil "~a = delete;" e))
		 "private:"
		 "array<T,N> a;"
		 "array<bool,N> used{};"
		 "deque<Ref<T>> r;")))
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       (defclass+ Widget ()
	 "public:"
	 "private:"
	 "int i{3};"
	 "float f{4.5F};")
             
       (do0
	(let ((a (space Arena (angle Widget N) (paren)))))

	(let ((v (deque<Ref<Widget>> ))))
	(dotimes (i N)
	  (let ((e (a.aquire))))
	  (assert (== i (e.idx)))
	  (v.push_back e))
	,(lprint :msg "#### CLEAR ####")
	(v.clear)
	,(lprint :msg "#### REUSE N ELEMENTS ####")
	(dotimes (i N)
	  (let ((e (a.aquire))))
	  (assert (== i (e.idx)))
	  (v.push_back e)))
       ,(lprint :msg "#### TRY TO GET ONE ELEMENT TOO MANY ####")
       (v.push_back (a.aquire))
                   
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil)
  (sb-ext:run-program "/usr/bin/clang-format"
		      `("-i"
			,@(loop for e in (directory (format nil "~a/*.*" *full-source-dir*))
				collect (format nil "~a" e))
			"-style=file")))

