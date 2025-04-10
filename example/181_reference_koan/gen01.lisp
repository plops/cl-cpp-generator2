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
    (defparameter *source-dir* #P"example/181_reference_koan/src/")
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
      condition_variable
      mutex
      thread
      )
     "using namespace std;"

     (space "template<typename T>"
      (defclass+ Ref ()
	"public:"

	#+nil
	(defmethod get ()
	  (declare (values T&))
	  (return ref))

	;; ctor
	(defmethod Ref (r)
	  (declare (type T& r)
		   (construct (ref r)
			      (sp (createPriv))
			      (q (new Q)))
		   (explicit)
		   (values :constructor))
	  ,(lprint :msg "Ref::ctor"
		   :vars `(&ref (use_count))))
	;; dtor
	(defmethod ~Ref ()
	  (declare (values :constructor))
	  (when q
	    (delete q)
	    (setf q nullptr))
	  ,(lprint :msg "~Ref"
		   :vars `(&ref
			   (use_count))))

	;; copy ctor
	(defmethod Ref (rhs)
	  (declare (type "const Ref&" rhs)
		   (construct (ref rhs.ref)
			      (sp (createPriv)))
		   (values :constructor))
	  ,(lprint :msg "Ref::copy-ctor")
	  #+nil (when (== this &rhs)
	    (return *this))
	  )
	(defmethod use_count ()
	  (declare (values int))
	  (return (dot sp (load ) (use_count))))
	"private:"
	(defclass+ Priv ())
	(defmethod createPriv ()
	  (declare (values "static shared_ptr<Priv>"))
	  (return (shared_ptr<Priv> (new Priv)
		   (lambda (p)
		     (declare (type Priv* p))
		     ,(lprint :msg "~shared_ptr" :vars `(p))
		     (delete p)))))
	(defclass+ Q ()
	  "private:"
	  "mutex m;"
	  "condition_variable c;")
	"T& ref;"
	"atomic<shared_ptr<Priv>> sp{nullptr};"
	"Q* q{nullptr};"))

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
			  (return el))))
		   )
		 
		 (defmethod ,name ()
		   (declare (values :constructor))
		   (for-range (e a)
			      (r.emplace_back e)))



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
       "constexpr int N{3};"
       
       (do0
	(let ((a (space Arena (angle Widget N) (paren))))
	  )
	#+nil(let ((e0 (a.aquire)))))

       (let ((v (deque<Ref<Widget>> ))))
       (dotimes (i (+ N 1))
	 (v.push_back (a.aquire)))
       #+nil
       (do0 (let ((as (space array (angle Widget N)
			     (paren)))))
	    (let ((ar (space deque (angle (space Ref (angle Widget)))
			     (paren))))
	      (for-range (e as)
			 (ar.emplace_back e)))
	    
	    ,(lprint :vars `((sizeof as)))
	    ,(lprint :vars `((sizeof ar)))
	    (let ((e (dot (aref ar 0)
			  (get)))))
	    (let ((qq (aref ar 0))))
	    ,(lprint :vars `((dot (aref ar 0) (use_count)))))
            
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil)
  (sb-ext:run-program "/usr/bin/clang-format"
		      `("-i"
			,@(loop for e in (directory (format nil "~a/*.*" *full-source-dir*))
				collect (format nil "~a" e))
			"-style=file")))

