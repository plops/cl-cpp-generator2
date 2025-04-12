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
      ;array
      ;deque
      vector
      memory
      atomic
      ;condition_variable
      ;mutex
      algorithm
					;thread
      cassert
      )
     "using namespace std;"

     "template<typename T> class Arena;"
     ,(let ((name "Ref"))
	`(space "template<typename T>"
		(defclass+ Ref "public enable_shared_from_this<Ref<T>>"
		  #+nil (comments "ctor private because this class must be instantiated as a shared_ptr (use factory function create() instead!)")
		  ;; ctor
		  

		  "public:"
		  (defmethod Ref (r index associatedArena)
		    (declare (type T& r)
			     (type "Arena<T>&" associatedArena)
			     (type int index)
			     (construct ;(enable_shared_from_this<Ref<T>>)
					(arena associatedArena)
					(ref r)
					(idx index)
					)
			     (explicit)
			     (values :constructor))
		    ,(lprint :msg "Ref-ctor"
			     :vars `(&arena &ref idx)))
		  #+nil(space
		   "template<typename... Ts>"
		   (defmethod create (params)
		     (declare (static)
			      (values "shared_ptr<Ref<T>>")
			      (type "Ts&&..." params))
		     (return (make_shared<Ref<T>> "forward<Ts>(params)..."))))
		  (defmethod getIndex ()
		    (declare (values int)
			     (const))
		    (return idx))
		  ;; dtor
		  (defmethod ~Ref ()
		    (declare (values :constructor))
		    ,(lprint :msg "Ref-dtor"
			     :vars `(idx))
		    ;; i think the problem is that the destructor is never called
		    ;; when i need it (with positive use count)
		    #+nil(when (== 3 (dot (this->shared_from_this)
					  (use_count)))
			   (arena.setUnused idx)
			   ))

		 #+nil (do0
		   ;; copy ctor
		   (defmethod Ref (rhs)
		     (declare (type "const Ref&" rhs)
			      (construct
			       (enable_shared_from_this<Ref<T>> rhs)
			       (arena rhs.arena)
			       (ref rhs.ref)
			       (idx rhs.idx))
			      (values :constructor))
		     ,(lprint :msg "Ref-copy-ctor"))
		   ;; copy assign
		   #-nil
		   (defmethod operator= (rhs)
		     (declare (type "const Ref&" rhs)
			      (values "Ref&"))
		     ,(lprint :msg "Ref-copy-assign")
		     (when (== this &rhs)
		       (return *this))
		     (enable_shared_from_this<Ref<T>>--operator= rhs)
		     (setf arena rhs.arena
			   ref rhs.ref
			   idx rhs.idx)
		     (return *this))
		   ;; move ctor
		   (defmethod Ref (rhs)
		     (declare (type "Ref&&" rhs)
			      (noexcept)
			      (construct (enable_shared_from_this<Ref<T>> rhs)
					 (arena rhs.arena)
					 (ref rhs.ref)
					 (idx rhs.idx))
			      (values :constructor))
		     ,(lprint :msg "Ref-move-ctor"))
		   ;; move assign
		   (defmethod operator= (rhs)
		     (declare (type "Ref&&" rhs)
			      (noexcept)
			      (values "Ref&"))
		     ,(lprint :msg "Ref-move-assign")
		     (when (== this &rhs)
		       (return *this))
		     (enable_shared_from_this<Ref<T>>--operator= (move rhs))
		     (setf arena rhs.arena
			   ref rhs.ref
			   idx rhs.idx)
		     (return *this))
		   ;; copy ctor, move ctor ...
		  ) 
		  ,@(loop for e in `(,(format nil "~a(const T&)" name)
				     ,(format nil "~a(T&&)" name)
				     "const T& operator=(const T&)"
				     "T& operator=(T&&)")
			  collect
			  (format nil "~a = delete;" e))
		  
		  "private:"
		  "Arena<T>& arena;"	      
		  "T& ref;"
		  "int idx;"
		  )))
     #+nil
     ,(let ((name "Arena"))
	`(space "template<typename T>"
	       (defclass+ ,name ()
		 "public:"
		 "using SRef = atomic<shared_ptr<Ref<T>>>;"
		#+nil (defmethod aquire ()
		   (declare (values SRef))
		   (let ((it (ranges--find  used
				   false)))
		     (when (== (used.end)
			     it)
			 (throw (runtime_error (string "no free arena element"))))
		     (setf *it true)
		     (let ((idx (- it (used.begin)))
			   #+nil (el
			     (dot r (at idx)))))
		     #+nil ,(lprint :msg "found unused element"
			      :vars `(idx))
		     (return (dot r (at idx)))
		     #+inil (return el)))

		(defmethod setUnused (idx)
		   (declare (type int idx))
		   #+nil ,(lprint :msg "Arena::setUnused"
			    :vars `(idx))
		   (setf (aref used idx) false))
		 (defmethod use_count (idx)
		   (declare (values "long int")
			    (type int idx))
		   (let ((count (dot r (at idx) (use_count)))))
		   #+nil ,(lprint :msg "Arena::use_count"
			    :vars `(count))
		   (return count))
		 
		 (defmethod ,name (n=0)
		   (declare (values :constructor)
			    (type int n=0)
			    (construct (a (vector<T> n))
				       (used (vector<bool> n))
				       (r (vector<SRef>))))
		   "int idx=0;"
		   (for-range (e a)
			      ;(r.push_back ("Ref<T>::create" e idx *this))
			      (r.push_back ("make_shared<Ref<T>>" e idx *this))
			      
			      (incf idx)))

		 ,@(loop for e in `(,(format nil "~a(const T&)" name)
				    ,(format nil "~a(T&&)" name)
				    "const T& operator=(const T&)"
				    "T& operator=(T&&)")
			 collect
			 (format nil "~a = delete;" e))
		 "private:"
		 "vector<T> a;"
		 "vector<bool> used{};"
		 "vector<SRef> r;")))

     ,(let ((name "Arena"))
	`(space "template<typename T>"
	       (defclass+ ,name ()
		 "public:"
		 (defmethod setUnused (idx)
		   (declare (type int idx)))
		 )))
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       (defclass+ Widget ()
	 "public:"
	 "private:"
	 "int i{3};"
	 "float f{4.5F};")
       "const int n=3;"
             
       (do0
	(let ((a (Arena<Widget>))))
	(let ((v (vector<Widget> 3))))
	(let ((e0 (make_shared<Ref<Widget>> (aref v 0) 0 a))))
	(let ((e1 (make_shared<Ref<Widget>> (aref v 1) 1 a))))
	(let ((e2 (make_shared<Ref<Widget>> (aref v 2) 2 a))))
	(setf e1 e0)
	(let ((c0 e0)
	      (c1 (move e1))))
	
	#+nil(let ((a (space Arena (angle Widget) (paren n) ))))

					;(let ((v (vector<Arena<Widget>--SRef>))))
	#+nil
	(do0 (dotimes (i n)
	       (let ((e (a.aquire))))
	       (assert (== i (-> (e.load) (getIndex))))
	       (v.push_back e))
	     ,(lprint :msg "#### CLEAR ####")
	     (v.clear)
	     ,(lprint :msg "#### REUSE N ELEMENTS ####")
	     (dotimes (i n)
	       (let ((e (a.aquire))))
	       (assert (== i (-> (e.load) (getIndex))))
	       (v.push_back e))
	     ,(lprint :msg "#### TRY TO GET ONE ELEMENT TOO MANY ####")
	     (v.push_back (a.aquire))))
                   
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil)
  (sb-ext:run-program "/usr/bin/clang-format"
		      `("-i"
			,@(loop for e in (directory (format nil "~a/*.*" *full-source-dir*))
				collect (format nil "~a" e))
			"-style=file")))

