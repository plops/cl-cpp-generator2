(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list ; :more
						 ))))

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
    (merge-pathnames "Ref.h"
		     *source-dir*))
   `(do0
     "#pragma once"
     (include<>
      memory
      atomic
      )
     "using namespace std;"

     "template<typename T> class Arena;"
     ,(let ((name "Ref"))
	`(space "template<typename T>"
		(defclass+ Ref ()
		  "public:"
		  ;; ctor
		  (defmethod Ref (r idx associatedArena)
		    (declare (type T& r)
			     (type "Arena<T>&" associatedArena)
			     (type int idx)
			     (construct (arena associatedArena)
					(ref r)
					(sp (make_shared<Priv> idx)))
			     (explicit)
			     (values :constructor)))
		  ;; dtor
		  (defmethod ~Ref ()
		    (declare (values :constructor))
		    (when (== 3 (use_count))
		      (arena.setUnused (idx))))

		  ;; copy ctor
		  (defmethod Ref (rhs)
		    (declare (type "const Ref&" rhs)
			     (construct (arena rhs.arena)
					(ref rhs.ref)
					(sp (rhs.sp.load)))
			     (values :constructor)))
		  ;; move ctor
		  #+nil (defmethod Ref (rhs)
			  (declare (type "Ref&&" rhs)
				   (noexcept)
				   (construct (ref (move rhs.ref))
					      (arena (move rhs.arena))
					      (sp (move (rhs.sp.load))))
				   (values :constructor)))
		  
		  ;; copy ctor, move ctor ...
		  #-nil
		  ,@(loop for e in `( ;,(format nil "~a(const T&)" name)
				     ,(format nil "~a(T&&)" name)
				     "const T& operator=(const T&)"
				     "T& operator=(T&&)")
			  collect
			  (format nil "~a = delete;" e))
		  
		  (defmethod use_count ()
		    (declare (values "long int")
			     (inline))
		    (return (dot sp (load)
				 (use_count))))
		  (defmethod idx ()
		    (declare (values "long int")
			     (inline))
 		    (return "sp.load()->idx"))
		  "private:"
		  (defclass+ Priv ()
		    "public:"
		    "int idx;")
		  "Arena<T>& arena;"	      
		  "T& ref;"
		  "atomic<shared_ptr<Priv>> sp{nullptr};"
		  ))))
   )

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "../tests/test_Ref.cpp"
		     *source-dir*))
   `(do0
     (include<>
      gtest/gtest.h
      vector
      )
     (include "Ref.h")
     "using namespace std;"
     (space "template<typename T>"
	    (defclass+ Arena ()
	      "public:"
	      (defmethod setUnused (idx)
		    (declare (type "long int" idx)))))
     
     (space TEST (paren Ref CopyConstructor_Copy_CountIncreases)
	    (progn
	      (let ((v (vector<int> 3))))
	      (let ((a (Arena<int>))))
	      (let ((r0 (Ref<int>  (aref v 0) 0 a))))
	      (EXPECT_EQ (r0.use_count) 2)
	      (let ((r1 r0)))
	      (EXPECT_EQ (r0.use_count) 3)
	      (EXPECT_EQ (r1.use_count) 3)
	      )))
   )

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "Arena.h"
		     *source-dir*))
   `(do0
     "#pragma once"
     (include<>
      vector
      memory
      atomic
      algorithm
      cassert
      )
     (include Ref.h)
     "using namespace std;"
     ,(let ((name "Arena"))
	`(space "template<typename T>"
		(defclass+ ,name ()
		  "public:"
		  (defmethod aquire ()
		    (declare (values Ref<T>)
			     (inline))
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
				 (el (aref r idx) ;(dot r (at idx))
				     )))
			   ,(lprint :msg "found unused element"
				    :vars `(idx))
			   (return el)))))

		  (defmethod setUnused (idx)
		    (declare (type int idx)
			     (inline))
		    ,(lprint :msg "Arena::setUnused"
			     :vars `(idx))
		    (setf (aref used idx) false))
		  (defmethod use_count (idx)
		    (declare (values "long int")
			     (type int idx)
			     (inline))
		    (let ((count (dot (aref r idx) (use_count)))))
		    ,(lprint :msg "Arena::use_count"
			     :vars `(count))
		    (return count))
		 
		  (defmethod ,name (n=0)
		    (declare (values :constructor)
			     (explicit)
			     (type int n=0)
			     (construct 
			      (used (vector<bool> n))
			      (r (vector<Ref<T>>))
			      (a (vector<T> n))))
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
		  "vector<bool> used{};"
		  "vector<Ref<T>> r;"
		  "vector<T> a;"
		  ))))
   
   )
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
					; iostream
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
     (include Arena.h)
     "using namespace std;"

          

     
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       (defclass+ Widget ()
	 "public:"
	 "private:"
	 "int i{3};"
	 "float f{4.5F};"
	 "char name[20];")
       "const int n=3;"
             
       (do0
	(let ((a (space Arena (angle Widget) (paren n) ))))

	(let ((v (vector<Ref<Widget>> ))))
	,(lprint :vars `((sizeof Widget)))
	,(lprint :vars `((sizeof a)))
	#+nil
	(do0
	 ,(lprint :vars `((sizeof a.used)))
	 ,(lprint :vars `((sizeof a.r)))
	 ,(lprint :vars `((sizeof a.a))))
	,(lprint :vars `((sizeof (aref v 0))))
	
	(dotimes (i n)
	  (let ((e (a.aquire))))
	  (assert (== i (e.idx)))
	  (v.push_back e))
	,(lprint :msg "#### CLEAR ####")
	(v.clear)
	,(lprint :msg "#### REUSE N ELEMENTS ####")
	(dotimes (i n)
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

