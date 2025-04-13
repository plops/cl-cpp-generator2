(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list  :more
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
      mutex
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
					(sp (make_shared<Priv> idx))
					)
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
		  ;; copy assign
		  (defmethod operator= (rhs)
		    (declare (type "const Ref&" rhs)
			     (values Ref&))
		    (unless (== this &rhs)
		      (setf arena rhs.arena
			    ref rhs.ref
			    sp (rhs.sp.load)))
		    (return *this))
		  ;; move ctor
		  #-nil (defmethod Ref (rhs)
			  (declare (type "Ref&&" rhs)
				   (noexcept)
				   (construct
				    (arena rhs.arena)
				    (ref rhs.ref)
				    (sp (rhs.sp.load)))
				   (values :constructor)))
		  ;; move assign
		  #-nil (defmethod operator= (rhs)
			  (declare (type "Ref&&" rhs)
				   (noexcept)
				   (values Ref&))
			  (unless (== this &rhs)
			    (setf arena rhs.arena
				  ref rhs.ref
				  sp (rhs.sp.load)))
			  (return *this))
		  ;; copy ctor, move ctor ...
		  
		  #+nil ,@(loop for e in `( ;,(format nil "~a(const ~a&)" name name)
					;,(format nil "~a(~a&&)" name name)
					;,(format nil "~a& operator=(const ~a&)" name name)
					   ,(format nil "~a& operator=(~a&&)" name name))
				collect
				(format nil "~a = delete;" e))
		  #+nil (defmethod getMutex ()
			  (declare (values mutex&))
			  (return m))
		  (defmethod use_count ()
		    (declare (values "long int")
			     (inline))
		    ;(let ((l (lock_guard arena.m))))
		    (return (dot sp (load)
				 (use_count))))
		  
		  "private:"
		  
		  (defmethod idx ()
		    (declare (values "long int")
			     (inline))
		    ;(let ((l (lock_guard arena.m))))
 		    (return "sp.load()->idx"))
		  (defclass+ Priv ()
		    "public:"
		    "int idx;")
		  "Arena<T>& arena;"	      
		  "T& ref;"
		  "atomic<shared_ptr<Priv>> sp{nullptr};"
		  ))))
   :omit-parens t
   :format nil
   :tidy nil
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
		(declare (type "long int" idx)))
	      "mutex m;"))
     
     (space TEST (paren Ref CopyConstructor_Copy_CountIncreases)
	    (progn
	      (let ((v (vector<int> 3))))
	      (let ((a (Arena<int>))))
	      
	      (let ((r0 (Ref<int>  (aref v 0) 0 a ))))
	      (EXPECT_EQ (r0.use_count) 2)
	      (let ((r1 r0)))
	      (EXPECT_EQ (r0.use_count) 3)
	      (EXPECT_EQ (r1.use_count) 3)
	      ))
     #+nil
     (space TEST (paren Ref CopyAssign_Assign_CountIncreases)
	    (progn
	      (let ((v (vector<int> 3))))
	      (let ((a (Arena<int>))))
	      (let ((r0 (Ref<int>  (aref v 0) 0 a))))
	      (let ((r1 (Ref<int>  (aref v 1) 1 a))))
	      (EXPECT_EQ (r0.use_count) 2)
	      (EXPECT_EQ (r1.use_count) 2)
	      (setf r1 r0)
	      (EXPECT_EQ (r0.use_count) 3)
	      (EXPECT_EQ (r1.use_count) 3)
	      ))
     #+nil
     (space TEST (paren Ref MoveConstructor_Move_CountUnmodified)
	    (progn
	      (let ((v (vector<int> 3))))
	      (let ((a (Arena<int>))))
	      (let ((r0 (Ref<int>  (aref v 0) 0 a))))
	      (EXPECT_EQ (r0.use_count) 2)
	      (let ((r1 (move r0))))
					;(EXPECT_EQ (r1.get) nullptr)
	      (EXPECT_EQ (r1.use_count) 3)
	      (comments "not sure why this is 3, strange")
	      ))
     #+nil
     (space TEST (paren Ref MoveAssign_Assign_CountUnmodified)
	    (progn
	      (let ((v (vector<int> 3))))
	      (let ((a (Arena<int>))))
	      (let ((r0 (Ref<int>  (aref v 0) 0 a))))
	      (let ((r1 (Ref<int>  (aref v 1) 1 a))))
	      (EXPECT_EQ (r0.use_count) 2)
	      (EXPECT_EQ (r1.use_count) 2)
	      (setf r1 r0)
	      (EXPECT_EQ (r0.use_count) 3)
	      (EXPECT_EQ (r1.use_count) 3)
	      (comments "i think the move operators actually perform a copy3")
	      ))

     )
   :omit-parens t
   :format nil
   :tidy nil
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
      mutex
      #+more iostream
      algorithm
      )
     (include Ref.h)
     "using namespace std;"
     ,(let ((name "Arena"))
	`(space "template<typename T>"
		(defclass+ ,name ()
		  "public:"
		  (defmethod firstUnused ()
		    (declare (values int))
		    ;(let ((l (lock_guard m))))
		    (let ((it (find (used.begin)
				    (used.end)
				    false)))
		      (when (== (used.end)
				it)
			(return -1))
		      (return (- it (used.begin))))
		    )
		  (defmethod acquire ()
		    (declare (values Ref<T>)
			     (inline))
		    (elementNowUnused.clear)
		    (let ((idx (firstUnused)))
		      (when (== -1 idx)
			(do0
			 #+nil
			 (throw (runtime_error (string "no free arena element")))
			 (comments "pikus p.549")
			 ,(lprint :msg "waiting for element to become unused")
			 (elementNowUnused.wait false memory_order_acquire)
			 (comments "according to standard this wait should not spuriously wake up. the book still adds this check because tsan thinks otherwise")
			 (do0 ;while (elementNowUnused.test memory_order_acquire)
			  (comments "new elements should now be present")
			  (let ((idx (firstUnused)))	
			    (when  (== -1 idx)
			      (throw (runtime_error (string "no free arena element"))))
			    (do0
			     (let ((el (dot r (at idx)))))
			     ,(lprint :msg "found unused element after wait"
				      :vars `(idx))
			     (progn (let ((l (lock_guard m))))
				    (setf (aref used idx) true))
			     (return el)))))
			
			)
		      (do0
		       (let (
			     (el (dot r (at idx)))))
		       ,(lprint :msg "found unused element"
				:vars `(idx))
		       (progn (let ((l (lock_guard m))))
			      (setf (aref used idx) true))
		       (return el))))

		  (defmethod setUnused (idx)
		    (declare (type int idx)
			     (inline))
		    
		    (progn
		      (let ((l (lock_guard m))))
		      ,(lprint :msg "Arena::setUnused"
			       :vars `(idx))
		      (setf (aref used idx) false))
		    (elementNowUnused.test_and_set memory_order_release)
		    (elementNowUnused.notify_one))
		  (defmethod capacity ()
		    (declare (values int))
		    (return (dot r (size))))

		  (defmethod nb_unused ()
		    (declare (values int))
		    (return (- (capacity)
			       (nb_used))))

		  (defmethod nb_used ()
		    (declare (values int))
		    (let ((sum 0))
		      (for-range (b used)
				 (when b
				   (incf sum))))
		    (return sum))
		  (defmethod use_count (idx)
		    (declare (values "long int")
			     (type int idx)
			     (inline))
		    (let ((count (dot (aref r idx) (use_count)))))
		    ,(lprint :msg "Arena::use_count"
			     :vars `(count))
		    (return count))
		  
		  (defmethod ,name (n=1)
		    (declare (values :constructor)
			     (explicit)
			     (type int n=1)
			     (construct 
					;(used (vector<bool> n))
			      (r (vector<Ref<T>>))
			      (a (vector<T> n))))
		    
		    (progn
		      (let ((l (lock_guard m))))
		      (used.reserve n)
		      (used.resize n)
		      (ranges--fill used false))
		    
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
		  "mutex m; // protect access to used[] and idx in Ref<T>"
		  "private:"
		  
		  "vector<bool> used{};"
		  "vector<Ref<T>> r;"
		  "vector<T> a;"
		  "atomic_flag elementNowUnused{false};"
		  
		  

		  ))))
   :omit-parens t
   :format nil
   :tidy nil)

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "../tests/test_Arena.cpp"
		     *source-dir*))
   `(do0
     (include<>
      gtest/gtest.h
      vector
      thread
      latch
      chrono
      )
     (include "Arena.h")
     "using namespace std;"
     "using namespace std::chrono_literals;"
     
     (space TEST (paren Arena acquire_perform_freeElementsShrink)
	    (progn
	      (space struct Widget (progn "int i;" "float a;"))
	      (let ((n 3)
		    (a (Arena<Widget> n))
		    (v (vector<Ref<Widget>>)))
		
		(dotimes (i n)
		  (v.push_back (a.acquire))
		  (EXPECT_EQ (a.capacity) n)
		  (EXPECT_EQ (a.nb_used) (+ 1 i)))
		#+nil (EXPECT_THROW (a.acquire)
				    runtime_error))))

     (space TEST (paren Arena acquire_performUntilWait_elementArrivesAfterWait)
	    (progn
	      (space struct Widget (progn "int i;" "float a;"))
	      (let ((n 3)
		    (a (Arena<Widget> n))
		    )
		(let ((la (latch 1))
		      (th
			(jthread
			 (lambda ( )
			   (declare (capture "&n" "&a" "&la"))
			   (let ((v (vector<Ref<Widget>>))))
			   (dotimes (i n)
			     (v.push_back (a.acquire))
			     (EXPECT_EQ (a.capacity) n)
			     (EXPECT_EQ (a.nb_used) (+ 1 i))
			     )
			   (la.count_down)
			   (this_thread--sleep_for 10ms)
			   ,(lprint :msg "exiting thread that held elements"))))))
		
		(la.wait) (comments "wait until the thread used all the elements")
		(let ((start (chrono--system_clock--now))))
		(a.acquire)
		(let ((end (chrono--system_clock--now))
		      (duration_ms (* (dot (paren (- end start)) (count)) 1e-6)))
		  )
		,(lprint :vars `(duration_ms))
		(EXPECT_GE duration_ms 10)
		(EXPECT_LE duration_ms 12)
		)))
     )
   :omit-parens t
   :format nil
   :tidy nil
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
       #+nil
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
	  (let ((e (a.acquire))))
	  (assert (== i (e.idx)))
	  (v.push_back e))
	,(lprint :msg "#### CLEAR ####")
	(v.clear)
	,(lprint :msg "#### REUSE N ELEMENTS ####")
	(dotimes (i n)
	  (let ((e (a.acquire))))
	  (assert (== i (e.idx)))
	  (v.push_back e))
	,(lprint :msg "#### TRY TO GET ONE ELEMENT TOO MANY ####")
	(v.push_back (a.acquire)))
       
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil)
  (sb-ext:run-program "/usr/bin/clang-format"
		      `("-i"
			,@(loop for e in
				      (append (directory (format nil "~a/*.*" *full-source-dir*))
					      (directory (format nil "~a/../tests/*.*" *full-source-dir*)))
				collect (format nil "~a" e))
			"-style=file")))

