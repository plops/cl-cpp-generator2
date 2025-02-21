(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more ;; command line parsing
						    
						    )))
  (setf *features* (set-exclusive-or *features* (list :more
						      ;:invert
						      ))))

(let ()
  
  (defparameter *source-dir* #P"example/165_fast/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (defun begin (arr)
    `(ref (aref ,arr 0)) )
  (defun end (arr)
    `(+ (ref (aref ,arr 0)) (dot ,arr (size))))
  
  (defun lprint (&key (msg "")
		   (vars nil)
		   )
    `(<< std--cout
	 (std--format
	  (string ,(format nil "(~a~{:~a '{}'~^ ~})\\n"
			   msg
			   (loop for e in vars collect (emit-c :code e  :omit-redundant-parentheses t)) ))
	  ,@vars)))
    
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
					;iostream
					;format
      vector
      memory
      boost/container/static_vector.hpp
      unordered_map
      list
      )

     (comments "simple container that keeps things together")
     (space
      template
      (angle "class T"
	     "size_t ChunkSize")
      (defclass+ stable_vector ()
	(static_assert (== 0 (% ChunkSize 2))
		       (string "ChunkSize needs to be a multiple of 2"))
	"public:"
	(defmethod operator[] (i)
	  (declare (type size_t i))
	  (return (aref (aref *mChunks (/ i
					  ChunkSize))
			(% i ChunkSize))))
	(defmethod push_back (value)
	  (declare (type "T&&" value))
	  (setf (aref *mChunks 0) value)
	  )
	"private:"
	(comments "similar to std::deque but that doesn't have a configurable chunk size, which is usually chosen too small by the compiler")
	(space using (setf Chunk "boost::container::static_vector<T,ChunkSize>"))
	(space "std::vector<std::unique_ptr<Chunk>>"
	       mChunks)))

     (defun BM_StableVector ()
       "stable_vector<int, 4*4096> v;"
       "std::list<int> tmp;"
       (dotimes (i "100'000")
	 (comments "randomize heap")
	 (dotimes (x 1000)
	   (tmp.push_back x))
	 (v.push_back i))
       #+nil 
       (for-range (_ state)))
     
     (defun main ()
       (declare (values int))


       "stable_vector<float,1024> mFloats;"
       "std::unordered_map<int,float*> mInstruments;"

       (BM_StableVector)

       (comments "Working set size (WSS) is the memory you work with, not how much memory you allocated or mapped. Measured in cache lines or pages (Brendan Gregg WSS estimation tool)"))

     )
   :omit-parens t
   :format t
   :tidy nil))
