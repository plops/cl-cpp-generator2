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
      ;vector
      ;memory
      ;boost/container/static_vector.hpp
      ;boost/container/devector.hpp
      ;boost/multi_array.hpp
      ;unordered_map
      list
      benchmark/benchmark.h
      stable_vector.h)

     #+nil
     (do0
      (comments "simple container that keeps things together")
      (space
       template
       (angle "class T"
	      "size_t ChunkSize")
       (defclass+ stable_vector ()
	 (static_assert (== 0 (% ChunkSize 2))
			(string "ChunkSize needs to be a multiple of 2"))
	 "public:"
	 (defmethod operator[] (index)
	   (declare (type size_t index)
		    (values T&))
	   #+nil (let ((frob (lambda (i)
			       (return (aref (paren (aref *mChunks (/ i
								      ChunkSize)))
					     (% i ChunkSize)))))
		       )
		   (return (frob index)))
	   (return (aref (paren (deref (aref mChunks (/ index
							ChunkSize))))
			 (% index ChunkSize))))
	 (defmethod push_back (value)
	   (declare (type "T" value))
	   (incf mN)
	   (deref (paren (dot mChunks
			      (push_back value))))
	   )
	 (defmethod size ()
	   (declare (values size_t))
	   (return mN))
	 
	 "private:"
	 
	 (comments "similar to std::deque but that doesn't have a configurable chunk size, which is usually chosen too small by the compiler")
	 (space using (setf Chunk "boost::container::static_vector<T,ChunkSize>"))
	 "std::vector<std::unique_ptr<Chunk>> mChunks;"

	 "size_t mN;"
	 
	 )))

     (space
      "template<typename T>"
      (defun Sum (v)
	(declare (type T ;"stable_vector<int,4*4096>"
		       v
		       )
		 (values int))
	"int sum{0};"
	(dotimes (i (v.size))
	  (incf sum (aref v i)))
	(return sum)))
     
     (defun BM_StableVector (state)
       (declare (type "benchmark::State&" state))
       "stable_vector<int, 4*4096> v;"
       "std::list<int> tmp;"
       (dotimes (i "100'000")
	 (comments "randomize heap by filling list (this makes the micro-benchmark more like the real thing)")
	 (dotimes (x 1000)
	   (tmp.push_back x))
	 (v.push_back i)
	 )
       
       (for-range (_ state)
		  (let ((sum ("Sum<stable_vector<int,4*4096>>" v)))
		    (benchmark--DoNotOptimize sum))))

     (defun BM_UnorderedMap (state)
       (declare (type "benchmark::State&" state))
       "std::unordered_map<int, int> v;"
       "std::list<int> tmp;"
       (dotimes (i "100'000")
	 (comments "randomize heap by filling list (this makes the micro-benchmark more like the real thing)")
	 (dotimes (x 1000)
	   (tmp.push_back x))
	 (v.emplace i i)
	 )
       
       (for-range (_ state)
		  (let ((sum ("Sum<std::unordered_map<int,int>>" v)))
		    (benchmark--DoNotOptimize sum))))

     (defun BM_UnorderedMapReserved (state)
       (declare (type "benchmark::State&" state))
       "std::unordered_map<int, int> v;"
       (v.reserve "200'000")
       "std::list<int> tmp;"
       (dotimes (i "100'000")
	 (comments "randomize heap by filling list (this makes the micro-benchmark more like the real thing)")
	 (dotimes (x 1000)
	   (tmp.push_back x))
	 (v.emplace i i)
	 )
       
       (for-range (_ state)
		  (let ((sum ("Sum<std::unordered_map<int,int>>" v)))
		    (benchmark--DoNotOptimize sum))))
     
     #+nil (defun main ()
	     (declare (values int))

	     #+nil
	     (do0
	      "boost::multi_array<float,3> a;"
	      (dotimes (i 10)
		(setf (aref a i i i) i)))
					;
					;"boost::container::devector d;"
					; "stable_vector<float,1024> mFloats;"
					;"std::unordered_map<int,float*> mInstruments;"

       

	     (comments "Working set size (WSS) is the memory you work with, not how much memory you allocated or mapped. Measured in cache lines or pages (Brendan Gregg WSS estimation tool wss.pl)")
	     (return 0))
     (BENCHMARK BM_StableVector)
     (BENCHMARK BM_UnorderedMap)
     (BENCHMARK BM_UnorderedMapReserved)
     (BENCHMARK_MAIN)
     )
   :omit-parens t
   :format t
   :tidy nil))
