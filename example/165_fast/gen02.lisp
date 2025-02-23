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
  
  (defparameter *source-dir* #P"example/165_fast/source02/src/")
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
      random
      vector
      
      benchmark/benchmark.h
      )


     (defun GenerateShuffledIndices (n)
       (declare (type uint64_t n)
		(values "std::vector<uint64_t>"))
       "std::mt19937 gen(std::random_device{}());"
       "std::vector<uint64_t> v(n);"
       (std--iota (v.begin)
		  (v.end)
		  0)
       (std--shuffle (v.begin)
		     (v.end)
		     gen)
       (return v))

     (defun Sum (v n)
       (declare (type "std::vector<uint64_t> const&" v)
		(type size_t n)
		(values uint64_t))
       "uint64_t sum{0};"
       (dotimes (pos n)
	 (incf sum (aref v (aref v pos))))
       (return sum))

     (defun BM_Walk (state)
       (declare (type "benchmark::State&" state))

       (let ((kbytes (static_cast<size_t> (state.range 0) ;"64"
					  ))
	     (n (/ (* kbytes 1024)
		   (sizeof uint64_t)))
	     (v (GenerateShuffledIndices n ))))
       (for-range (_ state)
		  (let ((sum ("Sum" v n)))
		    (benchmark--DoNotOptimize sum)))
       
       (state.SetBytesProcessed (* n
				   (sizeof uint64_t)
				   (state.iterations)
				   )))
     
     
     (-> (BENCHMARK BM_Walk)
	 (RangeMultiplier 2)
	 (Range 8 (<< 8 13)))
     (BENCHMARK_MAIN)
     )
   :omit-parens t
   :format t
   :tidy nil))
