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
      iostream
					;format
      ;vector
      ;memory
      ;boost/container/static_vector.hpp
					;boost/container/devector.hpp
					;boost/multi_array.hpp
      unordered_map
      map
      vector
      list
      benchmark/benchmark.h
      stable_vector.h

      
      )
     (include papipp.h)
     (space
      "template<typename T>"
      (defun Sum (v)
	(declare (type T v)
		 (values int))
	"int sum{0};"
	(dotimes (i (v.size))
	  (incf sum (aref v i)))
	(return sum)))

     ,(let ((l-events #-nil
		      `(PAPI_TOT_INS
			PAPI_TOT_CYC
			PAPI_BR_MSP
			PAPI_L1_DCM)
		      #+nil
		      `(PAPI_L1_DCM
			PAPI_L2_DCM
			PAPI_L2_ICM
			PAPI_TLB_DM
			PAPI_TLB_IM
			PAPI_BR_UCN
			PAPI_BR_CN
			PAPI_BR_TKN
			PAPI_BR_NTK
			PAPI_BR_MSP
			PAPI_BR_PRC
			PAPI_TOT_INS
			PAPI_FP_INS
			PAPI_BR_INS
			PAPI_VEC_INS
			PAPI_TOT_CYC
			PAPI_L2_DCH
			PAPI_L1_DCA
			PAPI_L2_DCR
			PAPI_L2_ICH
			PAPI_L2_ICA
			PAPI_L2_ICR
			PAPI_FML_INS
			PAPI_FAD_INS
			PAPI_FDV_INS
			PAPI_FSQ_INS
			PAPI_FP_OPS)))
	(flet ((papi (code)
		 `(do0
		   ,(format nil "papi::event_set<~{~a~^, ~}> events;" l-events)
		   (events.start_counters)
		   ,code
		   (events.stop_counters)
		   (do0
		    (<< std--cout
			events
			(string "\\n"))
		    ,@(loop for e in l-events
			    collect
			    (let ((en (string-downcase (format nil "~a" e))))
			      `(do0
				(let ((,en
					(dot events
					     (,(format nil "get<~a>" e))
					     (counter))))
				  ,(lprint :vars `(,en))))))
		    (let ((insPerCyc (/ (static_cast<float> papi_tot_ins) papi_tot_cyc)))
		      ,(lprint :vars `(insPerCyc))))
		   )))
	  
	  `(do0
	    (defun BM_StableVector (state)
	      (declare (type "benchmark::State&" state))
	      "stable_vector<int, 4*4096> v;"
	      "std::list<int> tmp;"
	      
	      ,(papi
		`(do0
		  (dotimes (i "100'000")
		    (comments "randomize heap by filling list (this makes the micro-benchmark more like the real thing)")
		    (dotimes (x 1000)
		      (tmp.push_back x))
		    (v.push_back i)
		    )
		  
		  (for-range (_ state)
			     (let ((sum ("Sum<stable_vector<int,4*4096>>" v)))
			       (benchmark--DoNotOptimize sum)))))
	      

	      
	      
	      
	      )
	    (defun BM_StableVectorReserved (state)
	      (declare (type "benchmark::State&" state))
	      "stable_vector<int, 4*4096> v;"
	      (v.reserve "100'000")
	      "std::list<int> tmp;"

	      ,(papi
		`(do0
		 (dotimes (i "100'000")
		   (comments "randomize heap by filling list (this makes the micro-benchmark more like the real thing)")
		   (dotimes (x 1000)
		     (tmp.push_back x))
		   (v.push_back i)
		   )
	       
		 (for-range (_ state)
			    (let ((sum ("Sum<stable_vector<int,4*4096>>" v)))
			      (benchmark--DoNotOptimize sum))))))

	    (defun BM_Vector (state)
	      (declare (type "benchmark::State&" state))
	      "std::vector<int> v;"
	      "std::list<int> tmp;"

	      ,(papi `(do0
		 (dotimes (i "100'000")
		   (comments "randomize heap by filling list (this makes the micro-benchmark more like the real thing)")
		   (dotimes (x 1000)
		     (tmp.push_back x))
		   (v.push_back i)
		   )
	       
		 (for-range (_ state)
			    (let ((sum ("Sum<std::vector<int>>" v)))
			      (benchmark--DoNotOptimize sum))))))


	    (defun BM_VectorReserved (state)
	      (declare (type "benchmark::State&" state))
	      "std::vector<int> v;"
	      (v.reserve "100'000")
	      "std::list<int> tmp;"
	      ,(papi `(do0
		 (dotimes (i "100'000")
		   (comments "randomize heap by filling list (this makes the micro-benchmark more like the real thing)")
		   (dotimes (x 1000)
		     (tmp.push_back x))
		   (v.push_back i)
		   )
	       
		 (for-range (_ state)
			    (let ((sum ("Sum<std::vector<int>>" v)))
			      (benchmark--DoNotOptimize sum))))))


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

	      ,(papi `(do0
		  (dotimes (i "100'000")
		    (comments "randomize heap by filling list (this makes the micro-benchmark more like the real thing)")
		    (dotimes (x 1000)
		      (tmp.push_back x))
		    (v.emplace i i)
		    )
	       
		  (for-range (_ state)
			     (let ((sum ("Sum<std::unordered_map<int,int>>" v)))
			       (benchmark--DoNotOptimize sum))))))

	    (defun BM_Map (state)
	      (declare (type "benchmark::State&" state))
	      "std::map<int, int> v;"
	      
	      "std::list<int> tmp;"
	      ,(papi `(do0 (dotimes (i "100'000")
			(comments "randomize heap by filling list (this makes the micro-benchmark more like the real thing)")
			(dotimes (x 1000)
			  (tmp.push_back x))
			(v.emplace i i)
			    )
			  
			  (for-range (_ state)
				     (let ((sum ("Sum<std::map<int,int>>" v)))
				       (benchmark--DoNotOptimize sum))))))

	    
	    )))

     

     

     

     

     

     
     
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
     (BENCHMARK BM_StableVectorReserved)
     (BENCHMARK BM_Vector)
     (BENCHMARK BM_VectorReserved)
     (BENCHMARK BM_UnorderedMap)
     (BENCHMARK BM_UnorderedMapReserved)
     (BENCHMARK BM_Map)
     (BENCHMARK_MAIN)
     )
   :omit-parens t
   :format t
   :tidy nil))
