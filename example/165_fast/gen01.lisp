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
      ;memory
  
      )

          

     (space
      template
      (angle "class T"
	     "size_t ChunkSize")
      (defclass stable_vector ()
	(static_assert (== 0 (% ChunkSize 2))
	 (string "ChunkSize needs to be a multiple of 2"))
	(defmethod operator[] (i)
	  (declare (type size_t i))
	  (return (aref (aref *mChunks (/ i
					  ChunkSize))
			(% i ChunkSize))))
	(space using (setf Chunk boost--container--static_vector (angle T ChunkSize)))
	(space std--vector (angle std--unique_ptr (angle Chunk))
	       mChunks)))
     
     (defun main ()
       (declare (values int)))

     )
   :omit-parens t
   :format t
   :tidy nil))
