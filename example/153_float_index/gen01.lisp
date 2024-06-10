(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ()
  (defparameter *source-dir* #P"example/153_float_index/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (load "util.lisp")
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      iostream
      cstdint
      cstring 
      vector
      )
     (comments "This is a mapping between floating point numbers and integer indices that keep order")

     (doc "

1. **Initialization and Offset:**
   - It takes an unsigned 32-bit integer `n` as input.
   - An offset of `(1U << 23) - 1` (which is 8388607) is added to `n`. This offset is important for aligning the integer representation with the bias used in the IEEE 754 standard for floating-point numbers.

2. **Sign Bit Manipulation:**
   - The code checks the most significant bit (sign bit) of `n`. 
   - If the sign bit is set (`n & (1U << 31) != 0u`):
     - It toggles the sign bit (`n ^= (1U << 31)`). This maps positive integers to negative floats and vice versa.
   - If the sign bit is not set:
     - It takes the bitwise complement of `n` (`n = ~n`). This ensures the correct mapping of negative floats.

3. **Conversion to Float:**
   -  A `float` variable `f` is declared.
   -  The bit pattern of `n` is copied directly into `f` using `memcpy`. This is the key step where the integer bit representation is reinterpreted as a floating-point representation.

4. **Return Value:** The function returns the float `f`.
")
     (defun index_to_float (n)
       (declare (values float)
		(type uint32_t n))
       (incf n (- (<< 1u 23) 1))
       (if (and n (paren (<< 1u 31)))
	   (setf n (^ n (paren (<< 1u 31))))
	   (setf n ~n))
       "float f;"
       (memcpy &f &n 4)
       (return f))

     (defun float_to_index (f)
       (declare (values uint32_t)
		(type float f))
       "uint32_t n;"
       (memcpy &n &f (sizeof n))

       (if (and n (paren (<< 1u 31)))
	   (setf n (~ (^ n (paren (<< 1u 31)))))
	   (setf n ~n))
       (comments "Ensure the subtraction is done as unsigned")
       (return ;n
	       ;(- n (- (<< 1u 23) 1u))
	       (- n (+ (<< 1u 31) (- (<< 1u 23) 1u))
		  ;2155872255
		  )
	       )
       )

     
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (doc "

 v='0'  index_to_float(v)='-inf'  float_to_index(index_to_float(v))='0' 
 v='1'  index_to_float(v)='-3.40282e+38'  float_to_index(index_to_float(v))='1' 
 v='12'  index_to_float(v)='-3.40282e+38'  float_to_index(index_to_float(v))='12' 
 v='1000'  index_to_float(v)='-3.40262e+38'  float_to_index(index_to_float(v))='1000' 
 v='10000'  index_to_float(v)='-3.4008e+38'  float_to_index(index_to_float(v))='10000' 
 v='100000'  index_to_float(v)='-3.38254e+38'  float_to_index(index_to_float(v))='100000' 
 v='1000000000'  index_to_float(v)='-458.422'  float_to_index(index_to_float(v))='1000000000' 
 v='1000000001'  index_to_float(v)='-458.422'  float_to_index(index_to_float(v))='1000000001' 
 v='2000000000'  index_to_float(v)='-6.09141e-34'  float_to_index(index_to_float(v))='2000000000' 
 v='2000000001'  index_to_float(v)='-6.09141e-34'  float_to_index(index_to_float(v))='2000000001' 
 v='3000000000'  index_to_float(v)='4.85143e-08'  float_to_index(index_to_float(v))='1278190081'  // from here the inverse function not working
 v='3000000001'  index_to_float(v)='4.85143e-08'  float_to_index(index_to_float(v))='1278190080' 
 v='4026531838'  index_to_float(v)='3.16913e+29'  float_to_index(index_to_float(v))='251658243' 
 v='4026531839'  index_to_float(v)='3.16913e+29'  float_to_index(index_to_float(v))='251658242' 

")
       (let ((vs (std--vector<uint32_t> (curly 0 1 12 1000 10000 100000 1000000000 1000000001
					       2000000000
					       2000000001
					       3000000000
					       3000000001
					       (- #xefffffff 1)
					       #xefffffff)))))
       (for-range (v vs)
	,(lprint :vars `(v
			 (index_to_float v)
			 (float_to_index (index_to_float v)))))
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))

(defparameter *bla* (+ (ash 1 31) (- (ash 1 23) 1)))
