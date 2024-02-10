(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "alexandria"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ((module-name "olcUTIL_Geometry2D_py"))
  (defparameter *source-dir* #P"example/144_geom2d/source01/python/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
 
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames (format nil "~a.cpp" module-name)
		     *source-dir*))
   
   `(do0
     (comments "get header with: wget https://raw.githubusercontent.com/OneLoneCoder/olcUTIL_Geometry2D/main/olcUTIL_Geometry2D.h")
     (comments ,(format nil "compile with: c++ -O3 -Wall -shared -std=c++20 -fPIC $(python3 -m pybind11 --includes) ~a.cpp -o ~a$(python3-config --extension-suffix)" module-name module-name))
     (include olcUTIL_Geometry2D.h)
     (include<> 
					;format
					;unistd.h
					;vector deque chrono
					;cmath
      pybind11/pybind11.h
      ;pybind11/stl.h
      pybind11/functional.h
      )
     (space namespace (setf py pybind11))
     (space using namespace olc--utils--geom2d)
     (space using namespace olc)

     (space
      (PYBIND11_MODULE ,module-name m)
      (progn
	(comments "Expose the v_2d<float> class to Python as \"v\"")
	(dot (py--class_<v_2d<float>> m (string "v"))
	     (def ("py::init<float,float>"))
	     ,@(loop for e in `(x y)
		     collect
		     `(def_readwrite
			  (string ,(format nil "~a" e))
			  ,(format nil "&v_2d<float>::~a" e)))
	     (def (string "__repr__")
		 (lambda (v)
		   (declare (capture "")
			    (type "const v_2d<float> &" v))
		   (return (+ (string "<v x=")
			      (std--to_string v.x)
			      (string ", y=")
			      (std--to_string v.y)
			      (string ">")))))

	     ,@(loop for e in `(area mag mag2 norm perp floor ceil max min dot
				     cross cart polar clamp lerp reflect)
		     collect
		     `(def
			  (string ,(format nil "~a" e))
			  ,(format nil "&v_2d<float>::~a" e)))

	     #+nil
	     ,@(loop for e in `(== "!=" + - * / += -=)
		     collect
		     `(def (space py--self
				  ,(format nil "~a" e) 
				  py--self)))
	     #+nil ,@(loop for e in `(*= /=)
			   collect
			   `(def (space py--self
					,(format nil "~a" e)
					(float))))

	     ,@(loop for e in `(__str__ ;__repr__
				)
		     collect
		     `(def 
			  (string ,(format nil "~a" e))
			"&v_2d<float>::str"))
	     
	     
	     )
	#+nil
	,@(loop for e in `(* +) and name in `(__mul__ __add__)
		appending
		
		`((dot m (def  (string ,name)
			     (lambda (lhs rhs)
			       (declare (capture "")
					(type float lhs)
					(type "v_2d<float>&" rhs))
			       (return (,e lhs rhs))
			       )))
		  (dot m (def (string ,name)
			     (lambda (lhs rhs)
			       (declare (capture "")
					(type float rhs)
					(type "v_2d<float>&" lhs))
			       (return (,e lhs rhs))
			       )))))

	,@(loop for e in `(pi epsilon)
		collect
		`(setf (m.attr (string ,e))
		       ,(format nil "utils::geom2d::~a" e)))

	,@(loop for e in `((:name line 
			    :constructor-args "const v_2d<float>&, const v_2d<float>&"
			    :elements (start end)
			    :elements-to-string (nil nil)
			    :functions (length vector))
			   (:name rect
			    :constructor-args "const v_2d<float>&, const v_2d<float>&"
			    :elements (pos size)
			    :elements-to-string (nil nil)
			    :functions (area))
			   (:name triangle
			    :constructor-args "const v_2d<float>&, const v_2d<float>&, const v_2d<float>&"
			    ;:elements ("pos[0]" "pos[1]" "pos[2]")
			    ;:elements-to-string (nil nil nil)
			    :functions (area))
			   (:name circle 
			    :constructor-args "const v_2d<float>&, float"
			    :elements (pos radius)
			    :elements-to-string (nil t)
			    :functions (area))
			   )
		collect
		(destructuring-bind (&key name (py-name name) constructor-args elements elements-to-string functions) e
		  `(do0
		    (comments ,(format nil "Expose the ~a<float> class to Python as \"~a\""
				       name py-name))
		    (dot (,(format nil "py::class_<~a<float>>" name) m (string ,py-name))
			 (def (,(format nil "py::init<~a>" constructor-args)))
			 ,@(loop for el in elements
				 collect
				 `(def_readwrite
				      (string ,(format nil "~a" el))
				      ,(format nil "&~a<float>::~a" name el)))
			 (def (string "__repr__")
			     (lambda (arg)
			       (declare (capture "")
					(type ,(format nil "const ~a<float>&" name) arg))
			       (return (+ (std--string (string ,(format nil "<~a" py-name)))
					  ,@(loop for el in elements and
						  conversion in elements-to-string
						  appending
						  `((string ,(format nil " ~a=" el))
						    ,(if conversion
							 `(std--to_string (dot arg ,el))
							 `(dot arg ,el (str))
							 )
						    ))
					  (string ">")))))
			 ,@(loop for fun in functions
				 collect
				 `(def (string ,fun)
				    ,(format nil "&~a<float>::~a" name fun)
				    ))
			 ))))
	,@(loop for (a b) in 
		(let ((res))
		  (alexandria:map-permutations #'(lambda (x)
						   (push x res))
					       `(v_2d circle line rect triangle)
					       :length 2)
		  res)
		collect
		`(do0
		  (comments ,(format nil "contains(~a,~a)" a b))
		  (m.def (string "contains")
			 ,(format nil "(bool (*) (const ~a<float>&, const ~a<float>&)) &contains"
				  a b))))
	#+nil
	(dot (py--class_<utils--geom2d--line<float>> m (string "line"))
	     (def ("py::init<v_2d<float>,v_2d<float>>"))
	    

	     ,@(loop for e in `(vector length length2 rpoint upoint side coefficients)
		     collect
		     `(def
			  (string ,(format nil "~a" e))
			  ,(format nil "&utils::geom2d::line<float>::~a" e))))
	

	
	#+nil
	(dot (py--class_<circle<float>> m (string "circle"))
	     (def ("py::init<float,float>"))
	     (def_readwrite (string "x") "&v_2d<float>::x")
	     (def_readwrite (string "y") "&v_2d<float>::y")
	     )))
     )
   :omit-parens t
   :format t
   :tidy nil))



