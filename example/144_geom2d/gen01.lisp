(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

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
	(comments "Expose the v_2d<float> class to Python as \"Vector2D\"")
	(dot (py--class_<v_2d<float>> m (string "Vector2D"))
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
		   (return (+ (string "<Vector2D x=")
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


	(comments "Expose the circle<float> class to Python as \"Circle\"")
	(dot (py--class_<circle<float>> m (string "Circle"))
	     (def ("py::init<const v_2d<float>&, float>"))
	     ,@(loop for e in `(pos radius)
		     collect
		     `(def_readwrite
			  (string ,(format nil "~a" e))
			  ,(format nil "&circle<float>::~a" e)))
	     (def (string "__repr__")
		 (lambda (c)
		   (declare (capture "")
			    (type "const circle<float>&" c))
		   (return (+ (string "<Circle pos=")
			      (dot c pos (str))
			      (string ", radius=")
			      (std--to_string c.radius)
			      (string ">")))))
	     )
	(comments "Expose the contains function for circle and point")
	(m.def (string "contains")
	       "(bool (*) (const circle<float>&, const v_2d<float>&)) &contains")
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



