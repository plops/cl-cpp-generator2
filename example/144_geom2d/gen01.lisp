(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  ;(ql:quickload "cl-ppcre")
  ;(ql:quickload "cl-change-case")
  )

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ()
  (defparameter *source-dir* #P"example/144_geom2d/source01/python/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  #+nil
  (let* ((name `DiagramBase)
	 (members `((max-cores :type int :param t)
		    (max-points :type int :param t)
		    (diagrams :type "std::vector<DiagramData>")
		    ;(x :type "std::vector<float>")
		    ;(y :type "std::vector<float>")
		    (name-y :type "std::string" :param t)
		    (time-points :type "std::deque<float>"))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> vector deque string)
			(space struct DiagramData (progn
					   "std::string name;"
					   "std::deque<float> values;"
					   ))
			(doc "@brief The DiagramBase class represents a base class for diagrams."))
     :implementation-preamble
     `(do0
       
       (include<>
		  stdexcept
		  format
		  )
       )
     :code `(do0
	     
	     (defclass ,name ()
	       "public:"
	       
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param (initform 0)) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param (initform 0)) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param (initform 0)) e
					(let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					      (nname_ (format nil "~a_"
							      (cl-change-case:snake-case (format nil "~a" name)))))
					  (cond
					    (param
					     `(,nname_ ,nname)) 
					    (initform
					     `(,nname_ ,initform)))))))
		   )
		  (explicit)	    
		  (values :constructor)
		  )
		 )

	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (get (cl-change-case:pascal-case (format nil "get-~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(defmethod ,get ()
					 (declare (values ,type))
					 (return ,nname_))))))
	       
	       "protected:"
	       
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_))))))))

    )

  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   
   `(do0
     (include olcUTIL_Geometry2D.h)
     (include<> 
					;format
					;unistd.h
					;vector deque chrono
					;cmath
      pybind11/pybind11.h
      ;pybind11/stl.h
      )
     (space namespace (setf py pybind11))
     (space using namespace olc)

     (space
      (PYBIND11_MODULE pygeometry m)
      (progn
	(dot (py--class_<v_2d<float>> m (string "v_2d"))
	     (def ("py::init<float,float>"))
	     ,@(loop for e in `(x y)
		     collect
		     `(def_readwrite
			  (string ,(format nil "~a" e))
			  ,(format nil "&v_2d<float>::~a" e)))

	     ,@(loop for e in `(area mag mag2 norm perp floor ceil max min dot
				     cross cart polar clamp lerp reflect)
		     collect
		     `(def
			  (string ,(format nil "~a" e))
			  ,(format nil "&v_2d<float>::~a" e)))

	     ,@(loop for e in `(== "!=" + - * / += -=)
		     collect
		     `(def py--self
			  ,(format nil "~a" e)
			py--self))
	     ,@(loop for e in `(*= /=)
		     collect
		     `(def py--self
			  ,(format nil "~a" e)
			(float)))

	     ,@(loop for e in `(__str__ __repr__)
		     collect
		     `(def 
			  (string ,(format nil "~a" e))
			  "&v_2d<float>::str"))
	     
	     
	     )
	,@(loop for e in `(* +)
		appending
		
		`((dot m (def (lambda (lhs rhs)
				(declare (capture "")
					 (type float lhs)
					 (type "v_2d<float>&" rhs))
				(return (,e lhs rhs))
				)))
		  (dot m (def (lambda (lhs rhs)
				(declare (capture "")
					 (type float rhs)
					 (type "v_2d<float>&" lhs))
				(return (,e lhs rhs))
				)))))

	,@(loop for e in `(pi epsilon)
		collect
		`(setf (m.attr (string ,e))
		      ,(format nil "utils::geom2d::~a" e)))

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



