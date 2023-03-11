(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/118_opencascade/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  
  ;(load "util.lisp")
  
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (include<> vector
		GL/glut.h
		memory)
     (include<> ,@(loop for e in `(Standard_Version
				   TopoDS
				   TopoDS_Edge
				   Geom_BSplineCurve
				   Geom_CartesianPoint
				   TColgp_HArray1OfPnt
				   IGESControl_Reader
				   IGESControl_Writer
				   ShapeUpgrade_UnifySameDomain
				   ShapeUpgrade_RemoveInternalWires
				   ShapeFix_Shape
				   TopTools_ListOfShape)
			collect
			(format nil "opencascade/~a.hxx" e)))

     (defun display ()
       (let ((knots (std--vector<double> (curly 0 1 2 3)
					 #+nil (curly "0.0"
						      "0.0"
						      "0.0"
						      "1.0"
						      "2.0"
						      "3.0"
						      "3.0"
						      "3.0")))
	     (knot_multi (std--vector<int> (curly 3 1 1 3)))
	     
	     (control_points (std--vector<gp_Pnt> (curly
						   (gp_Pnt "0.0" "0.0" "0.0")
						   (gp_Pnt "1.0" "2.0" "0.0")
						   (gp_Pnt "2.0" "-1.0" "0.0")
						   (gp_Pnt "3.0" "0.0" "0.0"))))
	     (points #+nil
		     (std--make_shared<TColgp_HArray1OfPnt>
		      1 (control_points.size))
		     (new (TColgp_HArray1OfPnt
		       1 (control_points.size)))))
	 (declare (type "Handle(TColgp_HArray1OfPnt)" points))
	 (for-range (p control_points)
		    (let ((i (- &p (ref (aref control_points 0)))))
		      (points->SetValue (+ i 1)
					(aref control_points i))))
	 (let ((curve #+nil (std--make_shared<Geom_BSplineCurve> points
							  knots
							  3)
		      (new (Geom_BSplineCurve points
					  knots
					  knot_multi
					  3))))
	   (declare (type "Handle(Geom_BSplineCurve)" curve)))))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (display)
       (return 0))))
  )


  
