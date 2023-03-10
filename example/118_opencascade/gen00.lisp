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
       (let ((knots (std--vector<double> (curly "0.0"
						"0.0"
						"0.0"
						"1.0"
						"2.0"
						"3.0"
						"3.0"
						"3.0")))
	     (control_points (std--vector<gp_Pnt> (curly
						   (gp_Pnt "0.0" "0.0" "0.0")
						   (gp_Pnt "1.0" "2.0" "0.0")
						   (gp_Pnt "2.0" "-1.0" "0.0")
						   (gp_Pnt "3.0" "0.0" "0.0"))))
	     (points (std--make_shared<TColgp_HArray1OfPnt>
		      1 (control_points.size))))
	 (dotimes (i (control_points.size))
	   (points->SetValue (+ 1 i)
			     (aref control_points i)))))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (display)
       (return 0))))
  )


  
