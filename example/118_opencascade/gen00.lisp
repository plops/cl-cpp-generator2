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
		GL/glut.h)
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
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       
       (return 0))))
  )


  
