(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

;; lesson 15
;; https://youtu.be/dq2-evewPeA?list=PL_WFkJrQIY2iVVchOPhl77xl432jeNYfQ

(progn
  (defparameter *source-dir* #P"example/118_opencascade/source01/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  
					;(load "util.lisp")
  (defun ptr0 (name)
    `(,(format nil "opencascade::handle<~a>" name)))

  (defun ptr-new (name-args)
    (let ((name name-args)
	  (args nil))
      (when (listp name-args)
	(setf name (first name-args)
	      args (second name-args)))
      `(,(format nil "opencascade::handle<~a>" name)
	(new ,(if args
		  `(,name ,@args)
		  name)))))

  (defun ptr (name-code)
    (destructuring-bind (name code) name-code
      `(,(format nil "opencascade::handle<~a>" name)
	,code)))
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (include<> vector
		GL/glut.h
		memory)
     (include<> ,@(loop for e in `(
				   ;; Standard_Version
				   ;; TopoDS
				   ;; TopoDS_Edge
				   ;; Geom_BSplineCurve
				   ;; Geom_CartesianPoint
				   ;; TColgp_HArray1OfPnt
				   ;; IGESControl_Reader
				   ;; IGESControl_Writer
				   ;; ShapeUpgrade_UnifySameDomain
				   ;; ShapeUpgrade_RemoveInternalWires
				   ;; ShapeFix_Shape
				   ;; TopTools_ListOfShape
				   TDocStd_Application
				   BinXCAFDrivers
				   BRepPrimAPI_MakeCylinder
				   XCAFDoc_ShapeTool
				   XCAFDoc_ColorTool
				   XCAFDoc_DocumentTool
				   )
			collect
			(format nil "opencascade/~a.hxx" e)))

     (defun BuildWheel (OD W)
       (declare (type "const double" OD W)
		(values TopoDS_Shape))
       (return
	 (BRepPrimAPI_MakeCylinder (gp_Ax2 (gp_Pnt -W/2
						    0 0)
					   (gp--DX))
				   (/ OD 2)
				   W)))
     (defun BuildAxle (D L)
       (declare (type "const double" D L)
		(values TopoDS_Shape))
       (return
	 (BRepPrimAPI_MakeCylinder (gp_Ax2 (gp_Pnt -L/2 0 0)
					   (gp--DX))
				   (/ D 2)
				   L)))

     (defun BuildWheelAxle (wheel axle L)
       (declare (type "const double" L)
		(type "const TopoDS_Shape&" wheel axle)
		(values TopoDS_Shape))
       (let ((comp (TopoDS_Compound))
	     (bbuilder (BRep_Builder))
	     (wheelT_right (gp_Trsf))
	     (wheelT_left (gp_Trsf))
	     )
	 (wheelT_right.SetTranslationPart (gp_Vec L/2 0 0))
	 (wheelT_left.SetTranslationPart (gp_Vec -L/2 0 0))
	 (bbuilder.MakeCompound comp)
	 (bbuilder.Add comp (dot wheel (Moved wheelT_right)))
	  (bbuilder.Add comp (dot wheel (Moved wheelT_left)))
	 (bbuilder.Add comp axle)
	 (return comp)))

     
     (defclass+ t_prototype ()
       "public:"
       (space TopoDS_Shape shape)
       (space TDF_Label label))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       "(void) argc;"
       "(void) argv;"
       (let ((app ,(ptr-new `TDocStd_Application) ))
	 (BinXCAFDrivers--DefineFormat app)
	 (let ((doc ,(ptr0 `TDocStd_Document)))
	   (app->NewDocument (string "BinXCAF")
			     doc)
	   (let ((ST ,(ptr `(XCAFDoc_ShapeTool (XCAFDoc_DocumentTool--ShapeTool (doc->Main)))))
		 (CT ,(ptr `(XCAFDoc_ColorTool (XCAFDoc_DocumentTool--ColorTool (doc->Main))))))
	   
	     (let ((OD 500d0)
		   (W 100d0)
		   (D 50)
		   (L 500)
		   )
	       (let ((wheelProto (t_prototype)))
		 (setf wheelProto.shape (BuildWheel OD W)
		       wheelProto.label (ST->AddShape
					 wheelProto.shape
					 false)))

	       (let ((axleProto (t_prototype)))
		 (setf axleProto.shape (BuildAxle D L)
		       axleProto.label (ST->AddShape
					 axleProto.shape
					 false)
		       ))

	       (let ((wheelAxleProto (t_prototype)))
		 (setf wheelAxleProto.shape (BuildWheelAxle wheelProto.shape
							    axleProto.shape
							    L)
		       wheelAxleProto.label (ST->AddShape
					 wheelAxleProto.shape
					 true)
		       ))
	       ))))
       
       (return 0))))
  )


  
