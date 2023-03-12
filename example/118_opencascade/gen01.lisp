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
     #+nil (include<> vector
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
				   STEPCAFControl_Writer
				   TDataStd_Name
				   Quantity_Color
				   TopExp
				   TopoDS
				   gp_Quaternion
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
	     
	     
	     )
	 

	 (wheelT_right.SetTranslationPart (gp_Vec L/2 0 0))

	 (let ((qn (gp_Quaternion (gp--DY)
				M_PI))
	       (Ry (gp_Trsf)))
	   (Ry.SetRotation qn)
	   (let ((wheelT_left (* (dot wheelT_right (Inverted))
				 Ry)))
	     ))
	 (bbuilder.MakeCompound comp)
	 (bbuilder.Add comp (dot wheel (Moved wheelT_right)))
	 (bbuilder.Add comp (dot wheel (Moved wheelT_left)))
	 (bbuilder.Add comp axle)
	 (return comp)))


     (defun BuildChassis (wheelAxle CL)
       (declare (type "const double" CL)
		(type "const TopoDS_Shape&" wheelAxle)
		(values TopoDS_Shape))
       (let ((comp (TopoDS_Compound))
	     (bbuilder (BRep_Builder))
	     (frontT (gp_Trsf))
	     (rearT (gp_Trsf))
	     )
	
	 (frontT.SetTranslationPart (gp_Vec 0 CL/2 0))
	 (rearT.SetTranslationPart (gp_Vec 0 -CL/2 0))
	
	 (bbuilder.MakeCompound comp)
	 (bbuilder.Add comp (dot wheelAxle (Moved frontT)))
	  (bbuilder.Add comp (dot wheelAxle (Moved rearT)))
	 
	 (return comp)))

     (defun WriteStep (doc filename)
       (declare (type "const char*" filename)
		(type "const Handle(TDocStd_Document)&" doc)
		(values bool))
       (let ((Writer (STEPCAFControl_Writer)))
	 (unless 
	  (Writer.Transfer doc)
	   (return false))
	 (unless (== IFSelect_RetDone
		     (Writer.Write filename))
	   (return false))
	 (return true))
       )

     

     
     (defclass+ t_prototype ()
       "public:"
       (space TopoDS_Shape shape)
       (space TDF_Label label))

     (defclass+ t_wheelPrototype "public t_prototype"
       "public:"
       (space TopoDS_Face frontFace)
       (space TDF_Label frontFaceLabel))
     
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
	   
	     (let ((OD 500d0) ;; wheel-diam
		   (W 100d0)  ;; wheel-width
		   (D 50d0)   ;; axle-diam
		   (L 500d0)  ;; axle-length
		   (CL 600d0) ;; chassis-length
		   )

	       ,@(loop for e in `((:name wheel :params (OD W) :color (1 0 0) :type t_wheelPrototype)
				  (:name axle :params (D L) :color (0 1 0))
				  (:name wheel-axle :assemble (wheel axle) :params (L))
				  (:name chassis :assemble (wheel-axle) :params (CL)))
		       collect
		       (destructuring-bind (&key name assemble params color (type 't_prototype)) e
			 (let ((proto (cl-change-case:camel-case (format nil "~a-proto" name)))
			       (build (cl-change-case:pascal-case (format nil "build-~a" name))))
			   `(let ((,proto
				    (,type)))
			      (setf (dot ,proto shape) ,(if assemble
							    `(,build
							      ,@(loop for obj in assemble
								      collect
								      `(dot ,(cl-change-case:camel-case
									      (format nil "~a-proto" obj))
									    shape))
							      ,@params)
							    `(,build ,@params))
				    (dot ,proto label) (ST->AddShape
							(dot ,proto shape)
							,(if assemble `true `false)))
			      (TDataStd_Name--Set (dot ,proto label)
						  (string ,name))
			      ,(if color
				   `(CT->SetColor (dot ,proto label)
						  (Quantity_Color ,@color
								  Quantity_TOC_RGB)
						  XCAFDoc_ColorGen)
				   `"")))))

	       (let ((allWheelFaces (TopTools_IndexedMapOfShape)))
		 (TopExp--MapShapes
		  wheelProto.shape
		  TopAbs_FACE
		  allWheelFaces)
		 (setf (dot wheelProto frontFace) (TopoDS--Face (allWheelFaces 2))
		       (dot wheelProto frontFaceLabel) (-> ST (AddSubShape wheelProto.label
									   wheelProto.frontFace)))
		 (CT->SetColor (dot wheelProto frontFaceLabel)
			       (Quantity_Color 0 0 1
					       Quantity_TOC_RGB)
			       XCAFDoc_ColorSurf))
	       ))))

       
       (let  ((status (app->SaveAs doc
				   (string "doc.xbf"))))
	 (unless (==  PCDM_SS_OK status)
	   (return 1)))

       (WriteStep doc (string "o.stp"))
       (return 0))))
  )


  
