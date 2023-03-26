(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)
(progn
  (defparameter *source-dir* #P"example/118_opencascade/source03/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  
  (load "util.lisp")
  (defun ptr0 (name)
    `(,(format nil "opencascade::handle<~a>" name)))

  #+nil
  (defun ptr-new (name-args)
    ;; name can be a list (name type)  handle<name> new(type, *args)
    (let ((name name-args)
	  (args nil))
      (when (listp name-args)
	(setf name (first name-args)
	      args (second name-args)))
      (let ((type name))
	(when (listp name)
	  (setf type (second name)
		name (first name)))
	`(,(format nil "opencascade::handle<~a>" name)
	  (new ,(if args
		    `(,type ,@args)
		    type))))))
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

  (defun inc (inc-defs)
    (loop for inc-def in inc-defs
	  collect
	  (destructuring-bind (prefix &rest headers) inc-def
	    `(include<> ,@(mapcar #'(lambda (header)
				      (let ((p (cl-change-case:pascal-case (format nil "~a" prefix))))
					(when (stringp prefix)
					  (setf p prefix))
					(if (string= header "")
					    (format nil "~a.hxx"
						    p
						    )
					    (format nil "~a_~a.hxx"
						    p
						    (cl-change-case:pascal-case (format nil "~a" header))))))
				  headers)))))
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (include<> iostream
		vector
		algorithm)
     ,@(inc `((b-rep tool)
	      (b-rep-algo-a-p-i fuse cut common)
	      (b-rep-builder-a-p-i make-edge make-face make-wire transform)
	      (b-rep-fillet-a-p-i make-fillet)
	      ;(b-rep-chamfer-a-p-i make-chamfer)
	      (b-rep-lib "")
	      (b-rep-offset-a-p-i make-thick-solid thru-sections)
	      (b-rep-prim-a-p-i make-cylinder make-prism make-revol MakeSphere MakeBox)
	      (g-c make-arc-of-circle make-segment)
	      (g-c-e2d make-segment)
	      ("gp" "" ax1 ax2 ax2d dir dir2d pnt pnt2d trsf vec)
	      (geom cylindrical-surface plane surface trimmed-curve curve parabola)
	      (geom2d ellipse trimmed-curve parabola)
	      (top-exp explorer)
	      (topo-d-s edge face wire shape compound)
	      (top-tools list-of-shape)
	      (t-doc-std application)
	      (bin-x-c-a-f-drivers "")
	      (x-c-a-f-doc shape-tool document-tool)
	      ("STEPCAFControl" writer)
	      (shape-upgrade unify-same-domain)))

     ,(flet (
	     (translate (xyz-code)
	       (destructuring-bind (&key (x 0) (y 0) (z 0) code) xyz-code
		`(BRepBuilderAPI_Transform
		  ,code
		  ((lambda ()
		     (declare (capture "&"))
		     (let ((a (gp_Trsf)))
		       (a.SetTranslation (gp_Vec ,x ,y ,z))
		       (return a)))))))
	    (fuse (shapes)
	      (let ((res `(BRepAlgoAPI_Fuse ,(elt shapes 0)
					    ,(elt shapes 1))))
		(loop for s in (subseq shapes 2)
		      do
		      (setf res `(BRepAlgoAPI_Fuse ,s ,res)))
		res)
	      )
	     (common (shapes)
	      (let ((res `(BRepAlgoAPI_Common ,(elt shapes 0)
					    ,(elt shapes 1))))
		(loop for s in (subseq shapes 2)
		      do
		      (setf res `(BRepAlgoAPI_Common ,s ,res)))
		res)
	      )
	     (cut (shapes)
	      (let ((res `(BRepAlgoAPI_Cut ,(elt shapes 0)
					    ,(elt shapes 1))))
		(loop for s in (subseq shapes 2)
		      do
		      (setf res `(BRepAlgoAPI_Cut ,s ,res)))
		res)
	      )
	     (fillet (thickness-shape)
	       (destructuring-bind (&key (radius 1.0) shape) thickness-shape
		`((lambda ()
		    (declare (capture "&"))
		    (let ((fillet (BRepFilletAPI_MakeFillet
				   ,shape))
			  (edgeExplorer (TopExp_Explorer ,shape TopAbs_EDGE)))
		      (while (edgeExplorer.More)
			     (let ((cur (edgeExplorer.Current))
				   (edge (TopoDS--Edge cur))
				   )
				,(lprint :msg "add fillet")		 
			       (fillet.Add ,radius
					   edge)
			       (edgeExplorer.Next)))
		      (return fillet )))))))

	
	`(do0
	 (defun MakeCrownedPulleyFlatShaft (shaftDiameter centralDiameter pulleyThickness
					    shaftLength flatLength flatThickness)
	   (declare (type "const Standard_Real" shaftDiameter centralDiameter pulleyThickness shaftLength flatLength flatThickness)
		    (values TopoDS_Shape))
	   (let ((sphere (BRepPrimAPI_MakeSphere (gp_Pnt 0 0 pulleyThickness/2)
						 centralDiameter/2))
		 (axis (gp_Ax2 (gp_Pnt 0 0 0)
			       (gp_Dir 0 0 1)))
		 (cylBig (BRepPrimAPI_MakeCylinder axis centralDiameter/2 pulleyThickness))
		 ;; full circular hole of the shaft
		 (cylShaft ,(translate `(:z flatLength
					 :code (BRepPrimAPI_MakeCylinder axis shaftDiameter/2 (- shaftLength flatLength)))))
		 ;; the hole through the entire disk (the flat sides will be added)
		 (cylShaftFullLength (BRepPrimAPI_MakeCylinder axis shaftDiameter/2 pulleyThickness))
		 ;; the shaft is flattened at the top
		 (shaftFlattening ,(translate `(:x -flatThickness/2
						:y -centralDiameter/2
						:z 0
						:code (BRepPrimAPI_MakeBox flatThickness centralDiameter flatLength))))
		 (cylShaft2 ,(common `(cylShaftFullLength
				       shaftFlattening))
			    )	     
		 (disk 
		   ,(common `(sphere cylBig))
		   )
		 
	     	 (shape ,(cut `(disk ,(fuse `(cylShaft2 cylShaft))))
			)
		 )
	     
	     (declare (type TopoDS_Shape shape))
	     (let ((unify (ShapeUpgrade_UnifySameDomain shape)))
	       (comments "remove unneccessary seams")
	       (unify.Build)
	       (setf shape
		     (dot unify
			  (Shape))))
	     
	     (return shape)
	     
	     ))
	 (defun MakeCrownedPulley (shaftDiameter centralDiameter pulleyThickness
					    )
	   (declare (type "const Standard_Real" shaftDiameter centralDiameter pulleyThickness )
		    (values TopoDS_Shape))
	   (let ((sphere (BRepPrimAPI_MakeSphere (gp_Pnt 0 0 pulleyThickness/2)
						 centralDiameter/2))
		 (axis (gp_Ax2 (gp_Pnt 0 0 0)
			       (gp_Dir 0 0 1)))
		 (cylBig (BRepPrimAPI_MakeCylinder axis centralDiameter/2 pulleyThickness))
		 
		 ;; the hole through the entire disk 
		 (cylShaftFullLength (BRepPrimAPI_MakeCylinder axis shaftDiameter/2 pulleyThickness))
		 
		 	     
		 (disk 
		   ,(common `(sphere cylBig)))
		 
	     	 (shape ,(translate `(:x 25 :code ,(cut `(disk cylShaftFullLength))))
			)
		 )
	     
	     (declare (type TopoDS_Shape shape))
	     (let ((unify (ShapeUpgrade_UnifySameDomain shape)))
	       (comments "remove unneccessary seams")
	       (unify.Build)
	       (setf shape
		     (dot unify
			  (Shape))))
	     
	     (return  shape)
	     
	     ))))

     (defun WriteStep (doc filename)
       (declare (type "const char*" filename)
		(type "const Handle(TDocStd_Document)&" doc)
		(values bool))
       (let ((Writer (STEPCAFControl_Writer)))
	 (unless 
	  (Writer.Transfer doc)
	   (return false))
	 (return (== IFSelect_RetDone
		     (Writer.Write filename))
	   ))
       )
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
					;(CT ,(ptr `(XCAFDoc_ColorTool (XCAFDoc_DocumentTool--ColorTool (doc->Main)))))
		 )
	     (let ( (shape (MakeCrownedPulleyFlatShaft (+ .01 4.92) 20.0 8.31 8.31 5.87 (+ .01 2.94)))
		   
		   (label (ST->AddShape shape false))))
	     (let ( (shape2 (MakeCrownedPulley (+ .01 12.82) 20.0 8.31 ))
		   
		   (label2 (ST->AddShape shape2 false))))
	     )))
       (do0
	(let  ((status (app->SaveAs doc
				    (string "doc.xbf"))))
	  (unless (==  PCDM_SS_OK status)
	    (return 1)))

	(WriteStep doc (string "o.stp")))
       (return 0))))
  )


  
