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
	      ))
      `(defun MakeParabolicCrownedPulley (shaftDiameter centralDiameter edgeDiameter pulleyThickness)
	(declare (type "const Standard_Real" shaftDiameter centralDiameter edgeDiameter pulleyThickness)
		 (values TopoDS_Shape))
	(let ((sphere (BRepPrimAPI_MakeSphere (gp_Pnt 0 0 0)
					      centralDiameter/2))
	      (axis (gp_Ax2 (gp_Pnt 0 0 0)
			    (gp_Dir 0 0 1)))
	      (cylBig ,(translate `(:z -pulleyThickness/2
				      :code (BRepPrimAPI_MakeCylinder axis centralDiameter/2 pulleyThickness))))
	      (cylShaft (BRepPrimAPI_MakeCylinder axis shaftDiameter/2 (- 8.31 5.87)))
	      (cylShaftFullLength (BRepPrimAPI_MakeCylinder axis shaftDiameter/2 pulleyThickness))
	      (shaftFlattening ,(translate `(:x -2.94/2
						:y -10
						:z (+ -pulleyThickness/2 (- 8.31 5.87))
				      :code (BRepPrimAPI_MakeBox 2.94 20 5.87))))
	      (cylShaft2 ,(common `(cylShaftFullLength
				    shaftFlattening))
			 #+nil(BRepAlgoAPI_Common (cylShaftFullLength.Shape)
					     (dot shaftFlattening	(Shape))))
	      #+nil (cylShaft3 )
	     
	      (disk ,(common `(sphere cylBig))
		    #+nil(BRepAlgoAPI_Common (sphere.Shape)
					(cylBig.Shape)))
	      #+nil (diskWithHole (BRepAlgoAPI_Cut (disk.Shape)
						   (dot (BRepBuilderAPI_Transform cylShaft aTrsf)
							(Shape))))
	     
	     
	      (shape (dot		;diskWithHole

		      disk
		      #+nil shaftFlattening
		      #+nil cylShaft3
		      #+nil
		      (BRepAlgoAPI_Cut
		       (disk.Shape)
		       cylShaft3 )
		     
		      (Shape)
		      )))
	  (declare (type TopoDS_Shape shape))
	 
	 
	  #+nil (let ((neckLocation (gp_Pnt 0 0 myHeight))
		      (neckAxis (gp--DZ))
		      (neckAx2 (gp_Ax2 neckLocation neckAxis))
		      (myNeckRadius (/ myThickness 4d0))
		      (myNeckHeight (/ myHeight 10d0))
		      (MKCylinder (BRepPrimAPI_MakeCylinder neckAx2 myNeckRadius myNeckHeight))
		      (myNeck (MKCylinder.Shape ))
		     
		      )
		  (comments "attach the neck to the body")
		  (setf myBody
			(BRepAlgoAPI_Fuse myBody myNeck)))


	  #+nil (let ((mkFillet (
				 (lambda ()
				   (declare (capture "&"))
				   (let ((fillet (BRepFilletAPI_MakeFillet myBody))
					 (edgeExplorer (TopExp_Explorer myBody TopAbs_EDGE)))
				     (while (edgeExplorer.More)
					    (let ((cur (edgeExplorer.Current))
						  (edge (TopoDS--Edge cur))
						 
						  (mz ((lambda ()
							 (declare (capture "&"))
							 (let ((uStart (Standard_Real 0))
							       (uEnd (Standard_Real 0))
							       (curve
								 ,(ptr
								   `(Geom_Curve (BRep_Tool--Curve edge
												  uStart
												  uEnd))))
							       (N 100)
							       (deltaU (/ (- uEnd uStart) (* 1s0 N)))
							       (points ((lambda ()
									  (declare (capture "&"))
									  (let ((points (std--vector<gp_Pnt>)))
									    (dotimes (i N)
									      (let ((u (+ uStart (* deltaU i)
											  )))
										(points.emplace_back (curve->Value u))))
									    (return points)))))
							      
							       (maxPointIt (std--max_element
									    (points.begin)
									    (points.end)
									    (lambda (a b)
									      (return (< (a.Z)
											 (b.Z)))))))
							   (let ((maxPoint (deref maxPointIt) ; (curve->Value uEnd)
									   ))
							     (return (maxPoint.Z))))))))
					      ,(lprint :vars `(mz))
					      (comments "i want to fillet the edge where the neck attaches to the body but not the top of the neck")
					      (when (<= mz 40)
						(fillet.Add (/ myThickness 12)
							    edge))
					      (edgeExplorer.Next)))
				     (return fillet)
				     ))))
		     
		      )
		  (comments "make the outside of the body rounder")
		  (setf myBody (mkFillet.Shape)))
	 
	  #+nil (let ((facesToRemove
			((lambda ()
			   (declare (capture "&")
				    )
			   (let ((faceToRemove (TopoDS_Face))
				 (zMax (Standard_Real -100))
			  
				 (explorer (TopExp_Explorer myBody  TopAbs_FACE)))
			     (for (() (explorer.More) (explorer.Next))
				  (let ((aFace (TopoDS--Face (explorer.Current)))
					(bas (BRepAdaptor_Surface aFace)))
					;,(lprint :msg "face" :vars `( (bas.GetType) GeomAbs_Plane))
				    (when (== GeomAbs_Plane
					      (bas.GetType))
				      (let ((plane (bas.Plane))
					    (aPnt (plane.Location)))
					(unless (dot plane
						     (Axis)
						     (Direction)
						     (IsParallel (gp--DZ)
								 (/ (* 1.0 M_PI)
								    180.0)))
				    
					  continue)
					(let ((aZ (aPnt.Z)))
					  (when (< zMax aZ)
					;,(lprint :vars `(zMax aZ))
					    (setf zMax aZ
						  faceToRemove aFace)))
					))
				    ))

			     (let ((facesToRemove (TopTools_ListOfShape)))
			       (facesToRemove.Append faceToRemove)
			       (return facesToRemove))
			     ))))
		      )
		  (do0
		   (comments "make inside of the bottle hollow")
		   (setf myBody
			     
			 ((lambda ()
			    (declare (capture "&")
				     )
			    (let ((aSolidMaker (BRepOffsetAPI_MakeThickSolid)))
			      (aSolidMaker.MakeThickSolidByJoin myBody facesToRemove
								-myThickness/50 1e-3)
			      (return (aSolidMaker.Shape))))
			  ))))

	  (let ((unify (ShapeUpgrade_UnifySameDomain shape)))
	    (comments "remove unneccessary seams")
	    (unify.Build)
	    (setf shape
		  (dot unify
		       (Shape))))
	 
	  (return shape)
	  #+nil (let ((aRes ((lambda ()
			       (declare
			
				(capture "&")
				(declare (type auto body)))
			       (let ((a (TopoDS_Compound))
				     (b (BRep_Builder)))
				 (comments "return compound so that we still see output in case a boolean operation fails")
				 (b.MakeCompound a)
				 (setf myBody
				       (BRepAlgoAPI_Fuse myBody myThreading))
				 (let ((unify (ShapeUpgrade_UnifySameDomain myBody)))
				   (comments "remove unneccessary seams")
				   (unify.Build)
				   (setf myBody
					 (dot unify
					      (Shape))))
				 (b.Add a myBody)
			  
				 (return a))))))
		  (return aRes))
	  )))

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
	     (let (
		   (shaftDiameter 4.92)
		   (centralDiameter 20.0)
		   (edgeDiameter 16.0)
		   (pulleyThickness 8.31)
		   (shape (MakeParabolicCrownedPulley shaftDiameter centralDiameter edgeDiameter pulleyThickness))
		   
		   (label (ST->AddShape shape false))))
	     )))
       (do0
	(let  ((status (app->SaveAs doc
				    (string "doc.xbf"))))
	  (unless (==  PCDM_SS_OK status)
	    (return 1)))

	(WriteStep doc (string "o.stp")))
       (return 0))))
  )


  
