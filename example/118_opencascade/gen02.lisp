(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

;; lesson 18
;; video: https://youtu.be/kPcD5liq8Cs?list=PL_WFkJrQIY2iVVchOPhl77xl432jeNYfQ
;; documentation: https://dev.opencascade.org/doc/overview/html/occt__tutorial.html
;; listing: https://git.dev.opencascade.org/gitweb/?p=occt.git;a=blob;f=samples/qt/Tutorial/src/MakeBottle.cxx;h=c89b96df84dc54bcedfd7bcc57c061aee386321b;hb=HEAD
(progn
  (defparameter *source-dir* #P"example/118_opencascade/source02/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  
  (load "util.lisp")
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
     (include<> iostream)
     ,@(inc `((b-rep tool)
	      (b-rep-algo-a-p-i fuse)
	      (b-rep-builder-a-p-i make-edge make-face make-wire transform)
	      (b-rep-fillet-a-p-i make-fillet)
	      (b-rep-lib "")
	      (b-rep-offset-a-p-i make-thick-solid thru-sections)
	      (b-rep-prim-a-p-i make-cylinder make-prism)
	      (g-c make-arc-of-circle make-segment)
	      (g-c-e2d make-segment)
	      ("gp" "" ax1 ax2 ax2d dir dir2d pnt pnt2d trsf vec)
	      (geom cylindrical-surface plane surface trimmed-curve)
	      (geom2d ellipse trimmed-curve)
	      (top-exp explorer)
	      (topo-d-s edge face wire shape compound)
	      (top-tools list-of-shape)
	      (t-doc-std application)
	      (bin-x-c-a-f-drivers "")
	      (x-c-a-f-doc shape-tool document-tool)
	      ("STEPCAFControl" writer)))

     (defun MakeBottle (myWidth myHeight myThickness)
       (declare (type "const Standard_Real" myWidth myHeight myThickness)
		(values TopoDS_Shape))
       (let ((p1 (gp_Pnt -myWidth/2 0 0))
	     (p2 (gp_Pnt -myWidth/2 -myThickness/4 0))
	     (p3 (gp_Pnt          0 -myThickness/2 0))
	     (p4 (gp_Pnt myWidth/2 -myThickness/4 0))
	     (p5 (gp_Pnt myWidth/2 0 0))
	     (anArcOfCircle ,(ptr `(Geom_TrimmedCurve (GC_MakeArcOfCircle p2 p3 p4))))
	     (aSegment1 ,(ptr `(Geom_TrimmedCurve (GC_MakeSegment p1 p2))))
	     (aSegment2 ,(ptr `(Geom_TrimmedCurve (GC_MakeSegment p4 p5))))
	     (anEdge1 (BRepBuilderAPI_MakeEdge aSegment1))
	     (anEdge2 (BRepBuilderAPI_MakeEdge anArcOfCircle))
	     (anEdge3 (BRepBuilderAPI_MakeEdge aSegment2))
	     (aWire (BRepBuilderAPI_MakeWire anEdge1 anEdge2 anEdge3))
	     ;; i could use std::invoke to make the invocation more obvious
	     (aTrsf ((lambda ()
		     #+nil (declare (values auto)
			)
		       (let ((xAxis (gp--OX))
			     (a (gp_Trsf)))
			 (a.SetMirror xAxis)
			 (return a)))))
	     (aBRepTrsf (BRepBuilderAPI_Transform aWire aTrsf))
	     (aMirroredShape (dot aBRepTrsf
				(Shape)))
	     (aMirroredWire (TopoDS--Wire aMirroredShape))
	     (mkWire ((lambda ()
			(declare ;(values auto)
				 (capture "&"))
			(let ((a (BRepBuilderAPI_MakeWire)))
			  (a.Add aWire)
			  (a.Add aMirroredWire)
			  (return a)))))
	     (myWireProfile (mkWire.Wire))
	     (myFaceProfile (BRepBuilderAPI_MakeFace myWireProfile))
	     (aPrismVec (gp_Vec 0 0 myHeight))
	     ;; extruded outside shape of the body of the bottle
	     (myBody (BRepPrimAPI_MakePrism myFaceProfile
					    aPrismVec)))
	 (declare (type TopoDS_Shape myBody))
	 #-nil (let ((mkFillet (
				(lambda ()
				  (declare (capture "&"))
				  (let ((fillet (BRepFilletAPI_MakeFillet myBody))
					(edgeExplorer (TopExp_Explorer myBody TopAbs_EDGE)))
				    (while (edgeExplorer.More)
					   (let ((edge (TopoDS--Edge (edgeExplorer.Current))))
					     (fillet.Add (/ myThickness 12)
							 edge)
					     (edgeExplorer.Next)))
				    (return fillet)
				    ))))
		     
		     )
		 (comments "make the outside of the body rounder")
		 (setf myBody (mkFillet.Shape)))

	 #-nil (let ((neckLocation (gp_Pnt 0 0 myHeight))
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
	 
	 #-nil (let ((facesToRemove
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
	 
	 (let ((aRes ((lambda ()
			(declare
			
			 (capture "&")
			 (declare (type auto body)))
			(let ((a (TopoDS_Compound))
			      (b (BRep_Builder)))
			  (b.MakeCompound a)
			  (b.Add a myBody)
			  (return a))))))
	   (return aRes))
	 ))

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
	     (let ((W 30d0)
		   (H 40d0)
		   (T 10d0)
		   (shape (MakeBottle W H T))
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


  
