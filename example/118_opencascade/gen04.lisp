(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)
(progn
  (defparameter *source-dir* #P"example/118_opencascade/source04/")
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

	  #+nil
	  (myThreading
	   ((lambda ()
	      (declare (capture "&"))
	      (let ((aCyl1 ,(ptr-new `(Geom_CylindricalSurface
				       (neckAx2 (* .99 myNeckRadius)))))
		    (aCyl2 ,(ptr-new `(Geom_CylindricalSurface
				       (neckAx2 (* 1.05 myNeckRadius)))))
		    (aPnt (gp_Pnt2d (* 2 M_PI)
				    (/ myNeckHeight 2)))
		    (aDir (gp_Dir2d (* 2 M_PI)
				    (/ myNeckHeight 4)))
		    (anAx2d (gp_Ax2d aPnt aDir))
		    (aMajor (Standard_Real (* 2 M_PI)))
		    (aMinor (Standard_Real (/ myNeckHeight 10)))
		    (anEllipse1 ,(ptr-new `(Geom2d_Ellipse (anAx2d aMajor aMinor))))
		    (anEllipse2 ,(ptr-new `(Geom2d_Ellipse (anAx2d aMajor (/ aMinor 4)))))
		    (anArc1 ,(ptr-new `(Geom2d_TrimmedCurve (anEllipse1 0 M_PI))))
		    (anArc2 ,(ptr-new `(Geom2d_TrimmedCurve (anEllipse2 0 M_PI))))
		    (anEllipsePnt1 (-> anEllipse1 (Value 0)))
		    (anEllipsePnt2 (-> anEllipse1 (Value M_PI)))
		    (aSegment ,(ptr  `(Geom2d_TrimmedCurve 
				       (GCE2d_MakeSegment anEllipsePnt1 anEllipsePnt2))))
		    
		    ,@(loop for (a b c d) in `((1 1 anArc1 aCyl1)
					       (2 1 aSegment aCyl1)
					       (1 2 anArc2 aCyl2)
					       (2 2 aSegment aCyl2)
					       )
			    collect
			    `(,(format nil "anEdge~aOnSurf~a" a b) (BRepBuilderAPI_MakeEdge ,c ,d)))
		    (threadingWire1 (BRepBuilderAPI_MakeWire anEdge1OnSurf1 anEdge2OnSurf1))
		    (threadingWire2 (BRepBuilderAPI_MakeWire anEdge1OnSurf2 anEdge2OnSurf2))
			  
		    )
		(do0g (BRepLib--BuildCurves3d threadingWire1)
		      (BRepLib--BuildCurves3d threadingWire2)
		      (let ((aTool (BRepOffsetAPI_ThruSections Standard_True)))
			(aTool.AddWire threadingWire1)
			(aTool.AddWire threadingWire2)
			(comments "because they come from ellipses, the splines will be compatible")
			(aTool.CheckCompatibility Standard_False)
			(comments "create thread")
			(let ((myThreading (aTool.Shape)))
			  (return myThreading))))))))
	  (comments "https://en.wikipedia.org/wiki/ISO_metric_screw_thread"
		    "https://dev.opencascade.org/doc/overview/html/occt__tutorial.html")
	  (defun MakeM2ScrewHole ()
	    (declare (type "const Standard_Real" )
		     (values TopoDS_Shape))
	   
	    (let ((depth 10)
		  (Dmaj 2d0)
		  (P .4d0)
		  (H (*  ,(* .5d0 (sqrt 3d0)) P))
		  (Dmin (- Dmaj (* 2 5 (/ H 8))))
		  (Dp (- Dmaj (* 2 3 (/ H 8))))
		  (axis (gp_Ax2 (gp_Pnt 0 0 0)
				(gp_Dir 0 0 1)))
		  (pnt (gp_Pnt2d (* 2 M_PI)
				 (* .5 depth)))
		  (dir (gp_Dir2d (* 2 M_PI)
				 (* .25 depth)))
		  (neckLocation (gp_Pnt 0 0 0))
		  (neckAxis (gp--DZ))
		  (neckAx2 (gp_Ax2 neckLocation neckAxis))
		  
		  (cylWide (BRepPrimAPI_MakeCylinder axis Dmaj (* .5 depth)))
		  (cylWideSurf ,(ptr-new `(Geom_CylindricalSurface
					   (neckAx2 Dmaj))))
		  (cylThin (BRepPrimAPI_MakeCylinder axis Dmin depth))
		  (cylThinSurf ,(ptr-new `(Geom_CylindricalSurface
					   (neckAx2 Dmin))))
		  
		  (anAx2d (gp_Ax2d pnt dir))
		  (e1 ,(ptr-new `(Geom2d_Ellipse (anAx2d (* 2 M_PI)
							 (- P (/ P 4))))))
		  (e2 ,(ptr-new `(Geom2d_Ellipse (anAx2d (* 2 M_PI)
							 (/ P 8)))))
		  (arc1 ,(ptr-new `(Geom2d_TrimmedCurve (e1 0 M_PI))))
		  (arc2 ,(ptr-new `(Geom2d_TrimmedCurve (e2 0 M_PI))))
		  (ep1 (-> e1 (Value 0)))
		  (ep2 (-> e1 (Value M_PI)))
		  (seg ,(ptr `(Geom2d_TrimmedCurve
			       (GCE2d_MakeSegment ep1 ep2))))
		  ,@(loop for (a b c d) in `((1 1 arc1 cylThinSurf)
					       (2 1 seg cylThinSurf)
					       (1 2 arc2 cylWideSurf)
					       (2 2 seg cylWideSurf)
					       )
			    collect
			    `(,(format nil "anEdge~aOnSurf~a" a b) (BRepBuilderAPI_MakeEdge ,c ,d)))
		  (threadingWire1 (BRepBuilderAPI_MakeWire anEdge1OnSurf1 anEdge2OnSurf1))
		  (threadingWire2 (BRepBuilderAPI_MakeWire anEdge1OnSurf2 anEdge2OnSurf2))
		
		  
		  
		 
	     	  (shape ,(fuse `(cylWide cylThin))
			 ))
	     
	      (declare (type TopoDS_Shape shape))


	      (do0 (BRepLib--BuildCurves3d threadingWire1)
		      (BRepLib--BuildCurves3d threadingWire2)
		      (let ((aTool (BRepOffsetAPI_ThruSections Standard_True)))
			(aTool.AddWire threadingWire1)
			(aTool.AddWire threadingWire2)
			(comments "because they come from ellipses, the splines will be compatible")
			(aTool.CheckCompatibility Standard_False)
			(comments "create thread")
			(let ((myThreading (aTool.Shape)))
			  )))
	      
	      (let ((unify (ShapeUpgrade_UnifySameDomain ,(fuse `(shape myThreading)))))
		(comments "remove unneccessary seams")
		(unify.Build)
		(setf shape
		      (dot unify
			   (Shape))))
	     
	      (return shape)
	     
	      ))
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
	     (let ( (shape (MakeM2ScrewHole))
		    
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


  
