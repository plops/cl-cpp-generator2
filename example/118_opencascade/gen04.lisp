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
	      (b-rep-builder-a-p-i make-edge make-polygon make-face make-wire transform)
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
	 
	  (comments "https://en.wikipedia.org/wiki/ISO_metric_screw_thread"
		    "https://dev.opencascade.org/doc/overview/html/occt__tutorial.html")
	  (defun MakeHolder ()
	    (declare (type "const Standard_Real" )
		     (values TopoDS_Shape))
	   
	    (let ((axis (gp_Ax2 (gp_Pnt 0 0 0)
				(gp_Dir 0 0 1)))
		  (thick (- 5.0 .01))
		  (adapterRad (* .5 (+ 29.49 .05)))
		  (centralCylOut (BRepPrimAPI_MakeCylinder axis (+ adapterRad 5) thick))
		  (centralCylIn (BRepPrimAPI_MakeCylinder axis adapterRad thick))
		  (centralCylClearance ,(translate `(:z thick :code (BRepPrimAPI_MakeCylinder axis (* .5 (+ 30.04 .2)) 20))))
		  (motorRadBottom (* .5 (+ 27.94 .04)))
		  (motorRadMid (* .5 (+ 28.62 .04)))
		  (leftMotorShiftX -31)
		  (leftMotorHoleBottom ,(translate `(:x leftMotorShiftX :code (BRepPrimAPI_MakeCylinder axis motorRadBottom thick))))
		  (leftMotorHoleMid ,(translate `(:x leftMotorShiftX
						  :z 1.4
						  :code (BRepPrimAPI_MakeCylinder axis motorRadMid 20))))
		  (leftMotorBlockMid ,(translate `(:x -55
						      :y -20
						   :z 1.4
						  :code (BRepPrimAPI_MakeBox 20 40 12))))
		  (leftMotorWall ,(translate `(:x leftMotorShiftX :code (BRepPrimAPI_MakeCylinder axis (+ 3 motorRadMid)  10))))
		  (leftPostHeight  (- 19.32 .83 .04))
		  (leftScrewPostNorth
		    ,(translate `(:x leftMotorShiftX
				  :y (/ 35 2)
				  :code (BRepPrimAPI_MakeCylinder axis 3.5 leftPostHeight))))
		  (leftScrewPostHoleNorth
		    ,(translate `(:x leftMotorShiftX
				  :y (/ 35 2)
				  :code (BRepPrimAPI_MakeCylinder axis (* .5 2.93) leftPostHeight))))
		  (leftScrewPostSouth  ,(translate `(:x leftMotorShiftX
						     :y (/ -35 2)
						     :code (BRepPrimAPI_MakeCylinder axis 3.5  leftPostHeight))))
		  (leftScrewPostHoleSouth  ,(translate `(:x leftMotorShiftX
						     :y (/ -35 2)
						     :code (BRepPrimAPI_MakeCylinder axis (* .5 2.93)  leftPostHeight))))

		  (rightMotorShiftX (- leftMotorShiftX))
		  (rightMotorWall ,(translate `(:x rightMotorShiftX :code (BRepPrimAPI_MakeCylinder axis (+ 3 motorRadMid) 5))))
		  
		  (rightPostHeight (+ 9.42 7.9 8))
		  (rightMotorHoleMid ,(translate `(:x rightMotorShiftX
						   :z 1.4
						   :code (BRepPrimAPI_MakeCylinder axis motorRadMid (- rightPostHeight 1.4))
						   )))
		  
		  (rightScrewPostNorth ,(translate `(:x rightMotorShiftX
						     :y (/ 35 2)
						     :code (BRepPrimAPI_MakeCylinder axis 3.5 rightPostHeight))))
		  (rightScrewPostSouth ,(translate `(:x rightMotorShiftX
						     :y (/ -35 2)
						     :code (BRepPrimAPI_MakeCylinder axis 3.5 rightPostHeight))))
		  (rightScrewPostHoleNorth ,(translate `(:x rightMotorShiftX
							 :y (/ 35 2)
							 :code (BRepPrimAPI_MakeCylinder axis (* .5 2.93) rightPostHeight))))
		  (rightScrewPostHoleSouth ,(translate `(:x rightMotorShiftX
							 :y (/ -35 2)
							 :code (BRepPrimAPI_MakeCylinder axis (* .5 2.93) rightPostHeight))))
		  (rightMotorHoleMidFill ,(translate `(
						       :y 40
						       
						       :code ,(cut `((BRepPrimAPI_MakeCylinder axis (- motorRadMid .1)
											       (- rightPostHeight 1.4))
								     (BRepPrimAPI_MakeCylinder axis (- motorRadMid .1 2)
											       (- rightPostHeight 1.4)
											       )))
						       )))
		  
		 
		  #+nil  (cylShaft ,(translate `(:z flatLength
						 :code (BRepPrimAPI_MakeCylinder axis shaftDiameter/2 (- shaftLength flatLength)))))
		 
		 #+nil (shaftFlattening ,(translate `(:x -flatThickness/2
						 :y -centralDiameter/2
						 :z 0
						 :code (BRepPrimAPI_MakeBox flatThickness centralDiameter flatLength))))
		 #+nil (cylShaft2 ,(common `(cylShaftFullLength
					shaftFlattening))
			     )	     
		 
		  
	     	 (shape ,(cut `(,(fuse `(leftMotorWall ,(cut `(leftScrewPostNorth leftScrewPostHoleNorth))
							,(cut `(leftScrewPostSouth leftScrewPostHoleSouth))
							,(cut `(rightScrewPostNorth rightScrewPostHoleNorth))
							,(cut `(rightScrewPostSouth rightScrewPostHoleSouth))
							rightMotorWall
							centralCylOut
						       rightMotorHoleMidFill
							))
				 ,(fuse `(leftMotorHoleBottom
					  leftMotorHoleMid
					  leftMotorBlockMid
					  rightMotorHoleMid
					  centralCylIn
					  centralCylClearance))))
			 )
		  
		 		  
		  
		  
		  )
	      
	     
	      (declare (type TopoDS_Shape shape))

	      
	      
	      
	      
	      (let ((unify (ShapeUpgrade_UnifySameDomain shape )))
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
	     (let ( (shape (MakeHolder))
		    
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


  
