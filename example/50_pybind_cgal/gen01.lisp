(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-cpp-generator2/example/50_pybind_cgal/")
  (defparameter *code-file* "run_00")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *host*
    "10.1.99.12")
  (defparameter *inspection-facts*
    `((10 "")))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
  (let* (
	 
	 (code
	   `(do0
	     (do0
		  
		  (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		  (imports ((plt matplotlib.pyplot)
			    (animation matplotlib.animation) 
                            ;(xrp xarray.plot)
			    ))
                  
		  (plt.ion)
					;(plt.ioff)
		  ;;(setf font (dict ((string size) (string 6))))
		  ;; (matplotlib.rc (string "font") **font)
		  )

	     (imports ((np numpy)))
	     
	    "from b.cgal_mesher import *"

	     (setf
	       _code_git_version
		  (string ,(let ((str (with-output-to-string (s)
					(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			     (subseq str 0 (1- (length str)))))
		  _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py")
					   )

		  _code_generation_time
		  (string ,(multiple-value-bind
				 (second minute hour date month year day-of-week dst-p tz)
			       (get-decoded-time)
			     (declare (ignorable dst-p))
		      (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
			      hour
			      minute
			      second
			      (nth day-of-week *day-names*)
			      year
			      month
			      date
			      (- tz)))))

	    (setf cdt (ConstrainedDelaunayTriangulation))
	     ,(let ((l `((a 100 269)
		    (b 246 269)
		    (c 246 223)
		    (d 303 223)
		    (e 303 298)
		    (f 246 298)
		    (g 246 338)
		    (h 355 338)
		    (i 355 519)
		    (j 551 519)
		    (k 551 445)
		    (l 463 445)
		    (m 463 377)
		    (n 708 377)
		    (o 708 229)
		    (p 435 229)
		    (q 435 100)
		    (r 100 100)

		    (s 349 236)
		    (t 370 236)
		    (u 370 192)
		    (v 403 192)
		    (w 403 158)
		    (x 349 158)

		    (y 501 336)
		    (z 533 336)
		    (1 519 307)
		    (2 484 307)

		    ))
	       (l2 `((a b)
		     (b c)
		     (c d)
		     (d e)
		     (e f)
		     (f g)
		     (g h)
		     (h i)
		     (i j)
		     (j k)
		     (k l)
		     (l m)
		     (m n)
		     (n o)
		     (o p)
		     (p q)
		     (q r)
		     (r a)

		     (s t)
		     (t u)
		     (u v)
		     (v w)
		     (w x)
		     (x s)

		     (y z)
		     (z 1)
		     (1 2)
		     (2 y))))
		`(do0
		  ,@(loop for (name e f) in l
			 collect
			 `(setf ,(format nil "v~a" name)
				(cdt.insert (Point ,e ,f))))
		  ,@(loop for (e f) in l2
			 collect
			  `(cdt.insert_constraint
			    ,(format nil "v~a" e)
			    ,(format nil "v~a" f)))))

	     (print (dot (string "number of vertices: {}")
			 (format (cdt.number_of_vertices))))
	     (do0
	      (setf mesher (Mesher cdt)
		    seeds (tuple (Point 505 325)
				 (Point 379 172)))
	      (mesher.seeds_from seeds))

	     (do0
	      (make_conforming_delaunay cdt)
	       (print (dot (string "number of vertices: {}")
			   (format (cdt.number_of_vertices)))))
	     (do0
	      (make_conforming_gabriel cdt)
	       (print (dot (string "number of vertices: {}")
			   (format (cdt.number_of_vertices)))))

	     (do0
	      (setf mesher.criteria.aspect_bound 0.125
		    mesher.criteria.size_bound 30.0)
	      (mesher.refine_mesh)
	       (print (dot (string "number of vertices: {}")
			   (format (cdt.number_of_vertices))))
	       )

	      (do0
	      (lloyd_optimize cdt :max_iteration_number 10)
	       (print (dot (string "number of vertices: {}")
			   (format (cdt.number_of_vertices))))
	       )
	      (print_faces_iterator_value_type)
	      ;(print ("list" (cdt.finite_vertices)))
	      ;(print ("list" (cdt.finite_faces)))
	      (setf point_to_index_map
		    (curly
		     (for-generator ((ntuple idx vertex)
				     (enumerate (cdt.finite_vertices)))
				    ,(format nil "~a: ~a" "vertex.point"
					     "idx"))))
	      (setf triangles_idx
		    ("list"
		     (for-generator (face (cdt.finite_faces))
				    ("tuple"
				     (for-generator (i (range 3))
						    (aref point_to_index_map
							  (dot face
							       (vertex_handle i)
							       point)))))))
	      (setf triangles
		    (np.array
		     ("list"
		      (for-generator
		       (face (cdt.finite_faces))
		       ("tuple"
			(for-generator (i (range 3))
				       (tuple (dot (dot face
							(vertex_handle i)
							point)
						   x)
					      (dot (dot face
							(vertex_handle i)
							point)
						   y))))))))

	      (do0
	       (plt.figure)
	       (setf g (plt.gca))
	       (for (i (range (aref triangles.shape 0)))
		    (setf tri (plt.Polygon (aref triangles i ":" ":")
					   :facecolor None
					   :edgecolor (string "k")))
		    (g.add_patch tri))
	       (plt.show)
	       (plt.grid)
	       (plt.xlim (tuple 0 800)
			   )
	       (plt.ylim (tuple 0 600)))
	      )
 	   )) 
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

