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
	    "from b.cgal_mesher import *"
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
	      (setf triangles
		    ("list"
		     (for-generator (face (cdt.finite_faces))
				    ("tuple"
				     (for-generator (i (range 3))
						    (aref point_to_index_map
							  (dot face
							       (vertex_handle i)
							       point)))))))
	      )
 	   )) 
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

