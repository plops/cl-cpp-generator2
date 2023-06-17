(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/126_quant_pow/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)

  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<> armadillo
		iostream
		memory
		SFML/Graphics.hpp)

     (defun compute_psi ()
       (declare (values "arma::vec"))
       (comments "N .. Number of discretization points"
		 "L .. Size of the box"
		 "dx .. Grid spacing")
       (let ((N 1000)
	     (L 1d0)
	     (dx (/ L (+ N 1)))
	     #+nil (main_diag (*  (arma--ones<arma--vec> N)
				  (/ 2d0 (* dx dx))))
	     #+nil  (off_diag (*  (arma--ones<arma--vec> (- N 1))
				  (/ -1d0 (* dx dx))))
	     (H (arma--sp_mat N N)
		#+nil (arma--spdiagmat (curly off_diag
					      main_diag
					      off_diag)
				       (curly -1 0 1))))
	 (dotimes (i N)
	   (when (< 0 i)
	     (comments "subdiagonal")
	     (setf (H i (- i 1))

		   (/ -1d0
		      (* dx dx))))
	   (comments "main diagonal")
	   (setf (H i i)
		 (/ 2d0
		    (* dx dx)))
	   
	   (when (< i (- N 1))
	     (comments "superdiagonal")
	     (setf (H i (+ i 1))
		   (/ -1d0
		      (* dx dx)))))
	 (comments "Initialize a random vector")
	 (let ((psi ("arma::randu<arma::vec>" N)))
	   #+nil (dotimes (iter 10000)
		   (setf psi (* H psi))
		   (/= psi (arma--norm psi)))
	   (comments "Normalize psi")
	   (/= psi (arma--norm psi))
	   (let ((energy (arma--vec))
		 ;; smallest magnitude
		 (status (arma--eigs_sym energy psi H 1 (string "sm"))))
	     (when (== false status)
	       (<< std--cout
		   (string "Eigensolver failed.")
		   energy
		   std--endl)
	       )
	     (<< std--cout
		 (string "Ground state energy: ")
		 (energy 0)
		 std--endl))))
       (return psi))
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"
       (let ((psi (compute_psi))
	     (win (std--make_unique<sf--RenderWindow>
		   (sf--VideoMode 800 600)
		   (string "psi plot")))
	     (plot (sf--VertexArray sf--LinesStrip
				    psi.n_elem)))
	 (dotimes (i psi.n_elem)
	   (let ((x (/ (float i)
		       (* (- psi.n_elem 1)
			  (-> win (dot (getSize) x)))))
		 (y (* (- 1s0
			  (std--abs (psi i)))
		       (-> win (dot (getSize) y)))))
	     (setf (dot (aref plot i)
			position)
		   (sf--Vector2f x y))))
	 (while (win->isOpen)
		(let ((event (sf--Event)))
		  (while (win->pollEvent event)
			 (when (== sf--Event--Closed
				   event.type)
			   (win->close)))
		  (win->clear)
		  (win->draw plot)
		  (win->display)
		  )))
       
       
       
       (return EXIT_SUCCESS)))
   :format t
   :tidy t))


