(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar")
  (defparameter *code-file* "run_01_plot_sar_image")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  
  (let* ((code
	 `(do0
	   #-nil(do0
                  (imports (matplotlib))
                  ;(matplotlib.use (string "Agg"))
                  (imports ((plt matplotlib.pyplot)))
                  (plt.ion)
                  (setf font (dict ((string size) (string 5))))
                  (matplotlib.rc (string "font") **font)
                  )

	   
	   (do0 "# %% imports"
		(imports (		;os
			  sys
			  time
					;docopt
			  pathlib
			  (np numpy)
			  numpy.fft
			  ;serial
			  (pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
			  ;scipy.ndimage
					;scipy.optimize
			  scipy.signal
					;nfft
			  ;sklearn
			  ;sklearn.linear_model
			  ;itertools
					;datetime
			  (et xml.etree.ElementTree)
			  ))
		#+nil (setf xmlfn (string "/home/martin/Downloads/s1a-iw1-slc-vh-20181106t135248-20181106t135313-024468-02aeb9-001.xml"
				    )
		      xm (et.parse xmlfn))
		(setf df (pd.read_csv (string "./o_range.csv")))
		
		(setf
		 cal_type_desc (list ,@(loop for e in `(tx_cal
							rx_cal
							epdn_cal
							ta_cal
							apdn_cal
							na_0
							na_1
							txh_iso_cal)
					  collect
					    `(string ,e)))
		 dfc (pd.read_csv (string "./o_cal_range.csv"))
		 (aref dfc (string "cal_type_desc")) 
		 ("list" (map (lambda (x)
				(aref cal_type_desc x))
			      dfc.cal_type))
		 dfc.cal_type_desc (dfc.cal_type_desc.astype (string "category"))
		 (aref dfc (string "pcc"))
		 (np.mod dfc.cal_iter 2)
		 )
		#+nil (setf fref 37.53472224
		      
		      row (aref df.iloc 0)
		      txprr row.txprr
		      txprr_ row.txprr_
		      txpsf row.txpsf
		      txpl row.txpl
		      txpl_ row.txpl_
		      ns (np.arange txpl_) #+nil (- 
			    (/ txpl_ 2))
		      xs (/ ns fref)
		      arg (+ (* txpsf xs)
						(* .5 txprr xs xs))
		      ys (np.exp (* -2j np.pi arg))
		      )
		#+nil
		(plt.plot xs ys)
		#+nil(do0
		 (setf kp (np.fft.fft (ys.astype np.complex128)))
		 (plt.plot (np.log (+ .001 (np.abs kp)))))
		(setf s (np.memmap (next (dot (pathlib.Path (string "./"))
					      (glob (string "o_cal*.cf"))))
				   :dtype np.complex64
				   :mode (string "r")
				   :shape #+nil (tuple 7000 ; 22778
						       15283 ;; range
						       )
				   #+nil
				    (tuple 16516
					   24695)
				    (tuple 800 6000)))
		(setf ss (np.memmap (next (dot (pathlib.Path (string "./"))
					      (glob (string "o_r*.cf"))))
				   :dtype np.complex64
				   :mode (string "r")
				   :shape #+nil (tuple 7000 ; 22778
						       15283 ;; range
						       )
				   
				    (tuple 16516
					   24695)
				    ))

		(setf u (dfc.cal_type_desc.unique)
		      un (dfc.number_of_quads.unique))
		,(let ((l `(tx_cal
			   rx_cal
			   epdn_cal
			   ta_cal
			   apdn_cal
			   txh_iso_cal)))
		   `(do0
		     ,@(loop for e in l collect
			    `(do0
			      
			      ,@(loop for n below 2 collect
				     (let ((name (format nil "~a_~a" e n)))
				      `(do0
					(setf sub (aref dfc (& (== dfc.cal_type_desc (string ,e))
							       (== dfc.number_of_quads (aref un ,n))
							       ,(if (member e `(apdn_cal txh_iso_cal))
								    `True
								    `(== dfc.pcc 0)))))
					(setf ,name (np.zeros (tuple (len sub)
								     6000)
							      :dtype np.complex64))
					(setf j 0)
					(for (i sub.cal_iter)
					     (setf (aref ,name j ":")
						   ,(if (member e `(apdn_cal txh_iso_cal))
							`(aref s i ":")
							`(* .5 (- (aref s i ":")
								  (aref s (+ i 1) ":")))))
					     (setf j (+ j 1))))))))
		     ,@(loop for e in l collect
			    `(do0
			      ,@(loop for n below 2 collect
				     (let ((name (format nil "~a_~a" e n)))
				       `(do0
					 (plt.plot (np.unwrap (np.angle (np.mean ,name :axis 0)))))))))))
		
		
		#+nil (do0
		 (setf fref 37.53472224)
		 (setf  input (* .5 (- (aref s 1 ":3000")
				       (aref s 0 ":3000")))
			xs (/ (np.arange (len input))
			      fref))
		 (do0
		  (plt.figure)
		  (setf chirp_phase (np.unwrap (np.angle input)))
		  (setf dfchirp (dot (pd.DataFrame (dict ((string "xs") xs)
							 ((string "mag") (np.abs input))))
				     (set_index (string "xs")))
			dfchirpon (< 150 dfchirp)
			chirp_start (dot (aref dfchirpon (== dfchirpon.mag True))
					 (aref iloc 0)
					 name)
			chirp_end (dot (aref dfchirpon (== dfchirpon.mag True))
				       (aref iloc -1)
				       name))
		  (setf chirp_poly (np.polynomial.polynomial.Polynomial.fit xs chirp_phase 2
									    :domain (list chirp_start
											  chirp_end)))
		  (plt.plot xs (np.unwrap (np.angle input)))
		  (plt.plot xs (chirp_poly xs)))


		 (do0
		  (plt.figure)
		  (plt.plot xs (np.real input))
		  (plt.plot xs (np.real (* 175 (np.exp (* 1j (chirp_poly xs)))))))
		 
		 #+nil (do0 
		  (plt.figure)
		  
		  (plt.plot xs (np.real input)
			    :label (string "real"))
		  (plt.plot xs (np.imag input)
			    :label (string "imag"))
		  (setf 
		   row (aref dfc.iloc 0)
		   txprr row.txprr
		   txprr_ row.txprr_
		   txpsf row.txpsf
		   txpl row.txpl
		   txpl_ row.txpl_
		   steps (+ -50 (np.linspace 0 3000 3001))
		   tn #+nil (np.arange (* -1 (// txpl_ 2))
				       (- (// txpl_ 2)
					  1))
		   (* steps (/ 1s0 fref))
		   #+nil (np.linspace (* -.5 txpl)
				      (* .5 txpl)
				      (* 2 row.number_of_quads))
		   p1 (- txpsf (* txprr -.5 txpl))
		   p2 (* .5 txprr)
		   arg (+ (* p1 tn)
			  (* p2 tn tn))
		   ys (* 175 (np.exp (* -2j np.pi arg)))

		   ))

		 #+nil (do0
		  (def chirp (tn amp p1 p2 xoffset xscale)
		    (setf tns (* xscale tn)
			  tnso (- tns xoffset))
		    (setf arg (+ (* p1 tnso)
				 (* p2 tnso tnso)))
		    (setf z (* amp (np.exp (* -2j np.pi arg))))
		    (return (np.concatenate
			     (tuple (np.real z)
				    (np.imag z)))))

		  (setf
		   p0  (tuple
			175s0
			(- txpsf (* txprr -.5 txpl))
			(* .5 txprr)
			0s0
			1s0)
		   (ntuple opt opt2)
		   
		   (scipy.optimize.curve_fit chirp
					     (/ (np.arange (len input))
						fref)
					     (np.concatenate
					      (tuple
					       (np.real input)
					       (np.imag input)))
					     :p0 p0)))
		 #+nil (do0
		  
		  (plt.plot xs
			    (aref (chirp xs *p0) ":3000")
			    :label (string "init_re"))
		  (plt.plot xs
			    (aref (chirp xs *opt) ":3000")
			    :label (string "fit_re"))
					;(plt.plot xs (np.real ys) :label (string "analytic_re"))
					;(plt.plot xs (np.imag ys) :label (string "analytic_im"))
		  (plt.legend)))
		#+nil (do0
		 (setf a2 (scipy.signal.decimate (np.abs (aref s "8000:" ":")) 10)
		       a3 (scipy.signal.decimate (np.abs a2) 10 :axis 0))
		 (del a2))
		#+nil (do0
		 (setf skip 0)
		 ;(setf spart (aref s (slice skip ":") ":"))
		 ;(del s)
		 (setf 
		  ys0
		  #+nil (np.zeros (tuple 1 24223)
			       :dtype np.complex64)
		       
					(np.empty_like s)
		       )
		 
		 #-nil (for (idx 
			   (range (aref ys0.shape 0))
			   )
		  (setf fref 37.53472224
			row (aref df.iloc (+ skip    idx))
			txprr row.txprr
			txprr_ row.txprr_
			txpsf row.txpsf
			txpl row.txpl
			txpl_ row.txpl_
			ns (np.arange txpl_) #+nil (- 
						    (/ txpl_ 2))
			xs (/ ns fref)
			arg (+ (* txpsf xs)
			       (* .5 txprr xs xs))
			ys (np.exp (* -2j np.pi arg))
			(aref ys0 idx (slice 0 (len ns))) ys))
		 
		 #-nil (do0
		  (setf k0 (np.fft.fft s
				       :axis 1))
		  (setf kp (np.fft.fft ys0 :axis 1
				       ))
		  (setf a (* k0 kp))
		  (del kp)
		  (del k0)
		  (setf img (np.fft.ifft a))
		  (del a)
		  )
					
		 
		 #+nil (plt.plot (np.log (+ .001 (np.abs (* k0 kp))))))
		#+nil (plt.imshow
		       (np.angle s))
		#+nil(do0
		      
		      (setf fig (plt.figure)
			    ax (fig.add_subplot (string "111")))
		      #+nil (ax.imshow (np.log (+ .001 (np.abs img))))
		      #-nil (ax.imshow
			     #+nil(np.log (+ .01 (np.abs a3)))
			     #+nil
			     (np.real s)
			     (np.angle

			       s)
		       #+nil (np.log (+ .01 (np.abs  s)
					     )))
		      (ax.set_aspect (string "auto")))
		#+nil (plt.imshow (np.real s))
		#+nil
		(do0
		 (setf k (np.fft.fft (s.astype np.complex128) :axis 1))
		 (plt.imshow (np.log (+ .001 (np.abs k)))))
		))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

