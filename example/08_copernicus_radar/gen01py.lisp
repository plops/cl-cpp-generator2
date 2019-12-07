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
					      (glob (string "o*.cf"))))
				   :dtype np.complex64
				   :mode (string "r")
				   :shape #+nil (tuple 7000 ; 22778
						       15283 ;; range
						       )
				   (tuple 15000
					  30213)))

		(do0
		 (setf skip 0)
		 ;(setf spart (aref s (slice skip ":") ":"))
		 ;(del s)
		 (setf 
		  ys0
		  #+nil (np.zeros (tuple 1 24223)
			       :dtype np.complex64)
		       
					(np.empty_like s)
		       )
		 
		 #+nil (for (idx 
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
		 
		 #+nil (do0
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
					
		 #+nil (plt.imshow (np.log (+ .001 (np.abs img))))
		 #+nil (plt.plot (np.log (+ .001 (np.abs (* k0 kp))))))
		#-nil (plt.imshow
		       (np.angle s))
		#+nil (plt.imshow
		       (np.log (+ .01 (np.abs  s))))
		#+nil (plt.imshow (np.real s))
		#+nil
		(do0
		 (setf k (np.fft.fft (s.astype np.complex128) :axis 1))
		 (plt.imshow (np.log (+ .001 (np.abs k)))))
		))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))
