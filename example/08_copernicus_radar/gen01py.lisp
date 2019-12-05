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
			  ))
		(setf s (np.memmap (next (dot (pathlib.Path (string "./"))
					      (glob (string "o*.cf"))))
				   :dtype np.complex64
				   :mode (string "r")
				   :shape (tuple 22778
						 15283
						 )))
		#+nil (plt.imshow
		 (np.angle s))
		(do0
		 (setf k (np.fft.fft (s.astype np.complex128) :axis 1))
		 (plt.imshow (np.log (+ .001 (np.abs k)))))
		))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))
