(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar")
  (defparameter *code-file* "run_01_plot_sar_image")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  
  (let* ((code
	 `(do0

	   
	   (do0 "# %% imports"
		(imports (		;os
			  sys
			  time
					;docopt
			  pathlib
			  (np numpy)
			  serial
			  (pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
			  scipy.ndimage
			  scipy.optimize
					;nfft
			  sklearn
			  sklearn.linear_model
			  itertools
			  datetime
			  ))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))
