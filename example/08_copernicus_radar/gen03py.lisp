(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar")
  (defparameter *code-file* "run_03_compress")
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
			  scipy.ndimage
					;scipy.optimize
			  scipy.signal
			  numpy.polynomial
			  numba
			  numba.cuda
			  cupy
			  ))
		"# %% echo packet information"
		(setf df (pd.read_csv (string "./o_range.csv")))
		(setf dfa (pd.read_csv (string "./o_all.csv")))
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
		 pol_desc (list ,@(loop for e in `(txh
						   txh_rxh
						   txh_rxv
						   txh_rxvh
						   txv
						   txv_rxh
						   txv_rxv
						   txv_rxvh)
				     collect
				       `(string ,e)))
		 rx_desc (list ,@(loop for e in `(rxv
						  rxh) collect
				      `(string ,e)))
		 signal_type_desc (list ,@(loop for e in `(echo
							   noise
							   na2
							   na3
							   na4
							   na5
							   na6
							   na7
							   tx_cal
							   rx_cal
							   epdn_cal
							   ta_cal
							   apdn_cal
							   na13
							   na14
							   txhiso_cal) collect
				      `(string ,e)))
		 )
		"# %% calibration packet information"
		(setf dfc (pd.read_csv (string "./o_cal_range.csv"))
		 (aref dfc (string "pcc")) (np.mod dfc.cal_iter 2)
		 )
		,@(loop for d in `(dfc df dfa) collect
		       `(do0
			 ,@(loop for e in (case d
					    (dfc `( pol rx signal_type cal_type))
					    (df `( pol rx signal_type))) collect
			    (let ((name (format nil "~a_desc" e)))
			      `(do0
				(setf (aref ,d (string ,name))
				
				      ("list" (map (lambda (x)
						     (aref ,name x)
						     )
						   (dot ,d ,e))))
				(setf (dot ,d ,name) (dot ,d ,name (astype (string "category")))))))))

		
		

		(do0 "# %% sample rate computation using 3.2.5.4, tab5.1-1, tab5.1-2, txpl and swl description"
		     (setf decimation_filter_bandwidth
		       (list 100 87.71 -1 74.25 59.44
			     50.62 44.89 22.2 56.59 42.86
			     15.1 48.35))
		     (setf decimation_filter_L (list 3 2 -1 5 4 3 1 1 3 5 3 4))
		     (setf decimation_filter_M (list 4 3 -1 9 9 8 3 6 7 16 26 11))
		     (setf decimation_filter_length_NF
			   (list 28 28 -1 32 40 48 52  92 36 68 120 44))
		     (setf decimation_filter_output_offset
			   (list 87 87 -1 88 90 92 93 103 89 97 110 91))
		     (setf decimation_filter_swath_desc
			   (list
			    ,@(loop for e in `(full_bandwidth
					       s1_wv1
					       n/a
					       s2
					       s3
					       s4
					       s5
					       ew1
					       iw1
					       s6_iw3
					       ew2_ew3_ew4_ew5
					       iw2_wv2)
				 collect
				   `(string ,e))))
		     "# table 5.1-1 D values as a function of rgdec and C. first index is rgdec"
		     (setf decimation_filter_D
			   (list (list 1 1 2 3)
				 (list 1 1 2)
				 (list -1)
				 (list 1 1 2 2 3 3 4 4 5)
				 (list 0 1 1 2 2 3 3 4 4)
				 (list 0 1 1 1 2 2 3 3)
				 (list 0 0 1)
				 (list 0 0 0 0 0 1)
				 (list 0 1 1 2 2 3 3)
				 (list 0 0 1 1 1 2 2 2 2 3 3 3 4 4 4 5)
				 (list 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 ;; as many as prev col
				       2 2 2 2  2 2 2 2 ; 8 times 2
				       3 3
				       )
				 (list 0 1 1 1 2 2 3 3 3 4 4))))
		
		,@(loop for d in `(dfc df dfa) collect
		       `(do0
			 ,@(loop for e in `(decimation_filter_bandwidth
					    decimation_filter_L
					    decimation_filter_M
					    decimation_filter_length_NF
					    decimation_filter_output_offset
					    decimation_filter_swath_desc)
			      collect
				(let ((name (format nil "~a" e)))
				  `(do0
				    (setf (aref ,d (string ,name))
					  ("list" (map (lambda (x)
							 (aref ,name x))
						       (dot ,d rgdec)))))))))
		(setf fref 37.53472224)
		,@(loop for d in `(dfc df dfa) collect
		       `(do0
			 (setf (aref ,d (string "fdec"))
			       (* 4 fref
				  (/ (dot ,d decimation_filter_L)
				     (dot ,d decimation_filter_M))))
			 (setf (aref ,d (string "N3_tx"))
			       (dot (np.ceil 
				     (* (dot ,d fdec)
					(dot ,d txpl)))
				    (astype np.int)))
			 (setf (aref ,d (string "decimation_filter_B"))
			       (- (* 2 (dot ,d swl))
				  (+ (dot ,d decimation_filter_output_offset) 17)))
			 (setf (aref ,d (string "decimation_filter_C"))
			       (- (dot ,d decimation_filter_B)
				  (* (dot ,d decimation_filter_M)
				     (// (dot ,d decimation_filter_B)
					 (dot ,d decimation_filter_M)))))
			 (setf (aref ,d (string "N3_rx"))
			       ("list"
				(map
				 (lambda (idx_row)
				   (* 2 
				      (+ (* (dot (aref idx_row 1) decimation_filter_L)
					    (// (dot (aref idx_row 1) decimation_filter_B)
						(dot (aref idx_row 1) decimation_filter_M)))
					 (aref (aref decimation_filter_D
						     (dot (aref idx_row 1) rgdec))
					       (dot (aref idx_row 1) decimation_filter_C))
					 1)))
				 (dot ,d (iterrows)))))))


		(do0 "# %% get pulse configuration that is rank pri_counts in the past"
		     ;;  dfap.loc[dfa.pri_count-dfa['rank']].txprr
		     (setf dfap (dot dfa (set_index (string "pri_count")))
			   )
		     ,@(loop for e in `(txprr
					txprr_
					txpl
					txpl_
					txpsf
					ses_ssb_tx_pulse_number)
			  collect
			    `(setf (aref dfa (string ,(format nil "ranked_~a" e)))
				   (dot dfap
					(aref loc
					      (- dfa.pri_count
						 (aref dfa (string "rank"))))
					(reset_index)
					,e))))

		
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
				   :offset (* 4 2 24890 10800)
				   :shape #+nil (tuple 1400 ; 22778
						       15283 ;; range
						       )
				   
				   (tuple 7400
					  24890)))

		
		(setf
		      xs_a_us  (/ (np.arange (aref s.shape 1)) fdec)
		      xs_off (- xs_a_us (* .5 (aref dfc.txpl 0))
				.5)
		      xs_mask (& (< (* -.5 (aref dfc.txpl 0)) xs_off)
				 (< xs_off (* .5 (aref dfc.txpl 0))))
		      arg_nomchirp (* -2 np.pi
				      (+ (* xs_off  (+ (aref dfc.txpsf 0)
						       (* .5
							  (aref dfc.txpl 0)
							  (aref dfc.txprr 0))))
					 (* (** xs_off 2)
					    .5
					    (aref dfc.txprr 0)))))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

