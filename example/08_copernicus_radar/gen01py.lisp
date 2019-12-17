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
			  scipy.ndimage
					;scipy.optimize
			  scipy.signal
			  numpy.polynomial
					;nfft
			  ;sklearn
			  ;sklearn.linear_model
			  ;itertools
					;datetime
			  ;(et xml.etree.ElementTree)
			  ))
		#+nil (setf xmlfn (string "/home/martin/Downloads/s1a-iw1-slc-vh-20181106t135248-20181106t135313-024468-02aeb9-001.xml"
				    )
			    xm (et.parse xmlfn))
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
		
		,@(loop for d in `(dfc df) collect
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
		,@(loop for d in `(dfc df) collect
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

		#+nil
		(do0 "# plot all packet infomration for echos, calibration and all packets"
		 (do0 "# %% plot calibration time sequence"
		      (plt.figure)
		      ,(let ((l `(sab_ssb_calibration_p
				  sab_ssb_elevation_beam_address
				  sab_ssb_azimuth_beam_address
				  cal_iter
				  ele_count
				  number_of_quads
				  signal_type
				  cal_mode
				  cal_type
				  ses_ssb_cal_mode
				  ses_ssb_signal_type
				  ses_ssb_swath_number
				  ses_ssb_tx_pulse_number
				  ranked_ses_ssb_tx_pulse_number
				  ses_ssb_swap
				  rank
				  ranked_txpl
				  ranked_txpsf
				  ranked_txprr)
			       ))
			 `(do0
			   (setf pl (tuple ,(length l) 1))
			   ,@(loop for e in l and i from 0 collect
				  `(do0
				    (plt.subplot2grid pl (tuple ,i 0))
				    (dot (aref dfa (string ,e)) (plot))
				    (plt.legend)
				    (plt.grid))))))

		 (do0 "# %% plot calibration time sequence only with cal"
		      (setf dfa_cal (dot (aref dfa (== dfa.sab_ssb_calibration_p 1))
					 (set_index (string "cal_iter"))))
		      (plt.figure)
		      ,(let ((l `(sab_ssb_calibration_p
				  sab_ssb_elevation_beam_address
				  sab_ssb_azimuth_beam_address
					; cal_iter
				  number_of_quads
				  signal_type
				  cal_mode
				  cal_type
				  ses_ssb_cal_mode
				  ses_ssb_signal_type
				  ses_ssb_swath_number
				  ses_ssb_tx_pulse_number
				  ranked_ses_ssb_tx_pulse_number
				  ses_ssb_swap
				  rank
				  ranked_txpl
				  ranked_txpsf
				  ranked_txprr)
			       ))
			 `(do0
			   (setf pl (tuple ,(length l) 1))
			   ,@(loop for e in l and i from 0 collect
				  `(do0
				    (plt.subplot2grid pl (tuple ,i 0))
				    (dot (aref dfa_cal (string ,e)) (plot))
				    (plt.legend)
				    (plt.grid)))
			   (plt.xlabel (string "cal_iter")))))

		 
		 (do0 "# %% plot calibration time sequence only with image echos"
		      (setf dfa_img (dot (aref dfa (== dfa.sab_ssb_calibration_p 0))
					 (set_index (string "ele_count"))))
		      (plt.figure)
		      ,(let ((l `(sab_ssb_calibration_p
				  sab_ssb_elevation_beam_address
				  sab_ssb_azimuth_beam_address
					; cal_iter
				  number_of_quads
				  signal_type
				  cal_mode
					;ses_ssb_cal_mode
				  ses_ssb_signal_type
				  ses_ssb_swath_number
				  ranked_ses_ssb_tx_pulse_number
				  ses_ssb_swap
					;cal_type
				  rank
				  ranked_txpl
				  ranked_txpsf
				  ranked_txprr)
			       ))
			 `(do0
			   (setf pl (tuple ,(length l) 1))
			   ,@(loop for e in l and i from 0 collect
				  `(do0
				    (plt.subplot2grid pl (tuple ,i 0))
				    (dot (aref dfa_img (string ,e)) (plot))
				    (plt.legend)
				    (plt.grid)))
			   (plt.xlabel (string "ele_count"))))))
		
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
				   (tuple 1000
					  24890)))
		
		(setf u (dfc.cal_type_desc.unique)
		      un (dfc.number_of_quads.unique))
		,(let ((l `(tx_cal
			   rx_cal
			   epdn_cal
			   ta_cal
			   apdn_cal
			   txh_iso_cal)))
		   `(do0
		     (setf count 0
			   kernel_size 8)
		     ,@(loop for e in l collect
			    `(do0
			      ,@(loop for n below 1 collect
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
		     #+nil(do0
		      (plt.figure)
		      ,@(loop for n below 2 collect
			     `(do0
					;(plt.figure)
			       ,@(loop for e in l collect
				      (let ((name (format nil "~a_~a" e n)))
					`(do0
					  (setf v (scipy.signal.savgol_filter
						   (np.unwrap (np.angle ,(case e
									   (apdn_cal `(np.mean ,name :axis 0))
									   #+nil (txh_iso_cal `(- ,(format nil "txh_cal_~a" n)
												  (np.mean ,name :axis 0)) )
									   (t `(aref ,name count ":")))))
						   ,(case n
						      (0 `(+ 1 (* 2 kernel_size)))
						      (1 `(+ 1 (* 4 kernel_size))))
						   2))
					  (plt.plot (np.linspace 0 (- (len v) 1) (len v))
						    v
						    :label (string ,(format nil "angle <~a>"  (case e
												#+nil (txh_iso_cal `txhcal)
												(t name))))))))
			       (plt.grid)
			       (plt.legend))))

		   #+nil ,@(loop for (code-name code) in `((mag (scipy.signal.savgol_filter
							       (np.abs q)
							       savgol_kernel_size
							       2))
							 (angle (scipy.signal.savgol_filter
								 (np.unwrap (np.angle q))
								 savgol_kernel_size
								 2))) collect
			      `(do0
				(plt.figure)
				,@(loop for n below 2 collect
				       `(do0
					 (dot (plt.gca)
					      (set_prop_cycle None))
					 ,@(loop for e in l collect
						(let ((name (format nil "~a_~a" e n))
						      (sav_name (format nil "sav_~a_~a" e n)))
						  `(do0
						    (setf savgol_kernel_size ,(case n
								    (0 `(+ 1 (* 2 kernel_size)))
								    (1 `(+ 1 (* 4 kernel_size)))))
						    (setf q ,(case e
							       (apdn_cal `(np.mean ,name :axis 0))
							       #+nil (txh_iso_cal `(- ,(format nil "txh_cal_~a" n)
										      (np.mean ,name :axis 0)) )
							       (t `(aref ,name count ":"))))
						    (setf sav_mag_q (scipy.signal.savgol_filter
									(np.abs q)
									savgol_kernel_size
									2)
							  sav_arg_q (scipy.signal.savgol_filter
								     (np.unwrap (np.angle q))
								     savgol_kernel_size
								     2))
						    (setf ,sav_name (* sav_mag_q							     
								       (np.exp (* 1j sav_arg_q))))
						    (setf scale ,(case code-name
								   (mag `1.0)
								   (angle (case n
									    (0 1.0)
									    (1 .5)))))
						    (setf v (* scale ,code))
						    (plt.plot (np.linspace 0 (/ (- (len v) 1)
										,(+ n 1)) (len v))
							      v
							      :linestyle ,(case n
									    (0 `(string "-"))
									    (1 `(string "--")))
							      :label (string ,(format nil (case code-name
											    (mag "|~a|")
											    (angle "arg ~a")) 
										      (case e
											#+nil (txh_iso_cal `txhcal)
											(t name))))))))))
				(plt.grid)
				(plt.legend)))))

		#+nil (do0
		 "# i want to compute rep_vv according to page 36 (detailed alg definition) using savgol filtered pulse"
		 (setf top (* ,@(loop for e in `(sav_tx_cal_0
						 sav_rx_cal_0
						 sav_ta_cal_0)
				   collect
				     `(np.fft.fft ,e)))
		       bot (* ,@(loop for e in `(sav_apdn_cal_0
						 sav_epdn_cal_0)
				   collect
				     `(np.fft.fft ,e))))
		 (do0
		  (plt.figure)
		  (plt.suptitle (string "savgol averaged pulse"))
		  (setf pl (tuple 4 1))
		  (do0
		   (plt.subplot2grid pl (tuple 0 0))
		   (plt.plot (np.abs top) :label (string "top"))
		   (plt.grid)
		   (plt.legend))
		  (do0
		   (plt.subplot2grid pl (tuple 1 0))
		   (plt.plot (np.abs bot) :label (string "bot"))
		   (plt.legend)
		   (plt.grid))
		  (do0
		   (plt.subplot2grid pl (tuple 2 0))
		   (plt.plot (np.abs (/ top bot)) :label (string "top/bot"))
		   (plt.legend)
		   (plt.grid))
		  (do0
		   (plt.subplot2grid pl (tuple 3 0))
		   (plt.plot (np.real (np.fft.ifft (/ top bot))) :label (string "ifft top/bot"))
		   (plt.legend)
		   (plt.grid))))

		
		(do0
		 #+nil
		 (do0 (plt.figure)
		      (plt.suptitle (string "single pulse"))
		      (setf pl (tuple 5 1))
		      ,@(loop for i below 5 collect
			      `(setf ,(format nil "ax~a" i)
				     (plt.subplot2grid pl (tuple ,i 0)))))
		 (do0
		  (setf reps (np.zeros tx_cal_0.shape
				       :dtype np.complex64)))
		 (for (count (range (aref reps.shape 0)))
		      (do0
			(setf top (* ,@(loop for e in `(tx_cal_0
							rx_cal_0
							ta_cal_0)
					  collect
					    `(np.fft.fft (aref ,e count ":")
							 )))
			      bot (* ,@(loop for e in `(apdn_cal_0
							epdn_cal_0)
					  collect
					    `(np.fft.fft ,(case e
							    (apdn_cal_0  `(np.mean ,e :axis 0))
							    (t  `(aref ,e count ":")))))))
			(setf win (np.fft.fftshift (scipy.signal.tukey (aref tx_cal_0.shape 1) :alpha .1)))
			(setf (aref reps count ":") (np.fft.ifft (* win (/ top bot))))))
		 #+nil(for (count (range 3 ))
		      (do0
		       "# i want to compute rep_vv according to page 36 (detailed alg definition)"
		       (do0
			(setf top (* ,@(loop for e in `(tx_cal_0
							rx_cal_0
							ta_cal_0)
					  collect
					    `(np.fft.fft (aref ,e count ":")
							 )))
			      bot (* ,@(loop for e in `(apdn_cal_0
							epdn_cal_0)
					  collect
					    `(np.fft.fft ,(case e
							    (apdn_cal_0  `(np.mean ,e :axis 0))
							    (t  `(aref ,e count ":")))))))
			(setf win (np.fft.fftshift (scipy.signal.tukey (aref tx_cal_0.shape 1) :alpha .1)))
			(setf (aref reps count ":") (np.fft.ifft (* win (/ top bot)))))
		       (do0
			(setf xs (np.fft.fftfreq (len top)))
			(do0
			 (ax0.plot xs (np.abs top) :label (string "top"))
			 (ax0.grid)
			 (ax0.legend))
			(do0
			 (ax1.plot xs (np.abs bot) :label (string "bot"))
			 (ax1.legend)
			 (ax1.grid))
			(do0
			 (ax2.plot xs win :label (string "tukey"))
			 (ax2.legend)
			 (ax2.grid))
			(do0
			 (ax3.plot xs (np.abs (/ top bot)) :label (string "top/bot"))
			 (ax3.plot xs (* win (np.abs (/ top bot))) :label (string "top/bot*win"))
			 (ax3.legend)
			 (ax3.grid))
			(do0
			 (ax4.plot (np.real (np.fft.ifft (/ top bot))) :label (string "ifft top/bot"))
			 (ax4.plot (np.real (np.fft.ifft (* win (/ top bot)))) :label (string "ifft top/bot*win"))
			 (ax4.legend)
			 (ax4.grid))))))
		
		#+nil
		(do0
		 "# page 38, (4-36) compress replicas using the first extracted replica"
		 (setf repsc (np.zeros (tuple
					(- (aref reps.shape 0) 1)
					(aref reps.shape 1))
				       :dtype np.complex64))
		 (for (i (range 1 (aref reps.shape 0)))
		      (setf (aref repsc (- i 1) ":")
			    (np.fft.ifft
			     (*
			      (np.fft.fft (aref reps i))
			      (np.conj
			       (np.fft.fft (aref reps 0))))))))

		#-nil
		(do0
		 "# %% fit polynomial to magnitude "
		 (setf a (np.abs (aref reps 0))
		       
		       th_level .6
		       th (* th_level (np.max a))
		       mask (< th a)
		       start (np.argmax mask)
		       end (- (len mask)
			      (np.argmax (aref mask "::-1")))
		       cut (aref a "start:end")
		       fdec (dot (aref dfc.iloc 0)
				 fdec)
		       start_us (/ start fdec)
		       end_us (/ end fdec))
		 (setf
		  
		  xs_a_us  (/ (np.arange (len a)) fdec)
		  xs (aref xs_a_us "start:end")
		  (ntuple cba cba_diag)
		  (np.polynomial.chebyshev.chebfit xs
						   cut
						   23
						   :full True))

		 
		 (do0 ;; plot magnitude and cheby poly
		  (plt.figure)
		  (setf pl (tuple 2 1))
		  (plt.subplot2grid pl (tuple 0 0))
		  (plt.plot xs_a_us a)
		  (plt.plot xs_a_us (np.real (aref reps 0))
			    :label (string "re reps0"))
		  (setf xs_off (- xs_a_us (* .5 (aref dfc.txpl 0))
				  .5)
			xs_mask (& (< (* -.5 (aref dfc.txpl 0)) xs_off)
				   (< xs_off (* .5 (aref dfc.txpl 0))))
			arg_nomchirp (* -2 np.pi
					(+ (* xs_off (+ (aref dfc.txpsf 0)
							(* .5 (aref dfc.txpl 0) (aref dfc.txprr 0))))
					   (* (** xs_off 2) .5 (aref dfc.txprr 0)))))
		  (do0
		   ,(let ((parm `(xs amp0 amp1 delta_t ph))
			  (parm0 `(0s0
				   0s0
				   0s0
				   0s0))
			  (fun-code
			   `(do0 (setf
				  amp (+ amp0 750s0 (* xs amp1))
				  ;xs (aref xs_a_us "start:end")
				  ;amp (np.zeros (len xs_a_us))
				  ;(aref amp "start:end") (np.polynomial.chebyshev.chebval xs cba)
				  xs_off (- xs (+ delta_t 
							(
							 + (* .5 (aref dfc.txpl 0))
							 .5)))
				  xs_mask (& (< (* -.5 (aref dfc.txpl 0)) xs_off)
					     (< xs_off (+ .5 (* .5 (aref dfc.txpl 0)))))
				  arg_nomchirp (+ (* (/ np.pi 180s0) ph) (* -2 np.pi
							   (+ (* xs_off (+ (aref dfc.txpsf 0)
									   (* .5 (aref dfc.txpl 0) (aref dfc.txprr 0))))
							      (* (** xs_off 2) .5 (aref dfc.txprr 0))))))
				 (setf z (* amp xs_mask (np.exp (* 1j arg_nomchirp)))))))
		      `(do0
			(def fun_nomchirp ,parm
			  
			  ,fun-code
			  (return (np.concatenate
				   (tuple (np.real z)
					  (np.imag z)))))
			(def fun_nomchirp_polar ,parm
			  
			  ,fun-code
			  (return (np.concatenate
				   (tuple (np.abs z)
					  ;arg_nomchirp ;
					  (np.unwrap (np.angle z))
					  ))))
			(def fun_nomchirpz ,parm
			  
			  ,fun-code
			  (return z))
			(def fun_nomchirparg ,parm
			  ,fun-code
			  (return arg_nomchirp))
			(setf
		   p0  (tuple
			,@parm0
			#+nil(+ (* .5 (aref dfc.txpl 0))
						 .5)
		       )
		   (ntuple opt opt2)
		   (scipy.optimize.curve_fit fun_nomchirp_polar
					     xs_a_us
					     (np.concatenate
					      (tuple
					       (np.abs (aref reps 0))
					       (np.unwrap (np.angle (aref reps 0)))))
					     :p0 p0))))

		   )
		  #+nil(plt.plot xs_a_us (* 750 xs_mask (np.real (np.exp (* 1j arg_nomchirp))))
			    :label (string "nomchirp"))
		  (plt.plot xs_a_us (np.real (fun_nomchirpz xs_a_us *opt))
			    :label (string "nomchirp_fit"))
		  (plt.plot xs (np.polynomial.chebyshev.chebval xs cba))
		  (do0 (plt.axvline :x start_us :color (string "r"))
		       (plt.axvline :x end_us :color (string "r")))
		  (plt.xlim (- start_us 10) (+ end_us 10))
		  (plt.xlabel (string "time (us)"))
		  (plt.legend)
		  (plt.subplot2grid pl (tuple 1 0))
		  
		  (plt.plot xs (- cut (np.polynomial.chebyshev.chebval xs cba))
			    :label (string "cheb res")
			    )
		  (plt.plot xs_a_us (np.abs (- (aref reps 0) (fun_nomchirpz xs_a_us *opt)))
			    :label (string "z res"))
		  (plt.legend)

		  (plt.xlim (- start_us 10) (+ end_us 10))
		  (do0 (plt.axvline :x start_us :color (string "r"))
		       (plt.axvline :x end_us :color (string "r")))
		  (plt.xlabel (string "time (us)"))))

		(do0
		 "# %% fit polynomial to phase"
		 
		 (setf a (np.abs (aref reps 0))
		       arg (np.unwrap (np.angle (aref reps 0)))
		       th (* th_level (np.max a))
		       mask (< th a)
		       start (np.argmax mask)
		       end (- (len mask)
			      (np.argmax (aref mask "::-1")))
		       cut (aref arg "start:end")
		       fdec (dot (aref dfc.iloc 0)
				 fdec)
		       start_us (/ start fdec)
		       end_us (/ end fdec))
		 (setf
		  xs_a_us  (/ (np.arange (len a)) fdec)
		  xs (aref xs_a_us "start:end")
		  (ntuple cbarg cbarg_diag)
		  (np.polynomial.chebyshev.chebfit xs
						   cut
						   22
						   :full True))

		 (do0 ;; plot phase and cheby poly
		  (plt.figure)
		  (setf pl (tuple 2 1))
		  (plt.subplot2grid pl (tuple 0 0))
		  (plt.plot arg :label (string "arg_meas"))
		  (plt.plot xs (np.polynomial.chebyshev.chebval xs cbarg) :label (string "arg_cheb"))
		  (plt.plot xs_a_us arg_nomchirp :label (string "arg_nomchirp"))
		  (plt.plot xs_a_us (fun_nomchirparg xs_a_us *opt)
			    :label (string "nomchirparg_fit"))
		  (do0 (plt.axvline :x start_us :color (string "r"))
		       (plt.axvline :x end_us :color (string "r")))
		  (plt.xlim (- start_us 10) (+ end_us 10))
		  (plt.xlabel (string "time (us)"))
		  (plt.legend)
		  (plt.subplot2grid pl (tuple 1 0))
		  (plt.plot xs (- cut (np.polynomial.chebyshev.chebval xs cbarg)))
		  (plt.xlim (- start_us 10) (+ end_us 10))
		  (plt.xlabel (string "time (us)"))
		  (do0 (plt.axvline :x start_us :color (string "r"))
		       (plt.axvline :x end_us :color (string "r")))))

		#+nil
		(do0 "# %% compute phase and amplitude polynomials with image time sampling and convolve with replica"
		     (setf xs (/ (np.arange (aref ss.shape 1))
				 (aref df.fdec 0))
			   amp (np.zeros (len xs))
			   arg (np.zeros (len xs))
			   amp_all (np.polynomial.chebyshev.chebval xs cba)
			   arg_all (np.polynomial.chebyshev.chebval xs cbarg)
			   mask (& (< start_us xs)
				   (< xs end_us ))
			   (aref amp mask) (aref amp_all mask)
			   amp (scipy.ndimage.gaussian_filter1d amp 120.0)
			   arg arg_all ;(aref arg mask) (aref arg_all mask)
			   repim (* amp (np.exp (* 1j arg))))
		     #+nil (setf krepim (np.fft.fft repim)
			   kss (np.fft.fft ss :axis 1)
			   rcomp (np.fft.ifft (* kss (np.conj krepim)))))
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

