(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "alexandria")
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-cpp-generator2")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-cpp-generator2/tree/master/")
  (defparameter *example-subdir* "example/65_filament")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *libs*
    `((np numpy)
      (pd pandas)
      ;(xr xarray)
      matplotlib
      (s skyfield)
					;(ds dataset)
      cv2
      ))
  (let ((nb-file "source/01_convert.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp convert01"))
      (python (do0
	       
	       "#export"
	       (do0
					;"%matplotlib notebook"
		(do0
		      
		      (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		      (imports ((plt matplotlib.pyplot)
					;  (animation matplotlib.animation) 
					;(xrp xarray.plot)
				))
                  
		      (plt.ion)
					;(plt.ioff)
		      ;;(setf font (dict ((string size) (string 6))))
		      ;; (matplotlib.rc (string "font") **font)
		      )
		(imports (		;os
					;sys
			  time
					;docopt
					;pathlib
					;(np numpy)
					;serial
					;(pd pandas)
					;(xr xarray)
			  ,@*libs*
			  (xrp xarray.plot)
			  skimage.restoration
			  skimage.morphology
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
					;scipy.optimize
					;scipy.stats
					;scipy.special
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
					; (np numpy)
					;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
					; copy
					;re
					;json
					; csv
					;io.StringIO
					;bs4
					;requests
					;(nx networkx)
					;(np jax.numpy)
					;(mpf mplfinance)
					argparse
					;(sns seaborn)
					; skyfield.api
					;skyfield.data
					; skyfield.data.hipparcos
			  ))
		
		"from cv2 import *"
	      		(imports-from (matplotlib.pyplot
			       plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))
		 
		)
	       ))
      (python
       (do0
	"#export"
	;(sns.set_theme)
	(setf
	 _code_git_version
	 (string ,(let ((str (with-output-to-string (s)
			       (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
		    (subseq str 0 (1- (length str)))))
	 _code_repository (string ,(format nil "~a/~a/~a" *repo-dir-on-github* *example-subdir* nb-file))
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

	(setf start_time (time.time)
	      debug True)))
      (python
       (do0
	"#export"
	(setf df_status
	      (pd.DataFrame
	       (list
		,@(loop for e in *libs*
			collect
			(cond
			  ((listp e)
			   (destructuring-bind (nick name) e
			     `(dictionary
			       :name (string ,name)
			       :version
			       (dot ,nick __version__)
			       )))
			  ((symbolp e)
			   `(dictionary
			       :name (string ,e)
			       :version
			       (dot ,e __version__)
			       ))
			  (t (break "problem")))))))
	(print df_status)))

      (python
       (do0
	(class ArgsStub
	       ()
	       (setf filename  (string "/home/martin/ISS Timelapse - Stars Above The World (29 _ 30 Marzo 2017)-8fCLTeY7tQg.mp4.part")
		     threshold 30
		     skip_frames 0
		     decimate_frames 1))))
      (python
       (do0
	"#export"
	(setf parser (argparse.ArgumentParser))
	(parser.add_argument (string "-i")
			     :dest (string "filename")
			     :required True
			     :default (string "/home/martin/stage/cl-py-generator/example/70_star_tracker/source/hip_main.dat")
			     :help (string "input file")
			     :metavar (string "FILE"))
	(setf args (parser.parse_args))
	(print args)
	))
      (markdown "https://heasarc.gsfc.nasa.gov/W3Browse/all/hipparcos.html
https://github.com/skyfielders/python-skyfield/blob/master/skyfield/data/hipparcos.py

```
;; H0          Catalog    * Not Displayed *  /Catalogue (H=Hipparcos)
;; H1          HIP        HIP_Number         /Identifier (HIP number)
;; H2          Proxy      Prox_10asec        /Proximity flag
;; H3          RAhms      RA                 /RA in h m s, ICRS (J1991.25)
;; H4          DEdms      Dec                /Dec in deg ' ", ICRS (J1991.25)
;; H5          Vmag       Vmag               /Magnitude in Johnson V
;; H6          VarFlag    Var_Flag           /Coarse variability flag
;; H7          r_Vmag     Vmag_Source        /Source of magnitude
;; H8          RAdeg      RA_Deg             /RA in degrees (ICRS, Epoch-J1991.25)
;; H9          DEdeg      Dec_Deg            /Dec in degrees (ICRS, Epoch-J1991.25)
;; H10         AstroRef   Astrom_Ref_Dbl     /Reference flag for astrometry
;; H11         Plx        Parallax           /Trigonometric parallax
;; H12         pmRA       pm_RA              /Proper motion in RA
;; H13         pmDE       pm_Dec             /Proper motion in Dec
;; H14         e_RAdeg    RA_Error           /Standard error in RA*cos(Dec_Deg)
;; H15         e_DEdeg    Dec_Error          /Standard error in Dec_Deg
;; H16         e_Plx      Parallax_Error     /Standard error in Parallax
;; H17         e_pmRA     pm_RA_Error        /Standard error in pmRA
;; H18         e_pmDE     pm_Dec_Error       /Standard error in pmDE
;; H19         DE:RA      Crl_Dec_RA         /(DE over RA)xCos(delta)
;; H20         Plx:RA     Crl_Plx_RA         /(Plx over RA)xCos(delta)
;; H21         Plx:DE     Crl_Plx_Dec        /(Plx over DE)
;; H22         pmRA:RA    Crl_pmRA_RA        /(pmRA over RA)xCos(delta)
;; H23         pmRA:DE    Crl_pmRA_Dec       /(pmRA over DE)
;; H24         pmRA:Plx   Crl_pmRA_Plx       /(pmRA over Plx)
;; H25         pmDE:RA    Crl_pmDec_RA       /(pmDE over RA)xCos(delta)
;; H26         pmDE:DE    Crl_pmDec_Dec      /(pmDE over DE)
;; H27         pmDE:Plx   Crl_pmDec_Plx      /(pmDE over Plx)
;; H28         pmDE:pmRA  Crl_pmDec_pmRA     /(pmDE over pmRA)
;; H29         F1         Reject_Percent     /Percentage of rejected data
;; H30         F2         Quality_Fit        /Goodness-of-fit parameter
;; H31         ---        * Not Displayed *  /HIP number (repetition)
;; H32         BTmag      BT_Mag             /Mean BT magnitude
;; H33         e_BTmag    BT_Mag_Error       /Standard error on BTmag
;; H34         VTmag      VT_Mag             /Mean VT magnitude
;; H35         e_VTmag    VT_Mag_Error       /Standard error on VTmag
;; H36         m_BTmag    BT_Mag_Ref_Dbl     /Reference flag for BT and VTmag
;; H37         B-V        BV_Color           /Johnson BV colour
;; H38         e_B-V      BV_Color_Error     /Standard error on BV
;; H39         r_B-V      BV_Mag_Source      /Source of BV from Ground or Tycho
;; H40         V-I        VI_Color           /Colour index in Cousins' system
;; H41         e_V-I      VI_Color_Error     /Standard error on VI
;; H42         r_V-I      VI_Color_Source    /Source of VI
;; H43         CombMag    Mag_Ref_Dbl        /Flag for combined Vmag, BV, VI
;; H44         Hpmag      Hip_Mag            /Median magnitude in Hipparcos system
;; H45         e_Hpmag    Hip_Mag_Error      /Standard error on Hpmag
;; H46         Hpscat     Scat_Hip_Mag       /Scatter of Hpmag
;; H47         o_Hpmag    N_Obs_Hip_Mag      /Number of observations for Hpmag
;; H48         m_Hpmag    Hip_Mag_Ref_Dbl    /Reference flag for Hpmag
;; H49         Hpmax      Hip_Mag_Max        /Hpmag at maximum (5th percentile)
;; H50         HPmin      Hip_Mag_Min        /Hpmag at minimum (95th percentile)
;; H51         Period     Var_Period         /Variability period (days)
;; H52         HvarType   Hip_Var_Type       /Variability type
;; H53         moreVar    Var_Data_Annex     /Additional data about variability
;; H54         morePhoto  Var_Curv_Annex     /Light curve Annex
;; H55         CCDM       CCDM_Id            /CCDM identifier
;; H56         n_CCDM     CCDM_History       /Historical status flag
;; H57         Nsys       CCDM_N_Entries     /Number of entries with same CCDM
;; H58         Ncomp      CCDM_N_Comp        /Number of components in this entry
;; H59         MultFlag   Dbl_Mult_Annex     /Double and or Multiple Systems flag
;; H60         Source     Astrom_Mult_Source /Astrometric source flag
;; H61         Qual       Dbl_Soln_Qual      /Solution quality flag
;; H62         m_HIP      Dbl_Ref_ID         /Component identifiers
;; H63         theta      Dbl_Theta          /Position angle between components
;; H64         rho        Dbl_Rho            /Angular separation of components
;; H65         e_rho      Rho_Error          /Standard error of rho
;; H66         dHp        Diff_Hip_Mag       /Magnitude difference of components
;; H67         e_dHp      dHip_Mag_Error     /Standard error in dHp
;; H68         Survey     Survey_Star        /Flag indicating a Survey Star
;; H69         Chart      ID_Chart           /Identification Chart
;; H70         Notes      Notes              /Existence of notes
;; H71         HD         HD_Id              /HD number <III 135>
;; H72         BD         BD_Id              /Bonner DM <I 119>, <I 122>
;; H73         CoD        CoD_Id             /Cordoba Durchmusterung (DM) <I 114>
;; H74         CPD        CPD_Id             /Cape Photographic DM <I 108>
;; H75         (V-I)red   VI_Color_Reduct    /VI used for reductions
;; H76         SpType     Spect_Type         /Spectral type
;; H77         r_SpType   Spect_Type_Source  /Source of spectral type
```
")
      (python
       (do0
	"#export"
	))
      (python
    
	"#export"
	(setf cap (cv2.VideoCapture args.filename #+nil  (string
					;"ISS Timelapse - Stars Above The World (29 _ 30 Marzo 2017)-8fCLTeY7tQg.mp4.part"
							  "/home/martin/stars_XnRy3sJqfu4.webm"
				     )))
	(unless (cap.isOpened)
	  (print (string "error opening video stream or file")))
	(cap.set cv2.CAP_PROP_POS_FRAMES
		 args.skip_frames)
	(while (cap.isOpened)
	       (for (n (range args.decimate_frames))
		(setf (ntuple ret frame)
		      (cap.read)))
	       (if ret
		   (do0
		    #+nil
		    (do0 
		     (setf da_rgb (aref frame
					(slice "" 512)
					(slice 900 "")
					":")
			   )
		     (setf da (aref frame
				    (slice "" 512)
				    (slice 900 "")
				    1))
		     (setf peaks (* 255 (skimage.morphology.h_maxima da args.threshold)))
		     (setf (aref da_rgb ":" ":" 1)
			   peaks
			   ))
		    (cv2.imshow (string "frame")
				frame ; da_rgb
				)
		    (when (== (& (cv2.waitKey 25)
				 #xff  )
			      (ord (string "q")))
		      break))
		   break)
	       )
	(cap.release)
	(cv2.destroyAllWindows)
	))))))



