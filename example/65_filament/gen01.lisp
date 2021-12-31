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
      ;matplotlib
      ;(s skyfield)
					;(ds dataset)
      ;cv2
      ))
  (let ((nb-file "script/01_convert.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp convert01"))
      (python (do0
	       
	       "#export"
	       (do0
					;"%matplotlib notebook"
		#+nil (do0
		      
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
			  ;(xrp xarray.plot)
			  ;skimage.restoration
			  ;skimage.morphology
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
		
		;"from cv2 import *"
	      	#+nil	(imports-from (matplotlib.pyplot
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
	       (setf filename  (string "/home/martin/stage/cl-py-generator/example/70_star_tracker/source/hip_main.dat")))))
      (python
       (do0
	"#export"
	(setf parser (argparse.ArgumentParser))
	(parser.add_argument (string "-i")
			     :dest (string "filename")
			    ; :required True
			     :default (string "/home/martin/stage/cl-py-generator/example/70_star_tracker/source/hip_main.dat")
			     :help (string "input file")
			     :metavar (string "FILE"))
	(parser.add_argument (string "-o")
			     :dest (string "ofile")
			     ;:required True
			     :default (string "out")
			     :help (string "output file")
			     :metavar (string "FILE"))
	(setf args (parser.parse_args))
	(print args)
	))
      #+nil(python
       (do0
	(comments "https://heasarc.gsfc.nasa.gov/W3Browse/all/hipparcos.html"
		  "https://github.com/skyfielders/python-skyfield/blob/master/skyfield/data/hipparcos.py")))
      
      ,(let ((l ` ((Catalog    "/Catalogue (H=Hipparcos)")
		     (HIP        "/Identifier (HIP number)")
		     (Proxy      "/Proximity flag")
		     (RAhms      "/RA in h m s, ICRS (J1991.25)")
		     (DEdms      "/Dec in deg, ICRS (J1991.25)")
		     (Vmag       "/Magnitude in Johnson V")
		     (VarFlag    "/Coarse variability flag")
		     (r_Vmag     "/Source of magnitude")
		     (RAdeg      "/RA in degrees (ICRS, Epoch-J1991.25)")
		     (DEdeg      "/Dec in degrees (ICRS, Epoch-J1991.25)")
		     (AstroRef   "/Reference flag for astrometry")
		     (Plx        "/Trigonometric parallax")
		     (pmRA       "/Proper motion in RA")
		     (pmDE       "/Proper motion in Dec")
		     (e_RAdeg    "/Standard error in RA*cos(Dec_Deg)")
		     (e_DEdeg    "/Standard error in Dec_Deg")
		     (e_Plx      "/Standard error in Parallax")
		     (e_pmRA     "/Standard error in pmRA")
		     (e_pmDE     "/Standard error in pmDE")
		     (DE--RA      "/(DE over RA)xCos(delta)")
		     (Plx--RA     "/(Plx over RA)xCos(delta)")
		     (Plx--DE     "/(Plx over DE)")
		     (pmRA--RA    "/(pmRA over RA)xCos(delta)")
		     (pmRA--DE    "/(pmRA over DE)")
		     (pmRA--Plx   "/(pmRA over Plx)")
		     (pmDE--RA    "/(pmDE over RA)xCos(delta)")
		     (pmDE--DE    "/(pmDE over DE)")
		     (pmDE--Plx   "/(pmDE over Plx)")
		     (pmDE--pmRA  "/(pmDE over pmRA)")
		     (F1         "/Percentage of rejected data")
		     (F2         "/Goodness-of-fit parameter")
		     (---        "/HIP number (repetition)")
		     (BTmag      "/Mean BT magnitude")
		     (e_BTmag    "/Standard error on BTmag")
		     (VTmag      "/Mean VT magnitude")
		     (e_VTmag    "/Standard error on VTmag")
		     (m_BTmag    "/Reference flag for BT and VTmag")
		     (B-V        "/Johnson BV colour")
		     (e_B-V      "/Standard error on BV")
		     (r_B-V      "/Source of BV from Ground or Tycho")
		     (V-I        "/Colour index in Cousins' system")
		     (e_V-I      "/Standard error on VI")
		     (r_V-I      "/Source of VI")
		     (CombMag    "/Flag for combined Vmag, BV, VI")
		     (Hpmag      "/Median magnitude in Hipparcos system")
		     (e_Hpmag    "/Standard error on Hpmag")
		     (Hpscat     "/Scatter of Hpmag")
		     (o_Hpmag    "/Number of observations for Hpmag")
		     (m_Hpmag    "/Reference flag for Hpmag")
		     (Hpmax      "/Hpmag at maximum (5th percentile)")
		     (HPmin      "/Hpmag at minimum (95th percentile)")
		     (Period     "/Variability period (days)")
		     (HvarType   "/Variability type")
		     (moreVar    "/Additional data about variability")
		     (morePhoto  "/Light curve Annex")
		     (CCDM       "/CCDM identifier")
		     (n_CCDM     "/Historical status flag")
		     (Nsys       "/Number of entries with same CCDM")
		     (Ncomp      "/Number of components in this entry")
		     (MultFlag   "/Double and or Multiple Systems flag")
		     (Source     "/Astrometric source flag")
		     (Qual       "/Solution quality flag")
		     (m_HIP      "/Component identifiers")
		     (theta      "/Position angle between components")
		     (rho        "/Angular separation of components")
		     (e_rho      "/Standard error of rho")
		     (dHp        "/Magnitude difference of components")
		     (e_dHp      "/Standard error in dHp")
		     (Survey     "/Flag indicating a Survey Star")
		     (Chart      "/Identification Chart")
		     (Notes      "/Existence of notes")
		     (HD         "/HD number <III 135>")
		     (BD         "/Bonner DM <I 119>, <I 122>")
		     (CoD        "/Cordoba Durchmusterung (DM) <I 114>")
		     (CPD        "/Cape Photographic DM <I 108>")
		     (_V-I_red   "/VI used for reductions")
		     (SpType     "/Spectral type")
		     (r_SpType   "/Source of spectral type"))))
	 `(python
	  (do0
	   "#export"
	   (setf column_names (list ,@(loop for (e f) in l
					    collect
					    `(string ,e))))
	   (setf df_ (pd.read_csv
		     args.filename
		     :sep (string "|")
		     :names column_names
		     :usecols (list ,@(loop for e in `(Vmag RAdeg DEdeg)
					    collect
					    `(string ,e)))
		     :na_values (list    (string "     ")
					 (string "       ")
					 (string "        ")
					 (string "            "))))
	   (setf df (aref df_ (< df_.Vmag 6)))
	   (print df)
	   (comments "store binary file")
	   (setf a (dot df values (astype np.float32)))
	   (setf fn (dot (string "{}_{}x{}_float32.raw")
			      (format args.ofile
				      (aref a.shape 0)
				      (aref a.shape 1))))
	   (print (dot (string "store in {}.")
		       (format fn)))
	   (dot a
		(tofile fn))
	   ))))))))



