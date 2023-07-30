(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(setf *features* (set-difference *features* (list :more
						  :memoize-plan
						  :guru-plan)))
(setf *features* (set-exclusive-or *features* (list :more
						    ;:guru-plan
						    ;:memoize-plan
						    )))


(let* ( ;; (elem-type "float") (acq-sdr-type 'SOAPY_SDR_CF32)
      (elem-type "short") (acq-sdr-type 'SOAPY_SDR_CS16)
      (acq-type (format nil "std::complex<~a>" elem-type))
      (fifo-type (format nil "std::deque<~a>" acq-type)))
	
  (progn
    (defparameter *source-dir* #P"example/131_sdr/source03/src/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  
  (let* ((name `GpsCACodeGenerator)
	(sat-def `((2 6)
		   (3 7)
		   (4 8)
		   (5 9)
		   (1 9)
		   (2 10)
		   (1 8)
		   (2 9)
		   (3 10)
		   (2 3)
		   (3 4)
		   (5 6)
		   (6 7)
		   (7 8)
		   (8 9)
		   (9 10)
		   (1 4)
		   (2 5)
		   (3 6)
		   (4 7)
		   (5 8)
		   (6 9)
		   (1 3)
		   (4 6)
		   (5 7)
		   (6 8)
		   (7 9)
		   (8 10)
		   (1 6)
		   (2 7)
		  (3 8)
		   (4 9)))
	(members `((registerSize :type "static constexpr size_t" :initform-class 10 :param nil)
		   (g1FeedbackBits :type "static constexpr std::array<int,2>"
				   :initform-class (curly 3 10) :param nil)
		   (g2FeedbackBits :type "static constexpr std::array<int,6>"
				   :initform-class (curly 2 3 6 8 9 10) :param nil)
		   (g2Shifts :type ,(format nil "static constexpr std::array<std::pair<int,int>,~a>" (length sat-def))
			     :initform-class
			     (curly ,@(loop for (e f)
					      in sat-def
					    collect
					    `(std--make_pair ,e ,f)) )
			     :param nil)
		   (g1 :type "std::deque<bool>")
		   (g2 :type "std::deque<bool>")
		   (prn :type int :param t)
		   )))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> vector
				   array
				   deque
				   cstddef)
			)
     :implementation-preamble
     `(do0
       (include<> stdexcept
		 ; cstring
		  cmath
		  iostream)
       )
     :code `(do0
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param initform initform-class) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param initform initform-class) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param initform initform-class) e
					(let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					      (nname_ (format nil "~a_"
							      (cl-change-case:snake-case (format nil "~a" name)))))
					  (cond
					    (param
					     `(,nname_ ,nname))
					    (initform
					     `(,nname_ ,initform)))))))
		   )
		  (explicit)	    
		  (values :constructor))
		 (when (logior (< prn_ 1)
			       (< ,(length sat-def) prn_ ))
		   (throw (std--invalid_argument (+ (string "Invalid PRN: ")
						    (std--to_string prn_))))
		   )
		 (do0 (g1_.resize register_size_ true)
			(g2_.resize register_size_ true))
		 )
	       
	       (defmethod generate_sequence (n)
		 (declare (type size_t n)
			  (values "std::vector<bool>"))
		 (let ((sequence (std--vector<bool>)))
		   (sequence.reserve n)
		   (dotimes (i n)
		     (declare (type size_t i))
		     (sequence.push_back (step)))
		   (return sequence)))

	       (defmethod print_square (v)
		 (declare (type "const std::vector<bool> &" v))
		 (let ((size (v.size))
		       (side (static_cast<size_t> (std--sqrt size))))
		   (dotimes (i size)
		     (declare (type size_t i))
		     (let ((o (? (aref v i)
				 (string "\\u2588")
				 (string " "))))
		      (<< std--cout o)
		       (when (== 0 (% (+ i 1 ) side))
			 (<< std--cout (string "\\n")))))))
	       "private:"
	       (defmethod step ()
		 (declare (values bool))
		 (let ((new_g1_bit false))
		   (for-range (i g1_feedback_bits_)
			      (setf new_g1_bit (^ new_g1_bit (aref g1_ (- i 1))))))
		 (let ((new_g2_bit false))
		   (for-range (i g2_feedback_bits_)
			      (setf new_g2_bit (^ new_g2_bit (aref g2_ (- i 1))))))
		 (do0 (g1_.push_front new_g1_bit)
		      (g1_.pop_back))
		 (do0 (g2_.push_front new_g2_bit)
		      (g2_.pop_back))
		 (let ((delay1 (dot (aref g2_shifts_ (- prn_ 1))
				    first))
		       (delay2 (dot (aref g2_shifts_ (- prn_ 1))
				    second))))
		 (return (^ (g1_.back)
			    (^ (aref g2_ (- delay1 1))
			       (aref g2_ (- delay2 1))))))
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (if initform-class
					  `(space ,type (setf ,nname_ ,initform-class))
					  `(space ,type ,nname_))))))

	       ))))

  (let* ((name `MemoryMappedComplexShortFile)
	 (members `((file :type "boost::iostreams::mapped_file_source" :param nil)
		    (data :type "std::complex<short>*" :initform-class nullptr :param nil)
		    (filename :type "const std::string&" :param t))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> iostream
				   fstream
				   vector
				   complex
				   boost/iostreams/device/mapped_file.hpp)
			)
     :implementation-preamble
     `(do0
       (include<> stdexcept
			)
       )
     :code `(do0
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param initform initform-class) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param initform initform-class) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param initform initform-class) e
					(let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					      (nname_ (format nil "~a_"
							      (cl-change-case:snake-case (format nil "~a" name)))))
					  (cond
					    (param
					     `(,nname_ ,nname))
					    (initform
					     `(,nname_ ,initform)))))))
		   )
		  (explicit)	    
		  (values :constructor))
		 (file_.open filename)
		 (when (!file_.is_open)
		   (throw (std--runtime_error (+ (string "Unable to open file: ")
						 filename))))
		 (setf data_ (reinterpret_cast<std--complex<short>*> (const_cast<char*> (file_.data))))
		 )

	       (defmethod "operator[]" (index)
		 (declare (type "std::size_t" index)
			  (values "std::complex<short>&")
			  (const))
		 (return (aref data_ index)))

	       (defmethod size ()
		 (declare (values "std::size_t")
			  (const))
		 (return (dot file_ (size))))
	       
	       (defmethod ~MemoryMappedComplexShortFile ()
		 (declare (values :constructor))
		 (file_.close))
	       
	       "private:"
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (if initform-class
					  `(space ,type (setf ,nname_ ,initform-class))
					  `(space ,type ,nname_)))))))))

    )

  (let* ((name `FFTWManager)
	 (members `(#+memoize-plan (plans :type "std::map<std::pair<int,int>,fftw_plan>" :param nil)
		    ;(window_size :type "int" :initform 512 :param t)
		    )))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> fftw3.h
				   map
				   vector
				   complex
				   )
			)
     :implementation-preamble
     `(do0
       (include<> stdexcept
		  fstream
		  #+more iostream)
       )
     :code `(do0
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param initform initform-class) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param initform initform-class) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param initform initform-class) e
					(let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					      (nname_ (format nil "~a_"
							      (cl-change-case:snake-case (format nil "~a" name)))))
					  (cond
					    (param
					     `(,nname_ ,nname))
					    (initform
					     `(,nname_ ,initform)))))))
		   )
		  (explicit)	    
		  (values :constructor))
		 )

	       #+nil (defmethod print_plans ()
		 (declare (const))
		 (for-range (pair plans_)
		      (let ((windowSize pair.first.first)
			    (nThreads pair.first.second)
			    (planAddress pair.second))
		       ,(lprint :msg "print_plans"))))

	       (defmethod get_plan (windowSize &key (nThreads 1))
		 (declare (type int windowSize nThreads)
			  (values fftw_plan))
		 (when (<= windowSize 0)
		   (throw (std--invalid_argument (string "window size must be positive"))))

		 
		#+nil (print_plans)
		 ,(lprint :msg "lookup plan"
			 :vars `(windowSize nThreads))
		 #+memoize-plan (let ((iter (plans_.find (curly windowSize nThreads))))) 
		 (do0 
		   (if #-memoize-plan true #+memoize-plan (== (plans_.end) iter)
		       (do0 #+memoize-plan ,(lprint :msg "The plan hasn't been used before. Try to load wisdom or generate it."
				     :vars `(windowSize nThreads))

			    (do0 (let ((wisdom_filename (+ (string "wisdom_")
							   (std--to_string windowSize)
							   (string ".wis")))))
				 (when (< 1 nThreads)
				   (setf wisdom_filename
					 (+ (string "wisdom_")
					    (std--to_string windowSize)
					    (string "_threads")
					    (std--to_string nThreads)
					    (string ".wis")))))
			    
			    (let ((*in (fftw_alloc_complex windowSize))
				  (*out (fftw_alloc_complex windowSize)))
			      (when (logior !in
					    !out)
				(do0 (fftw_free in)
				     (fftw_free out)
				     (throw (std--runtime_error (string "Failed to allocate memory for fftw plan")))))

			      (let ((wisdomFile (std--ifstream wisdom_filename)))
				(when (wisdomFile.good)
				  ,(lprint :msg "read wisdom from existing file"
					   :vars `(wisdom_filename))
				  (wisdomFile.close)
				  (fftw_import_wisdom_from_filename (wisdom_filename.c_str)))
				(if (< 1 nThreads)
				    (do0 ,(lprint :msg "plan 1d fft with threads"
						  :vars `(nThreads))
					 (fftw_plan_with_nthreads nThreads))
				    ,(lprint :msg "plan 1d fft without threads"
					     :vars `(nThreads)))
				#-guru-plan (let ((p (fftw_plan_dft_1d windowSize
								 in out
								 FFTW_FORWARD
								 FFTW_MEASURE
								 ))))
				#+guru-plan (let ((dim (fftw_iodim (designated-initializer :n windowSize
									       :is 1
									       :os 1)))
				      (p 
					(fftw_plan_guru_dft 1 ;; rank
							    &dim ;; dims
							    0 ;; howmany_rank
							    nullptr ;; howmany_dims
							    in	;; in 
							    out ;; out
							    FFTW_FORWARD ;; sign
					;FFTW_MEASURE ;; flags
							    (or FFTW_MEASURE FFTW_UNALIGNED)
							    )))
				  )
				(when !p
				  (do0 (fftw_free in)
				       (fftw_free out)
				       (throw (std--runtime_error (string "Failed to create fftw plan")))))
				,(lprint :msg "plan successfully created")
				(unless (wisdomFile.good)
				  ,(lprint :msg "store wisdom to file"
					   :vars `(wisdom_filename))
				  (wisdomFile.close)
				  (fftw_export_wisdom_to_filename (wisdom_filename.c_str)))
				)
			      (do0 (do0
				    ,(lprint :msg "free in and out")
				    (fftw_free in)
				    (fftw_free out)
				    #-memoize-plan (return p)
				    )
				   #+memoize-plan(do0
				    ,(lprint :msg "store plan in class"
					     :vars `(windowSize nThreads))
				     (let ((insertResult (dot plans_
									    (insert (curly (curly windowSize nThreads) p))
									    )))
						     (setf iter (dot insertResult first))
						     ,(lprint :msg "inserted new key"
							      :vars `((plans_.size) insertResult.second))))
				   )
			      ))
		       #+memoize-plan (do0
			,(lprint :msg "plan has been used recently, reuse it.")))
		   
 		   #+memoize-plan (return iter->second)))

	       
	       (defmethod fftshift (in)
		 (declare 
		  (type "const std::vector<std::complex<double>>&" in)
		  (values "std::vector<std::complex<double>>"))
		 (let ((mid (+ (in.begin)
			       (/ (in.size) 2)))
		       (out (std--vector<std--complex<double>> (in.size))))
		   (std--copy mid (in.end) (out.begin))
		   (std--copy (in.begin) mid (+ (out.begin)
						(std--distance mid (in.end))))
		   (return out)))

	       (defmethod fft (in windowSize)
		 (declare (type int windowSize)
			  (type "const std::vector<std::complex<double>>&" in)
			  (values "std::vector<std::complex<double>>"))
		 (when (!= windowSize (in.size))
		   (throw (std--invalid_argument (string "Input size must match window size."))))
		 (let ((out (std--vector<std--complex<double>> windowSize)))
		   (fftw_execute_dft (get_plan windowSize 6)
				     (reinterpret_cast<fftw_complex*>
				      (const_cast<std--complex<double>*> (in.data)))
				     (reinterpret_cast<fftw_complex*>
				      (const_cast<std--complex<double>*> (out.data))))
		   (return (fftshift out))))

	       (defmethod ~FFTWManager ()
		 (declare (values :constructor))
		 #+memoize-plan
		 (for-range (kv plans_)
			    (fftw_destroy_plan kv.second)))
	       
	       "private:"
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (if initform-class
					  `(space ,type (setf ,nname_ ,initform-class))
					  `(space ,type ,nname_)))))))))

    )
  
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      iostream
					;string
					;complex
      vector
					;algorithm
      
					;chrono

					;filesystem
					;unistd.h
					;cstdlib

      cmath)
     (include
      
      implot.h
      imgui.h
      imgui_impl_glfw.h
      imgui_impl_opengl3.h
      GLFW/glfw3.h
      )
     (include
      GpsCACodeGenerator.h
      MemoryMappedComplexShortFile.h
      FFTWManager.h)
	

     (defun glfw_error_callback (err desc)
       (declare (type int err)
		(type "const char*" desc))
       ,(lprint :msg "GLFW erro:"
		:vars `(err desc)))

     (defun DrawPlot (file)
       (declare (type "const MemoryMappedComplexShortFile&" file))

       (let ((start 0)
	    
	     
	     (maxStart (static_cast<int> (/ (file.size)
					    (sizeof "std::complex<short>")))))
	 (declare (type "static int" start ))
	 (ImGui--SliderInt (string "Start")
			   &start 0 maxStart)
	 
	 )
       ,(let* ((l-size-start 5)
	      (l-size-end 20)
	      (l-size-init (- 9 l-size-start))
	      (l-sizes `(,@(loop for i from l-size-start upto l-size-end
				collect
				(format nil "~a" (expt 2 i))))))
	 `(let (
	       
	       (windowSizeIndex ,l-size-init)
	       (itemsInt (std--vector<int> (curly ,@l-sizes)))
	       (windowSize (aref itemsInt windowSizeIndex))
	       (items (std--vector<std--string> (curly ,@(mapcar #'(lambda (x) `(string ,x))
								 l-sizes))))
	       )
	   (declare (type "static int" windowSizeIndex))
	    (when (ImGui--BeginCombo (string "Window size")
				    (dot (aref items windowSizeIndex)
					 (c_str)))
	      (dotimes (i (items.size))
		(let ((is_selected (== windowSizeIndex i)))
		  (when (ImGui--Selectable (dot (aref items i)
						(c_str))
					   is_selected)
		    (setf windowSizeIndex i
			  windowSize (aref itemsInt windowSizeIndex)))
		  (when is_selected
		    (ImGui--SetItemDefaultFocus))))
	      (ImGui--EndCombo))
	   ))
       (when (logand (<= (+ start windowSize) maxStart)
		     (< 0 windowSize))

	 (let ((x (std--vector<double> windowSize))
	       (y1 (std--vector<double> windowSize))
	       (y2 (std--vector<double> windowSize)))
	   (dotimes (i windowSize)
	     (let ((z (aref file (+ start i))))
	       (setf (aref x i) i	;(+ start i)
		     (aref y1 i) (z.real)
		     (aref y2 i) (z.imag))))

	   (do0
	    
	    (when (ImPlot--BeginPlot (string "Waveform (I/Q)"))
	      #+nil (ImPlot--SetNextAxisLimits ImAxis_X1 start (+ start windowSize))				
	      ,@(loop for e in `(y1 y2)
		      collect
		      `(ImPlot--PlotLine (string ,e)
					 (x.data)
					 (dot ,e (data))
					 windowSize))
	      (ImPlot--EndPlot)))

	   (let ((logScale false))
	     (declare (type "static bool" logScale))
	     (ImGui--Checkbox (string "Logarithmic Y-axis")
			      &logScale)
	     (handler-case
		 (let ((man (FFTWManager))
		       (in (std--vector<std--complex<double>> windowSize))
		       (nyquist (/ windowSize 2d0))
		       (sampleRate 10d6))
		   (dotimes (i windowSize)
		     (setf (aref x i) i))
		  
		   
		   (dotimes (i windowSize)
		     (let ((zs (aref file (+ start i)))
			   (zr (static_cast<double> (zs.real)))
			   (zi (static_cast<double> (zs.imag)))
			   (z (std--complex<double> zr zi)))
		       (setf (aref in i) z)))
		   (let ((out (man.fft in windowSize)))
		     
		     (if logScale
			 (dotimes (i windowSize)
			   (setf (aref y1 i) (* 10d0 (log10 (std--abs (aref out i))))))
			 (dotimes (i windowSize)
			   (setf (aref y1 i) (std--abs (aref out i)))))
		     (do0
		      ;; (ImPlot--SetNextAxisLimits ImAxis_X1 (* -.5 sampleRate) (* .5 sampleRate))				
		      (when (ImPlot--BeginPlot (string "FFT"))
			
			,@(loop for e in `(y1)
				collect
				`(ImPlot--PlotLine (string ,e)
						   (x.data)
						   (dot ,e (data))
						   windowSize))
			(ImPlot--EndPlot)))))
	       ("const std::exception&" (e)
		 (ImGui--Text (string "Error while processing FFT: %s")
			      (e.what))))))))
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))

       (glfwSetErrorCallback glfw_error_callback)
       (when (== 0 (glfwInit))
	 ,(lprint :msg "glfw init failed")
	 (return 1))
       
       #+more
       (do0		 ;let ((glsl_version (string "#version 130")))
	(glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 3)
	(glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0))
       (let ((*window (glfwCreateWindow 800 600
					(string "imgui_dsp")
					nullptr nullptr)))
	 (when (== nullptr window)
	   ,(lprint :msg "failed to create glfw window")
	   (return 1))
	 (glfwMakeContextCurrent window)
	 ,(lprint :msg "enable vsync")
	 (glfwSwapInterval 1)
	 (IMGUI_CHECKVERSION)
	 
	 (ImGui--CreateContext)
	 (ImPlot--CreateContext)

	 (let ((&io (ImGui--GetIO)))
	   (setf io.ConfigFlags (or io.ConfigFlags
				    ImGuiConfigFlags_NavEnableKeyboard)))
					;(ImGui--StyleColorsDark)
	 (ImGui_ImplGlfw_InitForOpenGL window true)
	 (ImGui_ImplOpenGL3_Init (string "#version 130"))
	 (glClearColor  0 0 0 1)
	 )
       
       #+nil (let ((ca (GpsCACodeGenerator 4)))
	       ,(lprint :msg "CA")
	       (ca.print_square (ca.generate_sequence 1023)))

       (handler-case
	   (do0
	    (let ((fn (string "/mnt5/capturedData_L1_rate10MHz_bw5MHz_iq_short.bin"))
		  (file (MemoryMappedComplexShortFile
			 fn)))
	      ,(lprint :msg "first element"
		       :vars `(fn (dot (aref file 0) (real))))
	      #+nil  (let ((z0 (aref file 0)))))
	    (while (!glfwWindowShouldClose window)
		   (glfwPollEvents)
		   (ImGui_ImplOpenGL3_NewFrame)
		   (ImGui_ImplGlfw_NewFrame)
		   (ImGui--NewFrame)
		   (DrawPlot file)
		   (ImGui--Render)
		   (let ((w 0)
			 (h 0))
		     (glfwGetFramebufferSize window &w &h)
		     (glViewport 0 0 w h)
		   
		     (glClear GL_COLOR_BUFFER_BIT)
		     )
		   (ImGui_ImplOpenGL3_RenderDrawData (ImGui--GetDrawData))
		   (glfwSwapBuffers window)))
	 ("const std::runtime_error&" (e)
	   ,(lprint :msg "error:"
		    :vars `((e.what)))
	   (return -1)))

       #+nil (do0
	      (ImGui_ImplOpenGL3_Shutdown)
	      (ImGui_ImplGlfw_Shutdown)
	      (ImPlot--DestroyContext)
	      (ImGui--DestroyContext)
	      (glfwDestroyWindow window)
	      (glfwTerminate))
       
       
       
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil))
