(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(setf *features* (set-difference *features* (list :more)))
(setf *features* (set-exclusive-or *features* (list :more)))


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

	       )

	     ))

    )

  (let* ((name `MemoryMappedComplexShortFile)
	 (members `((file :type "boost::iostreams::mapped_file_source" :param nil)
		    (data :type "std::complex<short>*" :initform-class nullptr :param nil)
		    (filename :type "const std::string&" :param t)
		   
		   )))
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
			  (values "std::complex<short>&"))
		 (return (aref data_ index)))

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
      MemoryMappedComplexShortFile.h)
	

     (defun glfw_error_callback (err desc)
       (declare (type int err)
		(type "const char*" desc))
       ,(lprint :msg "GLFW erro:"
		:vars `(err desc)))

     #-nil (defun DemoImplot ()
	     (let ((x)
		   (y1)
		   (y2))
	       (declare (type "static std::vector<double>" x y1 y2))
	       (when (x.empty)
		 (dotimes (i 1000)
		   (let ((x_ (* #.pi (/ 4d0 1000d0) i)))
		     (x.push_back x_)
		     (y1.push_back (cos x_))
		     (y2.push_back (sin x_)))))
					;(ImGuiMd--Render (string "# This is a plot"))
	       (when (ImPlot--BeginPlot (string "Plot"))
		 ,@(loop for e in `(y1 y2)
			 collect
			 `(ImPlot--PlotLine (string ,e)
					    (x.data)
					    (dot ,e (data))
					    (static_cast<int> (x.size))))
		 (ImPlot--EndPlot))))

     
      
    
    
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))


       

       (glfwSetErrorCallback glfw_error_callback)
       (when (== 0 (glfwInit))
	 (return 1))
       (let ((glsl_version (string "#version 130")))
	 (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 3)
	 (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0))
       (let ((*window (glfwCreateWindow 800 600
					(string "imgui_dsp")
					nullptr nullptr)))
	 (when (== nullptr window)
	   (return 1))
	 (glfwMakeContextCurrent window)
	 #+more ,(lprint :msg "enable vsync")
	 (glfwSwapInterval 1)
	 (IMGUI_CHECKVERSION)
	 
	 (ImGui--CreateContext)
	 (ImPlot--CreateContext)

	 (let ((&io (ImGui--GetIO)))
	   (setf io.ConfigFlags (or io.ConfigFlags
				    ImGuiConfigFlags_NavEnableKeyboard)))
					;(ImGui--StyleColorsDark)
	 (ImGui_ImplGlfw_InitForOpenGL window true)
	 (ImGui_ImplOpenGL3_Init glsl_version)
	 )
       
       (let ((ca (GpsCACodeGenerator 4)))
	 ,(lprint :msg "CA")
	 (ca.print_square (ca.generate_sequence 1023)))

       (handler-case
	   (do0
	    (let ((fn (string "/mnt5/capturedData_L1_rate10MHz_bw5MHz_iq_short.bin"))
		  (file (MemoryMappedComplexShortFile
			 fn)))
	      ,(lprint :msg "first element"
		       :vars `(fn (dot (aref file 0) (real)))))
	    (while (!glfwWindowShouldClose window)
		   (glfwPollEvents)
		   (ImGui_ImplOpenGL3_NewFrame)
		   (ImGui_ImplGlfw_NewFrame)
		   (ImGui--NewFrame)
		   (DemoImplot)
		   (ImGui--Render)
		   (let ((w 0)
			 (h 0))
		     (glfwGetFramebufferSize window &w &h)
		     (glViewport 0 0 w h)
		     (glClearColor  0 0 0 1)
		     (glClear GL_COLOR_BUFFER_BIT)
		     (ImGui_ImplOpenGL3_RenderDrawData (ImGui--GetDrawData)))
		   (glfwSwapBuffers window)))
	 ("const std::runtime_error&" (e)
	   ,(lprint :msg "error:"
		    :vars `((e.what)))
	   (return -1)))

       (do0
	(ImGui_ImplOpenGL3_Shutdown)
	(ImGui_ImplGlfw_Shutdown)
	(ImPlot--DestroyContext)
	(ImGui--DestroyContext)
	(glfwDestroyWindow window)
	(glfwTerminate))
       
       
       
       (return 0)))
   :omit-parens t
   :format t
   :tidy t)

  )
