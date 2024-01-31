(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(let ()
  (defparameter *source-dir* #P"example/143_ryzen_monitor/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  (let* ((name `DiagramBase)
	 (members `((max-cores :type int :param t)
		    (max-points :type int :param t)
		    (diagrams :type "std::vector<DiagramData>")
		    ;(x :type "std::vector<float>")
		    ;(y :type "std::vector<float>")
		    (time-points :type "std::deque<float>"))))
    (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames (format nil "../tests/test_~a.cpp" name)
		     *source-dir*))
   `(do0
     (include ,(format nil "~a.h" name))
     (include<> gtest/gtest.h
		unistd.h)

     (space (TEST ,name AddDataPoint_AddPointToEmpty_HaveOnePoint)
	    (progn
	      (comments Arrange)
	      (let ((values (std--vector<float> (curly 10s0 11s0)))
		    (diagram (DiagramBase #+color (curly (curly 1s0 0s0 0s0 1s0))
					  (values.size)
					  10))
		    ))
	      
	      (comments Act)
	      (diagram.AddDataPoint 1s0 values)

	      (comments Assert)
	      
	      (EXPECT_EQ (dot diagram (GetTimePoints) (size)) 1)
	      (EXPECT_EQ (dot diagram (aref (GetDiagrams) 0) values (size)) 1)))

     (space (TEST ,name AddDataPoint_AddPointToOne_HaveTwoPoints)
	    (progn
	      (comments Arrange)
	      (let ((values (std--vector<float> (curly 10s0 11s0)))
		    (diagram (DiagramBase #+color (curly (curly 1s0 0s0 0s0 1s0))
					  (values.size)
					  10))
		    ))
	      
	      (comments Act)
	      (diagram.AddDataPoint 1s0 values)
	      (diagram.AddDataPoint 2s0 values)

	      (comments Assert)
	      
	      (EXPECT_EQ (dot diagram (GetTimePoints) (size)) 2)
	      (EXPECT_EQ (dot diagram (aref (GetDiagrams) 0) values (size)) 2)))

     (space (TEST ,name AddDataPoint_AddLastPoint_HaveThreePoints)
	    (progn
	      (comments Arrange)
	      (let ((values (std--vector<float> (curly 10s0 11s0)))
		    (diagram (DiagramBase #+color (curly (curly 1s0 0s0 0s0 1s0))
					  (values.size)
					  3))
		    ))
	      
	      (comments Act)
	      (diagram.AddDataPoint 1s0 values)
	      (diagram.AddDataPoint 2s0 values)
	      (diagram.AddDataPoint 3s0 values)

	      (comments Assert)
	      
	      (EXPECT_EQ (dot diagram (GetTimePoints) (size)) 3)
	      (EXPECT_EQ (dot diagram (aref (GetDiagrams) 0) values (size)) 3)))

     (space (TEST ,name AddDataPoint_AddOneMorePointsThanFit_HaveThreePoints)
	    (progn
	      (comments Arrange)
	      (let ((values (std--vector<float> (curly 10s0 11s0)))
		    (diagram (DiagramBase #+color (curly (curly 1s0 0s0 0s0 1s0))
					  2
					  3))
		    ))
	      
	      (comments Act)
	      (diagram.AddDataPoint 1s0 (curly 10s0 100s0))
	      (diagram.AddDataPoint 2s0 (curly 20s0 200s0))
	      (diagram.AddDataPoint 3s0 (curly 30s0 300s0))
	      (diagram.AddDataPoint 4s0 (curly 40s0 400s0))

	      (comments Assert)
	      
	      (EXPECT_EQ (dot diagram (GetTimePoints) (size)) 3)
	      (EXPECT_EQ (dot diagram (GetTimePoints) (at 2)) 4s0)
	      (EXPECT_EQ (dot diagram (GetDiagrams) (at 0) values (size)) 3)
	      (EXPECT_EQ (dot diagram (GetDiagrams) (at 0) values (at 2)) 40s0))))
   :omit-parens t
   :format t
   :tidy nil)
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> vector deque string)
			#+color (space struct Color (progn "float r,g,b,a;"))
			(space struct DiagramData (progn
					   "std::string name;"
					   "std::deque<float> values;"
					   #+color "Color color;")))
     :implementation-preamble
     `(do0
       
       (include<>
		  stdexcept
		  format
		  )
       )
     :code `(do0
	     
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name (#+color colors ,@(remove-if #'null
					   (loop for e in members
						 collect
						 (destructuring-bind (name &key type param (initform 0)) e
						   (let ((nname (intern
								 (string-upcase
								  (cl-change-case:snake-case (format nil "~a" name))))))
						     (when param
						       nname))))))
		 (declare
		  #+color (type "const std::vector<Color>&" colors)
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param (initform 0)) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param (initform 0)) e
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
		  (values :constructor)
		  )
		 (diagrams_.reserve max_cores_)
		 (dotimes (i max_cores_)
		   (diagrams_.push_back (curly (std--format (string "Core {}")
							    i)
					       (curly)
					       #+color (aref colors (% i (colors.size))))))
		 ;(x_.reserve max_points_)
		 ;(y_.reserve max_points_)
		 )

	       (defmethod AddDataPoint (time values)
		 (declare 
			  (type float time)
			  (type "const std::vector<float>&" values))
		 (unless (== (values.size)
			     (diagrams_.size))
		   (throw (std--invalid_argument (std--format (string "Number of values doesn't match the number of diagrams. expected: {} actual: {}")
							      (values.size)
							      (diagrams_.size)))))
		 (when (<= max_points_ (time_points_.size))
		   (time_points_.pop_front)
		   (for-range (diagram diagrams_)
			      (declare (type "auto&" diagram))
			      (unless (diagram.values.empty)
				(diagram.values.pop_front))))
		 (time_points_.push_back time)
		 (dotimes (i (values.size))
		  (dot (aref diagrams_ i)
		       values
		       (push_back (aref values i)))))

	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (get (cl-change-case:pascal-case (format nil "get-~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(defmethod ,get ()
					 (declare (values ,type))
					 (return ,nname_))))))
	       
	       "protected:"
	       
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_))))))))

    )

  (let* ((name `DiagramWithGui)
	 (members `()))
   
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> vector deque string )
			(include DiagramBase.h))
     :implementation-preamble
     `(do0
       (include<> stdexcept
		  format)
       (include implot.h))
     :code `(do0
	     (defclass ,name "public DiagramBase"
	       "public:"
	       "using DiagramBase::DiagramBase;"
	       
	       (defmethod RenderGui ()
		 (space struct PlotData
			(progn
			  "const std::deque<float>& time_points_;"
			  "const std::vector<DiagramData>& diagrams_;"
			  "int i;"))
		 (when (ImPlot--BeginPlot (string ""))
		   (dotimes (i max_cores_)
		     (let ((data (PlotData time_points_ diagrams_ i))))
		     (let ((getter (lambda (idx data)
				     (declare (type void* data)
					      (type int idx)
					      (values ImPlotPoint)
					      (capture ""))
				     (let ((*d (static_cast<PlotData*> data))))
				     (let ((x (dot d->time_points_ (at idx)))
					   (y (dot d->diagrams_ (at d->i) values (at idx)))))
				     (return (ImPlotPoint x y))))))
					;(ImPlot--SetNextLineStyle (aref coreColors i))
		     (ImPlot--SetupAxes (string "X")
					(string "Y")
					ImPlotAxisFlags_AutoFit
					ImPlotAxisFlags_AutoFit)
		     ;(ImPlot--SetupFinish)
		     (ImPlot--PlotLineG (dot (std--format (string "Core {:2}")
							  i)
					     (c_str))
					getter
					(reinterpret_cast<void*> &data)
					(time_points_.size)))
		   (ImPlot--EndPlot)))))))

  (let* ((name `CpuAffinityManagerBase)
	 (members `((selectedCpus :type std--bitset<12> :initform nil)
		    (pid :type pid_t :param t))))
    (write-source 
     (asdf:system-relative-pathname
      'cl-cpp-generator2 
      (merge-pathnames (format nil "../tests/test_~a.cpp" name)
		       *source-dir*))
     `(do0
       (include CpuAffinityManagerBase.h)
       (include<> gtest/gtest.h
		  unistd.h)

       (space (TEST ,name GetSelectedCpus_Initialized_FullBitset)
	      (progn
		(let ((manager (CpuAffinityManagerBase (getpid)))))
		(comments "FIXME: this only works on a twelf core cpu")
		(let ((expected_result (std--bitset<12> (string "111111111111")))
		      (actual_result (manager.GetSelectedCpus))))
		(EXPECT_EQ actual_result expected_result)))

       (space (TEST ,name SetSelectedCpus_Set_ValidBitset)
	      (progn
		(let ((manager (CpuAffinityManagerBase (getpid)))))
		(comments "FIXME: this only works on a twelf core cpu")
		(let ((expected_result (std--bitset<12> (string "101010101010")))
		      ))
		(manager.SetSelectedCpus expected_result)
		(let ((actual_result (manager.GetSelectedCpus))))
		(EXPECT_EQ actual_result expected_result)))

       (space (TEST ,name GetAffinity_Initialized_FullBitset)
	      (progn
		(let ((manager (CpuAffinityManagerBase (getpid)))))
		(comments "FIXME: this only works on a twelf core cpu")
		(let ((expected_result (std--bitset<12> (string "111111111111")))
		      ))
	      
		(let ((actual_result (manager.GetAffinity))))
		(EXPECT_EQ actual_result expected_result)))

       (space (TEST ,name ApplyAffinity_Set_ValidBitset)
	      (progn
		(let ((manager (CpuAffinityManagerBase (getpid)))))
		(comments "FIXME: this only works on a twelf core cpu")
		(let ((expected_result (std--bitset<12> (string "101010101010")))
		      ))

		(manager.SetSelectedCpus expected_result)
		(manager.ApplyAffinity)
	      
		(let ((actual_result (manager.GetAffinity))))
		(EXPECT_EQ actual_result expected_result))))
     :omit-parens t
     :format t
     :tidy nil)
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> sched.h
				   unistd.h
				   bitset
				   cstring
				   string))
     :implementation-preamble
     `(do0
       
       (include<> sched.h
		  stdexcept
		  )
       )
     :code `(do0
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param (initform 0)) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param (initform 0)) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param (initform 0)) e
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
		  (values :constructor)
		  )
		 (let ((cpuset (cpu_set_t)))
		   (if (== 0 (sched_getaffinity pid_ (sizeof cpu_set_t) &cpuset))
		       (dotimes (i 12)
			 (when (CPU_ISSET i &cpuset)
			   (selected_cpus_.set i)))
		       (throw (std--runtime_error (string "Failed to get CPU affinity"))))))
	       (defmethod GetSelectedCpus ()
		 (declare (values "std::bitset<12>"))
		 (return selected_cpus_))
	       (defmethod SetSelectedCpus (selected_cpus)
		 (declare (type "std::bitset<12>" selected_cpus))
		 (setf selected_cpus_ selected_cpus))
	       (defmethod GetAffinity ()
		 (declare (values "std::bitset<12>"))
		 (let ((cpuset (cpu_set_t)))
		   (when (== 0 (sched_getaffinity pid_
						  (sizeof cpu_set_t)
						  &cpuset))
		     (let ((affinity (std--bitset<12>)))
		       (dotimes (i 12)
			 (when (CPU_ISSET i &cpuset)
			   (affinity.set i)))
		       (return affinity)))
		   (throw (std--runtime_error (string "Failed to get CPU affinity")))))
	       (defmethod ApplyAffinity ()
		 (let ((cpuset (cpu_set_t)))
		   (CPU_ZERO &cpuset)
		   (dotimes (i 12)
		     (when (aref selected_cpus_ i)
		       (CPU_SET i &cpuset)))
		   (when (!= 0 (sched_setaffinity pid_ (sizeof cpu_set_t) &cpuset))
		     (throw (std--runtime_error (string "Failed to set CPU affinity"))))))
	       
	       "protected:"
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_)))))))))

  (let* ((name `CpuAffinityManagerWithGui)
	 (members `((selectedCpus :type std--bitset<12> :initform nil)
		    (pid :type pid_t :param t))))
  
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> sched.h
				   unistd.h
				   bitset
				   cstring
				   string
				   )
			(include CpuAffinityManagerBase.h
				 DiagramWithGui.h))
     :implementation-preamble
     `(do0
       (include imgui.h)
       (include<> sched.h
		  stdexcept
		  
		  )
       )
     :code `(do0
	     (defclass ,name "public CpuAffinityManagerBase"
	       "public:"
	       "using CpuAffinityManagerBase::CpuAffinityManagerBase;"
	       (defmethod RenderGui ()
		 (ImGui--Begin (string "CPU Affinity"))
		 (ImGui--Text  (string "Select CPUs for process ID: %d") pid_)
		 (let ((affinityChanged false))
		   (dotimes (i 12)
		     (let ((label (+ (std--string (string "CPU "))
				     (std--to_string i)))))
		     (let ((isSelected  (aref selected_cpus_ i)))
		       (declare (type bool isSelected)))
		     (when (ImGui--Checkbox (label.c_str)
					    &isSelected)
		       (setf (aref selected_cpus_ i) isSelected)
		       (setf affinityChanged true)))
		   (when affinityChanged
		     (ApplyAffinity))
		   (ImGui--End)
		   ))
	       ))))
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include imgui.h
	      imgui_impl_glfw.h
	      imgui_impl_opengl3.h
	      implot.h
	      CpuAffinityManagerWithGui.h
	      )
     (include<> GLFW/glfw3.h
		format
		iostream
		unistd.h
		vector deque chrono
		cmath)
     (space extern "\"C\""
	    (progn
	      
	      (include<> libsmu.h
			 readinfo.h
			 pm_tables.h
			 )
	      "extern smu_obj_t obj;"
	      "void start_pm_monitor(unsigned int);"
	      "int select_pm_table_version(unsigned int version, pm_table *pmt, unsigned char *pm_buf);"
	      "void disabled_cores_0x400005(pm_table *pmt, system_info *sysinfo);"
	      ))

     
     
     (defun glfw_error_callback (err description)
       (declare (type int err)
		(type "const char*" description))
       ,(lprint :vars `(err description)))


     (defun start_pm_monitor2 ()
       (declare (values "std::tuple<system_info,unsigned char*,pm_table>"))
       (unless (smu_pm_tables_supported &obj)
	 ,(lprint :msg "pm tables not supported on this platform")
	 (exit 0))
       (let ((pm_buf ("static_cast<unsigned char*>"
		      (calloc obj.pm_table_size
			      (sizeof "unsigned char")))))
	 
	 (unless pm_buf
	   ,(lprint :msg "could not allocate PM Table")
	   (exit 0)))
       (let ((pmt (pm_table)))
	 (unless (select_pm_table_version
		  obj.pm_table_version
		  &pmt
		  pm_buf)
	   ,(lprint :msg "pm table version not supported")
	   (exit 0)))

       (when (< obj.pm_table_size pmt.min_size)
	 ,(lprint :msg "pm table larger than the table returned by smu")
	 (exit 0))
       (let ((sysinfo (system_info))))
       (setf sysinfo.enabled_cores_count pmt.max_cores
	     sysinfo.cpu_name (get_processor_name)
	     sysinfo.codename (smu_codename_to_str &obj)
	     sysinfo.smu_fw_ver (smu_get_fw_version &obj))
       (when (== (hex #x400005)
		 obj.pm_table_version)
	 (when (== SMU_Return_OK
		   (smu_read_pm_table &obj pm_buf obj.pm_table_size))
	   ,(lprint :msg "PMT hack for cezanne's core_disabled_map")
	   (disabled_cores_0x400005 &pmt &sysinfo)))
       (get_processor_topology &sysinfo
			       pmt.zen_version)
       (case obj.smu_if_version
	 ,@(loop for i from 9 upto 13 collect
		 `(,(format nil "IF_VERSION_~a" i)
		   (setf sysinfo.if_ver ,i)))
	 (t (setf sysinfo.if_ver 0)))

       (return (std--make_tuple sysinfo pm_buf pmt))
       
       )

     "#define pmta(elem) ((pmt.elem)?(*(pmt.elem)):std::nanf(\"1\"))"
     "#define pmta0(elem) ((pmt.elem)?(*(pmt.elem)):0F)"
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))

       #+nil (let ((update_time_s 1)
		   (show_disabled_cores 0)))

       (when (logand (!= 0 (getuid))
		     (!= 0 (geteuid)))
	 ,(lprint :msg "Program must be run as root")
	 (return 1))

       (let ((ret (static_cast<smu_return_val> (smu_init &obj))))
					;(declare (type smu_return_val ret))
	 (unless (== SMU_Return_OK ret)
	   ,(lprint :msg "error"
		    :vars `((smu_return_to_str ret)))
	   (return 1)))

       #+nil(let ((force 0))
	      (start_pm_monitor force))
       
       
       (do0
	(glfwSetErrorCallback glfw_error_callback)
	(unless (glfwInit)
	  ,(lprint :msg "glfwInit failed")
	  (return 1))
	(let ((glsl_version (string "#version 130")))
	  (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 4)
	  (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 6))
	(let ((window (glfwCreateWindow 1280 720
					(string "ryzen_mon_glgui")
					nullptr
					nullptr)))
	  (when (== nullptr window)
	    ,(lprint :msg "Can't open glfw window")
	    (return 1))
	  (glfwMakeContextCurrent window)
	  (let ((vsyncOn true)))
	  (glfwSwapInterval (? vsyncOn 1 0))
	  (IMGUI_CHECKVERSION)
	  (ImGui--CreateContext)
	  (ImPlot--CreateContext)
	  (let ((io (ImGui--GetIO)))
	    (setf io.ConfigFlags
		  (or io.ConfigFlags
		      ImGuiConfigFlags_NavEnableKeyboard)))
	  (ImGui--StyleColorsDark)
	  (ImGui_ImplGlfw_InitForOpenGL window true)
	  (ImGui_ImplOpenGL3_Init glsl_version))
	)

       
       ,(let ((l-store `(coreFrequency corePower coreTemperature)))
	  `(do0

	    (let (((bracket sysinfo pm_buf pmt) (start_pm_monitor2))))
	
	    (let ((show_demo_window true)
		  (clear_color (ImVec4 .4s0 .5s0 .6s0 1s0)))
	      
	  
	      (let (#+q(coreColors (std--vector<ImVec4>
				 (curly
				  ,@(loop for (r g b name) in `((1 0 0 red)
								(0 1 0 green)
								(0 0 1 blue)
								(1 1 0 yellow) 
								(1 0 1 magenta)
								(0 1 1 cyan)
								(.5 .5 .5 gray)
								(1 .5 0 orange))
					  collect
					  `(ImVec4 ,(coerce r 'single-float)
						   ,(coerce g 'single-float)
						   ,(coerce b 'single-float)
						   1s0)))))
		    (maxDataPoints 1024
				   )
		    #+q(x (std--vector<float> maxDataPoints))
		    #+q(y (std--vector<float> maxDataPoints))
		    #+q(timePoints (std--deque<float>))
		    #+q,@(loop for e in l-store
			    collect
			    `(,e (std--vector<std--deque<float>> pmt.max_cores)))
		    (startTime (std--chrono--steady_clock--now))))

	      (let ((diagramVoltage (DiagramWithGui #+color (curly
				  ,@(loop for (r g b name) in `((1 0 0 red)
								(0 1 0 green)
								(0 0 1 blue)
								(1 1 0 yellow) 
								(1 0 1 magenta)
								(0 1 1 cyan)
								(.5 .5 .5 gray)
								(1 .5 0 orange))
					  collect
					  `(Color ,(coerce r 'single-float)
						   ,(coerce g 'single-float)
						   ,(coerce b 'single-float)
						   )))
						    8
						    maxDataPoints))))
	      
	      (let ((affinityManager (CpuAffinityManagerWithGui (getpid)))))
	      
	      (while (!glfwWindowShouldClose window)
		     (glfwPollEvents)
		     (ImGui_ImplOpenGL3_NewFrame)
		     (ImGui_ImplGlfw_NewFrame)
		     (ImGui--NewFrame)
		     (when show_demo_window
		       (ImGui--ShowDemoWindow &show_demo_window)
		       (ImPlot--ShowDemoWindow))

		     (do0
		      (when (== SMU_Return_OK
				(smu_read_pm_table &obj pm_buf obj.pm_table_size))
			(when sysinfo.available
			  (ImGui--Begin (string "Ryzen"))

			  (affinityManager.RenderGui)
			  (when (ImGui--Checkbox (string "vsync")
						 &vsyncOn)
			    (glfwSwapInterval (? vsyncOn 1 0)))

			  ,@(loop for var in `(cpu_name codename cores ccds ccxs
							;; fixme: different for older ryzen
							cores_per_ccx smu_fw_ver if_ver)
				  collect
				  `(ImGui--Text
				    (string "%s")
				    (dot (std--format
					  (string ,(format nil "~a='{}'"
							   var))
					  (dot sysinfo ,var))
					 (c_str))))

			  (let ((package_sleep_time 0s0)
				(average_voltage 0s0)))
		      
			  (if pmt.PC6
			      (setf package_sleep_time (/ (pmta PC6) 100s0)
				    average_voltage (/ (- (pmta CPU_TELEMETRY_VOLTAGE)
							  (* .2s0 package_sleep_time))
						       (- 1s0 package_sleep_time)))
			      (setf average_voltage (pmta CPU_TELEMETRY_VOLTAGE)))

			  (let ((currentTime (std--chrono--steady_clock--now))
				(elapsedTime (dot (std--chrono--duration<float> (- currentTime startTime))
						  (count)))))

			  #+nil (do0
			   (comments "If the deque has reached its maximum size, remove the oldest data point  ")
			   (when (<= maxDataPoints (timePoints.size))
			     (timePoints.pop_front)
			     (comments "Also remove the oldest data point from each core's frequency and power data  ")
			     ,@(loop for e in l-store
				     collect
				     `(for-range (deq ,e)
						 (declare (type "auto&" deq))
						 (unless (deq.empty)
						   (deq.pop_front))))))
			  #+nil (do0
			   (comments "Add the new timepoint")
			   (timePoints.push_back elapsedTime))
			
			  (let ((voltageValues (std--vector<float>  pmt.max_cores
						))))
			  (dotimes (i pmt.max_cores)
			    (let ((core_disabled (and (>> sysinfo.core_disable_map i) 1))
				  (core_frequency (* (pmta (aref CORE_FREQEFF i))
						     1000s0))
				  (core_voltage_true (pmta (aref CORE_VOLTAGE i)))
				  (core_sleep_time (/ (pmta (aref CORE_CC6 i))
						      100s0))
				  (core_voltage (+ (* (- 1s0 core_sleep_time)
						      average_voltage)
						   (* .2 core_sleep_time)))
				  (core_temperature (pmta (aref CORE_TEMP i))))
			      
			      (do0
			       ;(voltageValues.push_back core_voltage_true)
			       ;#+nil
			       (setf (aref voltageValues i) core_voltage_true))

			      #+nil(do0
			       (comments "Update the frequency, power and temperature data for each core  ")
			       ,@(loop for e in l-store and f in `(core_frequency core_voltage core_temperature)
				       collect
				       `(dot (aref ,e i) (push_back ,f) )))
			    
			      (if core_disabled
				  (ImGui--Text (string "%s")
					       (dot (std--format (string "{:2} Disabled")
								 i )
						    (c_str)))
				  (if (<= 6s0 (pmta (aref CORE_C0 i)))
				      (do0
				       (ImGui--Text (string "%s")
						
						    (dot (std--format
							  (string "{:2} Sleeping   {:6.3f}W {:5.3f}V {:5.3f}V {:6.2f}C C0: {:5.1f}% C1: {:5.1f}% C6: {:5.1f}%")
							  i (pmta (aref CORE_POWER i))
							  core_voltage core_voltage_true (pmta (aref CORE_TEMP i))
							  (pmta (aref CORE_C0 i))
							  (pmta (aref CORE_CC1 i))
							  (pmta (aref CORE_CC6 i)))
							 (c_str))))
				      (do0
				       (ImGui--Text (string "%s")
						
						    (dot (std--format
							  (string "{:2} {:7.1f}MHz {:6.3f}W {:5.3f}V {:5.3f}V {:6.2f}C C0: {:5.1f}% C1: {:5.1f}% C6: {:5.1f}%")
						      
							  i core_frequency
							  (pmta (aref CORE_POWER i))
							  core_voltage core_voltage_true (pmta (aref CORE_TEMP i))
							  (pmta (aref CORE_C0 i))
							  (pmta (aref CORE_CC1 i))
							  (pmta (aref CORE_CC6 i)))
							 (c_str))))
				      ))))
			  (diagramVoltage.AddDataPoint elapsedTime
						       voltageValues)
			  (diagramVoltage.RenderGui)
			  #+nil 
			  ,@(loop for e in l-store
				  collect
				  `(when (ImPlot--BeginPlot (string ,e))
				     (dotimes (i pmt.max_cores)
					;(ImPlot--SetNextLineStyle (aref coreColors i))
				       (x.assign (timePoints.begin)
						 (timePoints.end))
				       (y.assign (dot (aref ,e i) (begin))
						 (dot (aref ,e i) (end)))
				       #+nil (let ((x (std--vector<float> (curly (timePoints.begin)
										 (timePoints.end))))
						   (y (std--vector<float> (curly (dot (aref ,e i) (begin))
										 (dot (aref ,e i) (end)))))))
				       #+nil(let ((x_min (deref (std--min_element (timePoints.begin)
										  (timePoints.end))))
						  (x_max (deref (std--max_element (timePoints.begin)
										  (timePoints.end)))))
					      (ImPlot--SetNextAxisLimits x_min x_max ImGuiCond_Always))
				       (ImPlot--SetupAxes (string "X") (string "Y") ImPlotAxisFlags_AutoFit
							  ImPlotAxisFlags_AutoFit)
				       (ImPlot--PlotLine (dot (std--format (string "Core {:2}")
									   i )
							      (c_str))
							 (x.data)
							 (y.data)
							 (y.size)))
				     (ImPlot--EndPlot)))
			  (ImGui--End))))
		 
		     (ImGui--Render)
		     (let ((w 0)
			   (h 0))
		       (glfwGetFramebufferSize window &w &h)
		       (glViewport 0 0 w h)
		       (glClearColor (* clear_color.x clear_color.w)
				     (* clear_color.y clear_color.w)
				     (* clear_color.z clear_color.w)
				     clear_color.w)
		       (glClear GL_COLOR_BUFFER_BIT)
		       (ImGui_ImplOpenGL3_RenderDrawData
			(ImGui--GetDrawData))
		       (glfwSwapBuffers window))))))
       (do0
	(ImGui_ImplOpenGL3_Shutdown)
	(ImGui_ImplGlfw_Shutdown)
	(ImPlot--DestroyContext)
	(ImGui--DestroyContext)
	(glfwDestroyWindow window)
	(glfwTerminate)
	(return 0))
       ))
   :omit-parens t
   :format t
   :tidy nil))

