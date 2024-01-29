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
	      )
     (include<> GLFW/glfw3.h
		format
		iostream
		unistd.h
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
	  (glfwSwapInterval 1)
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

       

       (do0

	(let (((bracket sysinfo pm_buf pmt) (start_pm_monitor2))))
	
	(let ((show_demo_window true)
	      (clear_color (ImVec4 .4s0 .5s0 .6s0 1s0)))
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
		      
		      
		      (dotimes (i pmt.max_cores)
			(let ((core_disabled (and (>> sysinfo.core_disable_map i) 1))
				(core_frequency (* (pmta (aref CORE_FREQEFF i))
						  1000s0))
			       (core_voltage_true (pmta (aref CORE_VOLTAGE i)))
			       (core_sleep_time (/ (pmta (aref CORE_CC6 i))
						   100s0))
			       (core_voltage (+ (* (- 1s0 core_sleep_time)
						   average_voltage)
						(* .2 core_sleep_time))))

			    (unless core_disabled
			      (if (<= 6s0 (pmta (aref CORE_C0 i)))
				  (do0
				   (ImGui--Text (string "%s")
						
						(dot (std--format
						      (string "{:2} Sleep {:6.3f}W {:5.3f}V {:6.2f}C C0: {:5.1f}% C1: {:5.1f}% C6: {:5.1f}%")
						      i (pmta (aref CORE_POWER i))
						      core_voltage (pmta (aref CORE_TEMP i))
						      (pmta (aref CORE_C0 i))
						      (pmta (aref CORE_CC1 i))
						      (pmta (aref CORE_CC6 i)))
						     (c_str))))
				  (do0
				   (ImGui--Text (string "%s")
						
						(dot (std--format
						      (string "{:2} {:7.1f}MHz {:6.3f}W {:5.3f}V {:6.2f}C C0: {:5.1f}% C1: {:5.1f}% C6: {:5.1f}%")
						      
						      i core_frequency
						      (pmta (aref CORE_POWER i))
						      core_voltage (pmta (aref CORE_TEMP i))
						      (pmta (aref CORE_C0 i))
						      (pmta (aref CORE_CC1 i))
						      (pmta (aref CORE_CC6 i)))
						     (c_str))))
				  ))))
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
		   (glfwSwapBuffers window)))))
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

