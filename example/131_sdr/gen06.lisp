(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(setf *features* (set-difference *features* (list :more)))
(setf *features* (set-exclusive-or *features* (list :more
						   )))


(progn
  (defparameter *source-dir* #P"example/131_sdr/source06/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  (defparameter *benchmark-counter* 0)
  (defun benchmark (code)
    (let ((start (format nil "startBenchmark~2,'0d" *benchmark-counter*))
	  (end (format nil "endBenchmark~2,'0d" *benchmark-counter*))
	  (elapsed (format nil "elapsed~2,'0d" *benchmark-counter*))
	  (elapsed_ms (format nil "elapsed_ms~2,'0d" *benchmark-counter*)))
      (incf *benchmark-counter*)
      `(let ((,start (std--chrono--high_resolution_clock--now)))
	 ,code
	 (let ((,end (std--chrono--high_resolution_clock--now))
	       (,elapsed (std--chrono--duration<double> (- ,end ,start)))
	       (,elapsed_ms (* 1000 (dot ,elapsed (count))))
	       )
	   ,(lprint :msg (format nil "benchmark ~2,'0d" (- *benchmark-counter* 1))
		    :vars `(,elapsed_ms))))))

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
					;queue
      ;deque
					;chrono

      ;filesystem
					;unistd.h
      ;cstdlib

      cmath

					;omp.h
      ;unistd.h
      array
      complex
      exception
      ;memory
      stdexcept
      ;string
      ;utility
      liquid/liquid.h
      )
     (include
      
      implot.h
      ;imgui.h
      imgui_impl_glfw.h
      imgui_impl_opengl3.h
      GLFW/glfw3.h
      )
     
     

     (defun glfw_error_callback (err desc)
       (declare (type int err)
		(type "const char*" desc))
       ,(lprint :msg "GLFW error:"
		:vars `(err desc)))


     (let ((Sim (lambda ()
		  (let ((phase_offset .8)
			(frequency_offset .01)
			(n 40)
			(x (std--complex<float> 0.0))
			(y (std--complex<float> 0.0))
			(phase_error 0.0)
			(phi_hat .0)))
		  (let ((wn .1)
			(zeta .707)
			(K 1000.0)
			(a (std--vector<float> 3))
			(b (std--vector<float> 3))
			)
		    (iirdes_pll_active_lag wn zeta K (b.data) (a.data)))
		  
		  (dotimes (i n)
		    (setf x (std--exp (* (std--complex<float> 0d0 1d0)
					 (+ phase_offset (* i frequency_offset)))))
		    (setf y (std--exp (* (std--complex<float> 0d0 1d0)
					 phi_hat)))
		    (setf phase_error (std--arg (* x (std--conj y))))
		    ,(lprint :msg "result"
			     :vars `(i phi_hat phase_error))
		    )))))
     
     (let ((DrawPlot (lambda ()
		       (handler-case
			   (do0
			    (ImGui--Text (string "hello"))
			    (Sim))       
			 ("const std::exception&" (e)
			   (ImGui--Text (string "Error while processing signal: %s")
					(e.what))))))))

     (let ((initGL (lambda ()
		     (do0 (glfwSetErrorCallback glfw_error_callback)
			  (when (== 0 (glfwInit))
			    ,(lprint :msg "glfw init failed"))
			  
			  #+more
			  (do0		 ;let ((glsl_version (string "#version 130")))
			   (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 3)
			   (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0))
			  (let ((*window (glfwCreateWindow 800 600
							   (string "imgui_dsp")
							   nullptr nullptr)))
			    (when (== nullptr window)
			      ,(lprint :msg "failed to create glfw window")
			      )
			    (glfwMakeContextCurrent window)
			    ,(lprint :msg "enable vsync")
			    (glfwSwapInterval 1)
			    (IMGUI_CHECKVERSION)
			    ,(lprint :msg "create imgui context")
			    (ImGui--CreateContext)
			    (ImPlot--CreateContext)

			    (let ((&io (ImGui--GetIO)))
			      (setf io.ConfigFlags (or io.ConfigFlags
						       ImGuiConfigFlags_NavEnableKeyboard)))
					;(ImGui--StyleColorsDark)
			    (ImGui_ImplGlfw_InitForOpenGL window true)
			    (ImGui_ImplOpenGL3_Init (string "#version 130"))
			    (glClearColor  0 0 0 1)
			    ))
		     (return window)
		     ))))

     
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       (Sim) 
       #+Nil
       (handler-case
	   (do0
	    (let ((*window (initGL))))
	    
	    (while (!glfwWindowShouldClose window)
		   (glfwPollEvents)
		   (ImGui_ImplOpenGL3_NewFrame)
		   (ImGui_ImplGlfw_NewFrame)
		   (ImGui--NewFrame)
		   (DrawPlot)
		   (ImGui--Render)
		   (let ((w 0)
			 (h 0))
		     (glfwGetFramebufferSize window &w &h)
		     (glViewport 0 0 w h)
		     (glClear GL_COLOR_BUFFER_BIT))
		   (ImGui_ImplOpenGL3_RenderDrawData (ImGui--GetDrawData))
		   (glfwSwapBuffers window))
	    
	    (do0
	     (ImGui_ImplOpenGL3_Shutdown)
	     (ImGui_ImplGlfw_Shutdown)
	     (ImPlot--DestroyContext)
	     (ImGui--DestroyContext)
	     (glfwDestroyWindow window)
	     (glfwTerminate)))
	 
	 ("const std::runtime_error&" (e)
	   ,(lprint :msg "error 1422:"
		    :vars `((e.what)))
	   (return -1))
	 ("const std::exception&" (e)
	   ,(lprint :msg "error 1426:"
		    :vars `((e.what)))
	   (return -1)))
       
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil))
