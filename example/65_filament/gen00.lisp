(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     ;(ql:quickload "cl-ppcre")
     ;(ql:quickload "cl-change-case")
     ) 
(in-package :cl-cpp-generator2)
(progn
  (defparameter *source-dir* #P"example/65_filament/source/")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  #+nil (defun logprint (msg &optional rest)
      `(progn				;do0
	 " "
	 #-nolog
	 (do0 ;let
	  #+nil ((lock (std--unique_lock<std--mutex> ,(g `_stdout_mutex)))
	   )
	  
	  (do0
					;("std::setprecision" 3)
	   (<< "std::cout"
	       ;;"std::endl"
	       ("std::setw" 10)
	       (dot ("std::chrono::high_resolution_clock::now")
		    (time_since_epoch)
		    (count))
					;,(g `_start_time)
	       
	       (string " ")
	       ("std::this_thread::get_id")
	       (string " ")
	       __FILE__
	       (string ":")
	       __LINE__
	       (string " ")
	       __func__
	       (string " ")
	       (string ,msg)
	       (string " ")
	       ,@(loop for e in rest appending
		       `(("std::setw" 8)
					;("std::width" 8)
			 (string ,(format nil " ~a='" (emit-c :code e)))
			 ,e
			 (string "'")))
	       "std::endl"
	       "std::flush")))))
  (let ((type-definitions
	  `(do0
	    
	    (defclass App ()
	      "public:"
	      ,@(loop for (e f) in `((filament--VertexBuffer* vb)
				     (filament--IndexBuffer* ib)
				     (filament--Material* mat)
				     (filament--Camera* cam)
				     (utils--Entity camera)
				     (filament--Skybox* skybox)
				     (utils--Entity renderable))
		      collect
		      (format nil "~a ~a;" (emit-c :code  e) f)))
	    (defclass Vertex ()
	      "public:"
	      ,@(loop for (e f) in `(("filament::math::float2" position)
				     (uint32_t color))
		      collect
		      (format nil "~a ~a;" e f)))
	    )))

    (let ((fn-h (asdf:system-relative-pathname
			  'cl-cpp-generator2
			  (merge-pathnames #P"star_tracker.h"
					   *source-dir*))))
     (with-open-file (sh fn-h
			 :direction :output
			 :if-exists :supersede
			 :if-does-not-exist :create)
       (emit-c :code
	       `(do0
		 "#pragma once"
		 (include 
			  <iostream>
			  <chrono>
			  <thread>)
		 (include ,@(loop for e in `(Camera
					     Engine
					     IndexBuffer
					     Material
					     MaterialInstance
					     RenderableManager
					     Scene
					     Skybox
					     TransformManager
					     VertexBuffer
					     View)
				  collect
				  (format nil "<filament/~a.h>" e))
			  <utils/EntityManager.h>
			  #+nil,@(loop for e in `(Config
					     FilamentApp)
				  collect
				  (format nil "<filamentapp/~a.h>" e)))
		 #+nil(do0 "using namespace filament;"
		      "using utils::Entity;"
		      "using utils::EntityManager;")
		 ,type-definitions)
	       :hook-defun #'(lambda (str)
                               (format sh "~a~%" str))
               :hook-defclass #'(lambda (str)
                                  (format sh "~a;~%" str))
	       :header-only t))
     (sb-ext:run-program "/usr/bin/clang-format"
                         (list "-i"  (namestring fn-h))))
    
    (write-source (asdf:system-relative-pathname
		  'cl-cpp-generator2
		  (merge-pathnames #P"star_tracker.cpp"
				   *source-dir*))
		  
		 `(do0
		   (include "star_tracker.h")
		   (do0 "using namespace filament;"
		 "using utils::Entity;"
		 "using utils::EntityManager;")
		   ;,type-definitions
		   (let ((triangle_vertices (curly (curly (curly 1 0) "0xffff0000u")
						   (curly (curly ,(cos (* 2 (/ pi 3)))
								 ,(sin (* 2 (/ pi 3))))
							  "0xff00ff00u")
						   (curly (curly ,(cos (* 4 (/ pi 3)))
								 ,(sin (* 4 (/ pi 3))))
							  "0xff0000ffu")))
			 (triangle_indices (curly 0 1 2)))
		     (declare (type (array "static const Vertex" 3)
				    triangle_vertices)
			      (type (array "static constexpr uint16_t" 3)
				    triangle_indices)))
		  
		   (defun main (argc argv)
		     (declare (type int argc)
			      (type char** argv)
			      (values int))
		     (let (;(config)
			   (app))
		       (declare (type Config config)
				(type App app))
					;(setf config.title (string "hello triangle"))

		       (let ((setup (lambda (engine view scene)
				      (declare (type Engine* engine)
					       (type View* view)
					       (type Scene* scene)
					       (capture &app))
				      (setf app.skybox
					    (dot (Skybox--Builder)
						 (color (curly .1 .125 .25 1.0))
						 (build *engine)))
				      (scene->setSkybox app.skybox)
				      (view->setPostProcessingEnabled false)
				      (static_assert (== 12 (sizeof Vertex))
						     (string "strange vertex size"))))))
		       )
		     (return 0)
		     )
		   )))
  #+nil
  (with-open-file (s "source/CMakeLists.txt" :direction :output
					     :if-exists :supersede
					     :if-does-not-exist :create)
    (macrolet ((out (fmt &rest rest)
		 `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
      (out "cmake_minimum_required( VERSION 3.4 )")
      (out "project( mytest LANGUAGES CXX )")
      (out "set( CMAKE_CXX_COMPILER clang++ )")
      (out "set( CMAKE_CXX_FLAGS \"\"  )")
      (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
      (out "set( CMAKE_CXX_STANDARD 17 )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")
      
					;(out "set( CMAKE_CXX_FLAGS )")
      (out "find_package( OpenCV REQUIRED )")
      (out "set( SRCS ~{~a~^~%~} )"
	   (directory "source/*.cpp"))
      (out "add_executable( mytest ${SRCS} )")
      ;(out "target_include_directories( mytest PUBLIC /home/martin/stage/cl-cpp-generator2/example/58_stdpar/source/ )")
      
      (out "target_link_libraries( mytest ${OpenCV_LIBS} )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
      )
    ))

 

