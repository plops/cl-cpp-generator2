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
	    (defclass StarEntry ()
	      "public:"
	      "float magnitude, ra, dec;")
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
			  <fstream>
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
			  #-nil,@(loop for e in `(Config
						  FilamentApp)
				  collect
				  (format nil "<filamentapp/~a.h>" e)))
		 (include "generated/resources.h"
			  )
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
		   (include <imgui.h>)
		   (do0 "using namespace filament;"
		 "using utils::Entity;"
		 "using utils::EntityManager;")
		   ;,type-definitions
		   
		   ,(let ((n ;4995
			    118218
			   ))
		   `(defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (do0
			 (let ((file (std--ifstream (string ,(format nil "/home/martin/stage/cl-cpp-generator2/example/65_filament/script/out_~ax3_float32.raw" n))
						    (logior std--ios--in
							    std--ios--binary
					;std--ios--trunc
							    )))
			       (size ,n)
			       (star_data))
			   (declare	;(type std--ifstream file)
			    (type std--vector<StarEntry> star_data))
			   #+nil (file.open (string "/home/martin/stage/cl-cpp-generator2/example/65_filament/script/out_4995x3_float32.raw")
					    (logior std--ios--in
						    std--ios--binary
						    std--ios--trunc))
			   (star_data.resize size)
			   (file.read (cast char* (star_data.data))
				      (* size (sizeof StarEntry ;star_data.value_type
						      )))
			   (file.close))

			 (let ((triangle_vertices #+nil (curly (curly (curly 1 0) "0xffff0000u")
							 (curly (curly ,(cos (* 2 (/ pi 3)))
								       ,(sin (* 2 (/ pi 3))))
								"0xff00ff00u")
							 (curly (curly ,(cos (* 4 (/ pi 3)))
								       ,(sin (* 4 (/ pi 3))))
								"0xff0000ffu")))
			       (triangle_indices ;(curly 0 1 2)
				 ))
			   (declare (type (array "static Vertex" ,n)
					  triangle_vertices)
				    (type (array "static uint16_t" ,n)
					  triangle_indices))
			   (dotimes (i size)
			     (setf (aref triangle_indices i) i))
			   (dotimes (i size)
			     (setf (dot (aref triangle_vertices i)
					position
					x)
				   (* 1 (/ (- (dot (aref star_data i)
					       ra)
					  180)
				       180))
				   )
			     (setf (dot (aref triangle_vertices i)
					position
					y)
				   (* 2 (/ (dot (aref star_data i)
					    dec)
				       90))
				   )
			     (let ((m (static_cast<int> (* (/ 256 16) (- 16 (+ 2 (dot (aref star_data i)
										    magnitude)))))))
			       (setf (dot (aref triangle_vertices i)
					  color)
				     (logior (<< #xff (* 3 8))
					     (<< #xff (* 2 8))
					     (<< m (* 1 8))
					     (<< #xff (* 0 8))
					))))))

		     
		      (let ((config)
			    (app))
			(declare (type Config config)
				 (type App app))
			(setf config.title (string "hello triangle"))
			(setf config.backend Engine--Backend--VULKAN)
					;(setf config.backend Engine--Backend--OPENGL)
			(let ((setup
				(lambda (engine view scene)
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
						 (string "strange vertex size"))
				  (setf app.vb
					(dot (VertexBuffer--Builder)
					     (vertexCount ,n)
					     (bufferCount 1)
					     (attribute VertexAttribute--POSITION
							0
							VertexBuffer--AttributeType--FLOAT2 0 12)
					     (attribute VertexAttribute--COLOR
							0
							VertexBuffer--AttributeType--UBYTE4 8 12)
					     (normalized VertexAttribute--COLOR)
					     (build *engine)))
				  (app.vb->setBufferAt *engine 0
						       (VertexBuffer--BufferDescriptor
							triangle_vertices (* (sizeof float) 2 ,n) ; 36
							nullptr))
				  (setf app.ib (dot (IndexBuffer--Builder)
						    (indexCount ,n) ;3
						    (bufferType IndexBuffer--IndexType--USHORT)
						    (build *engine)))
				  (app.ib->setBuffer *engine
						     (IndexBuffer--BufferDescriptor
						      triangle_indices (* (sizeof uint16_t) ,n) ;6
						      nullptr
						      )
						     )
				  (setf app.mat
					(dot (Material--Builder)
					     (package RESOURCES_BAKEDCOLOR_DATA
						      RESOURCES_BAKEDCOLOR_SIZE
						      )
					     (build *engine)))
				  (setf app.renderable (dot (EntityManager--get)
							    (create)))
				  (dot (RenderableManager--Builder 1)
				       (boundingBox (curly (curly -1 -1 -1)
							   (curly 1   1  1)))
				       (material 0 (app.mat->getDefaultInstance))
				       (geometry 0
						 RenderableManager--PrimitiveType--POINTS ;TRIANGLES
						 app.vb app.ib 0 ,n) ;
				        (culling false)
				       (receiveShadows false)
				       (castShadows false)
				       (build *engine app.renderable))
				  (scene->addEntity app.renderable)
				  (setf app.camera (dot (utils--EntityManager--get)
							(create)))
				  (setf app.cam (engine->createCamera app.camera))
				  (view->setCamera app.cam)
				  ))
			      (cleanup
				(lambda (engine view scene)
				  (declare (type Engine* engine)
					   (type View* view)
					   (type Scene* scene)
					   (capture &app))
				  ,@(loop for e in `(skybox
						     renderable
						     mat
						     vb
						     ib)
					  collect
					  `(engine->destroy (dot app ,e)))
				  (engine->destroyCameraComponent app.camera)
				  (dot (utils--EntityManager--get)
				       (destroy app.camera))))
			      (gui
				(lambda (engine view)
				  (declare (type Engine* engine)
					   (type View* view)
					   (capture &app))
				  (ImGui--Begin (string "Parameters"))
				  (progn
				    (let ((gain 1s0))
				      (declare (type "static float" gain))
				      (ImGui--SliderFloat (string "gain")
							  &gain
							  0s0 1s0)))
				  (ImGui--End)))
			      )

			  (let ((filament_app (FilamentApp--get)))
			    (declare (type "static FilamentApp&" filament_app))
			    (dot filament_app
				 (animate
				  (lambda (engine view now)
				    (declare (type Engine* engine)
					     (type View* view)
					     (type double now)
					     (capture &app))
				    (let ((zoom 1.5f)
					  (w (dot (view->getViewport)
						  width))
					  (h (dot (view->getViewport)
						  height))
					  (aspect (/ (static_cast<float> w) h))
					  )
				      (app.cam->setProjection
				       Camera--Projection--ORTHO
				       (* -1 aspect zoom)
				       (* aspect zoom)
				       (* -1 zoom)
				       zoom
				       0 1
				       )
				      (let ((&tcm (engine->getTransformManager)))
					(tcm.setTransform
					 (tcm.getInstance app.renderable)
					 (filament--math--mat4f--rotation
					  now
					  (space filament--math--float3 (curly 0 0 -1)))))))
				  ))
			   
			    (dot filament_app
				 (run config setup cleanup
				      gui
				      ))
			    )))
		      (return 0)
		      ))
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

 

