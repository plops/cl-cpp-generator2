(declaim (optimize 
	  (safety 3)
	  (speed 0)
	  (debug 3)))

(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)



(setf *features* (union *features* `()))

(setf *features* (set-difference *features*
				 '()))

(progn
  (defparameter *source-dir* #P"example/41_gtk3_gl_thread/source/")
  
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  
  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " ")
    (defparameter *global-code* nil)
    (defun emit-global (&key code)
      (push code *global-code*)
      " "))
  (progn
    
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)
    (defun logprint (msg &optional rest)
      `(do0
	" "
	#-nolog
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
	     "std::flush"))))
    (defun emit-globals (&key init)
      (let ((l `((_start_time ,(emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
							   (time_since_epoch)
							   (count)))))
		 ,@(loop for e in *module-global-parameters* collect
			(destructuring-bind (&key name type default)
			    e
			  (declare (ignorable default))
			  `(,name ,type))))))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(declare (ignorable type))
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      (include <chrono>)
	      (defstruct0 State
		  ,@(loop for e in l collect
 			 (destructuring-bind (name type &optional value) e
			   (declare (ignorable value))
			   `(,name ,type))))))))
    (defun define-module (args)
      "each module will be written into a c file with module-name. the global-parameters the module will write to will be specified with their type in global-parameters. a file global.h will be written that contains the parameters that were defined in all modules. global parameters that are accessed read-only or have already been specified in another module need not occur in this list (but can). the prototypes of functions that are specified in a module are collected in functions.h. i think i can (ab)use gcc's warnings -Wmissing-declarations to generate this header. i split the code this way to reduce the amount of code that needs to be recompiled during iterative/interactive development. if the module-name contains vulkan, include vulkan headers. if it contains glfw, include glfw headers."
      (destructuring-bind (module-name global-parameters module-code) args
	(let ((header ()))
	  #+nil (format t "generate ~a~%" module-name)
	  (push `(do0
		  " "
		  (include "utils.h")
		  " "
		  (include "globals.h")
		  " "
		  ;(include "proto2.h")
		  " ")
		header)
	  (unless (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
	    (push `(do0 "extern State state;")
		  header))
	  (push `(:name ,module-name :code (do0 ,@(reverse header) ,module-code))
		*module*))
	(loop for par in global-parameters do
	     (destructuring-bind (parameter-name
				  &key (direction 'in)
				  (type 'int)
				  (default nil)) par
	       (declare (ignorable direction))
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))

  (let*  ()
    (define-module
       `(base ((_main_version :type "std::string")
		    (_code_repository :type "std::string")
		    (_code_generation_time :type "std::string")
		 )
	      (do0
	    
	    
		    (include <iostream>
			     <chrono>
			     <thread>
			     
			     )

		    ;(include <gtkmm.h>)
		    " "

		    
		    ;; gtk-demo/example_listview_applauncher.cc
		    (split-header-and-code
		     (do0
		      "// header"
		       
		      #+nil (include ;<gtkmm/button.h>
		       <gtkmm/window.h>
		       <gtkmm/grid.h>)
		      (include <gtkmm.h>
			       ;<gtkmm/widget.h>
			       ;<gtkmm/cssprovider.h>
			       ;<gtkmm/styleproperty.h>
			       )
		      (include <epoxy/gl.h>
			       <gtk/gtk.h>
			       <gdk/gdkx.h>
			       <GL/gl.h>)
		      #+nil (include <glibmm/extraclassinit.h>
			       <glibmm/ustring.h>)
		      #+nil (include <gdkmm/general.h>)
		      " "
		      )
		     (do0
		      "// implementation"
		      (include "vis_00_base.hpp")
		      " "
		      ))
		    

		    (defclass GraphicsArea "public Gtk::Window"
		      "public:"
		      (defmethod GraphicsArea ()
			(declare (values :constructor))
			(set_title (string "graphics-area"))
			(set_default_size 640 360)
			(add vbox)
			(area.set_hexpand true)
			(area.set_vexpand true)
			(area.set_auto_render true)
			(vbox.add area)
			(dot area
			     (signal_render)
			     (connect
			      (sigc--mem_fun *this
					     &GraphicsArea--render)))
			(area.show)
			(vbox.show)
			)
		      (defmethod ~GraphicsArea ()
			(declare (virtual)
				 (values :constructor)))
		      (defmethod run ()
			(while true
			  (dispatcher.emit)
			  (std--this_thread--sleep_for
			   (std--chrono--milliseconds 50))))
		      (defmethod onNotifcationFromThread ()
			,(logprint "" `())
			(queue_draw))
		      (defmethod render (ctx)
			(declare (type "const Glib::RefPtr<Gdk::GLContext>&" ctx)
				 (values bool)
				 (virtual))
			(area.throw_if_error)
			(glClearColor .5 .5 .5 1.0)
			(glClear GL_COLOR_BUFFER_BIT)
			(do0
			 (glColor3f .8 .9 .7)
			 (let ((a .1s0))
			   (declare (type "static float" a))
			   (incf a .1s0)
			  (glTranslatef a 0 .0))
			 (glBegin GL_QUADS)


			 ,@(loop for (x y) in `((0 -1)
						(0 1)
						(.5 1)
						(.5 -1))
			      collect
				`(glVertex2f ,(coerce x 'single-float)
					     ,(coerce y 'single-float)))
			 (glEnd))
			(glFlush)
			(return true))
		      "public:"
		      "Gtk::GLArea area;"
		      "Gtk::Box vbox{Gtk::ORIENTATION_VERTICAL,false};"
		      "private:"
		      "Glib::Dispatcher dispatcher;")
		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      ,(logprint "start" `(argc (aref argv 0)))
		      (let ((app (Gtk--Application--create
				  argc argv
				  (string "gtk3-gl-threads")))
			    (hw)
			    
			    )
			(declare (type GraphicsArea hw))
			"std::thread th(&GraphicsArea::run, &hw);"
			(app->run hw)
			)))))
  )
  
  (progn
    (progn ;with-open-file
      #+nil (s (asdf:system-relative-pathname 'cl-cpp-generator2
					(merge-pathnames #P"proto2.h"
							 *source-dir*))..
	 :direction :output
	 :if-exists :supersede
	 :if-does-not-exist :create)
      #+nil (format s "#ifndef PROTO2_H~%#define PROTO2_H~%~a~%"
		    (emit-c :code `(include <cuda_runtime.h>
					    <cuda.h>
					    <nvrtc.h>)))

      ;; include file
      ;; http://www.cplusplus.com/forum/articles/10627/
      
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       
	       (unless cuda
		 #+nil (progn (format t "emit function declarations for ~a~%" name)
			      (emit-c :code code
				      :hook-defun #'(lambda (str)
						      (format t "~a~%" str))
				      :header-only t))
		 #+nil (emit-c :code code
			 :hook-defun #'(lambda (str)
					 (format s "~a~%" str)
					 )
			 :hook-defclass #'(lambda (str)
					    (format s "~a;~%" str)
					    )
			 :header-only t
			 )
		 (let* ((file (format nil
				      "vis_~2,'0d_~a"
				      i name
				      ))
			(file-h (string-upcase (format nil "~a_H" file))))
		   (with-open-file (sh (asdf:system-relative-pathname 'cl-cpp-generator2
								      (format nil "~a/~a.hpp"
									      *source-dir* file))
				       :direction :output
				       :if-exists :supersede
				       :if-does-not-exist :create)
		     (format sh "#ifndef ~a~%" file-h)
		     (format sh "#define ~a~%" file-h)
		     
		     (emit-c :code code
			     :hook-defun #'(lambda (str)
					     (format sh "~a~%" str)
					     )
			     :hook-defclass #'(lambda (str)
						(format sh "~a;~%" str)
						)
			     :header-only t
			     )
		     (format sh "#endif")
		     ))

		 )

	       #+nil (format t "emit cpp file for ~a~%" name)
	       (write-source (asdf:system-relative-pathname
			      'cl-cpp-generator2
			      (format nil
				      "~a/vis_~2,'0d_~a.~a"
				      *source-dir* i name
				      (if cuda
					  "cu"
					  "cpp")))
			     code))))
      #+nil (format s "#endif"))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"utils.h"
				    *source-dir*))
		  `(do0
		    "#ifndef UTILS_H"
		    " "
		    "#define UTILS_H"
		    " "
		    (include <vector>
			     <array>
			     <iostream>
			     <iomanip>)
		    
		    " "
		    (do0
		     
		     " "
		     ,@(loop for e in (reverse *utils-code*) collect
			  e))
		    " "
		    "#endif"
		    " "))
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames
								     #P"globals.h"
								     *source-dir*))
		  `(do0
		    "#ifndef GLOBALS_H"
		    " "
		    "#define GLOBALS_H"
		    " "

		    #+nil (include <complex>)
		    #+nil (include <deque>
			     <map>
			     <string>)
		    #+nil (include <thread>
			     <mutex>
			     <queue>
			     <condition_variable>
			     )
		    " "

		    " "
		    ;(include "proto2.h")
		    " "
		    ,@(loop for e in (reverse *global-code*) collect
			 e)

		    " "
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))))



