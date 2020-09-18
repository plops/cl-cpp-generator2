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
  (defparameter *source-dir* #P"example/41_gtk3_popover/source/")
  
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
		       
		      (include ;<gtkmm/button.h>
		       <gtkmm/window.h>
		       <gtkmm/grid.h>)
		      (include ;<gtkmm.h>
			       <gtkmm/widget.h>
			       <gtkmm/cssprovider.h>
			       <gtkmm/styleproperty.h>
			       )
		      (include <glibmm/extraclassinit.h>
			       <glibmm/ustring.h>)
		      (include <gdkmm/general.h>)
		      " "
		      )
		     (do0
		      "// implementation"
		      (include "vis_01_application_window.hpp")
		      " "
		      ))
		    
		    (defun main (argc argv
				 )
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      ,(logprint "start" `(argc (aref argv 0)))
		      (let ((app (Gtk--Application--create ; argc argv
							  ; (string "org.gtkmm.example")
				  ))
			    (hw)
			    )
			(declare (type ExampleWindow
				  ;Gtk--Window
				  hw))
			#+nil(let ((p)
			      )
			  (declare (type PenroseWidget p))
			  ;(hw.add p)
			  )
			(app->run hw)
			)))))

    (define-module
       `(application_window ()
	      (do0
	       (include <iostream>
			<chrono>
			<thread>
			)
	       
	       (split-header-and-code
		     (do0
		      "// header"
		       
		      (include ;<gtkmm/button.h>
		       <gtkmm/window.h>
		       <gtkmm/grid.h>)
		      (include ;<gtkmm.h>
			       <gtkmm/widget.h>
			       <gtkmm/cssprovider.h>
			       <gtkmm/styleproperty.h>
			       )
		      (include <glibmm/extraclassinit.h>
			       <glibmm/ustring.h>)
		      (include <gdkmm/general.h>)
		      " "
		      )
		     (do0
		      "// implementation"
		      (include "vis_01_application_window.hpp")
		      " "
		      ))
		    (defclass ExampleWindow "public Gtk::Window"
		      "public:"
		      (defmethod ExampleWindow ()
			(declare (values :constructor)
				 )
			(set_title (string "custom widget example"))
			(set_border_width 6)
			(set_default_size 600 400)
					;(m_grid.set_margin 6)
					;(m_grid.set_row_spacing 10)
					;(m_grid.set_column_spacing 10)
			;;(Gtk--Container--add m_grid)
			;(add m_penrose)
					;(add m_grid)
			;;(this->add m_grid)
					;(m_grid.attach m_penrose 0 0)
					;(m_penrose.show)
			(show_all_children)
			
			)
		      (defmethod ~ExampleWindow ()
			(declare (values :constructor)
				 (virtual)))
		      ; "protected:"
		      #+nil (defmethod on_button_quit ()
			      (hide))
		      ;"Gtk::Grid m_grid;"
		      ;"PenroseWidget m_penrose;"
		      )

		    

		    
		    )))

    (define-module
       `(drawing_widget ()
	      (do0
	       (include <iostream>
			<chrono>
			<thread>
			)
	       
	       (split-header-and-code
		     (do0
		      "// header"
		       
		      (include ;<gtkmm/button.h>
		       <gtkmm/window.h>
		       <gtkmm/grid.h>)
		      (include ;<gtkmm.h>
			       <gtkmm/widget.h>
			       <gtkmm/cssprovider.h>
			       <gtkmm/styleproperty.h>
			       )
		      (include <glibmm/extraclassinit.h>
			       <glibmm/ustring.h>)
		      (include <gdkmm/general.h>)
		      " "
		      )
		     (do0
		      "// implementation"
		      (include "vis_02_drawing_widget.hpp")
		      " "
		      ))

	       (defclass PenroseWidget "public Gtk::Widget"
		      "public:"
		      (defmethod PenroseWidget ()
			(declare (values :constructor)
				 (construct (Glib--ObjectBase (string "PenroseWidget"))
					    ;(PenroseExtraInit (string "penrose-widget")) ;; name in css file
					    (Gtk--Widget)
					    (m_scale_prop *this (string "example_scale") 500)
					    (m_scale 1000)
					    ))
			(set_has_window true)
			(set_name (string "penrose-widget"))
			#+nil
			,(logprint "gtype name" `((G_OBJECT_TYPE_NAME (gobj))))
			(setf m_refCssProvider (Gtk--CssProvider--create))
			(let ((style (get_style_context)))
			  (style->add_provider m_refCssProvider
					       GTK_STYLE_PROVIDER_PRIORITY_APPLICATION)
			  (dot (m_refCssProvider->signal_parsing_error)
			       (connect
				(sigc--mem_fun *this
					       &PenroseWidget--on_parsing_error)
			       #+nil
				(lambda (section error)
			(declare (type "const Glib::RefPtr<Gtk::CssSection>&" section)
				 (type "const Glib::Error&" error))
			,(logprint "parse" `((error.what) (-> section
							      (get_file)
							      (get_uri))
					     (-> section
						 (get_start_location))
					     (-> section
						 (get_end_location)))))))
			  (m_refCssProvider->load_from_path (string "custom_gtk.css"))))
		      (defmethod ~PenroseWidget ()
			(declare (virtual)
				 (values :constructor)))
		      "protected:"
		      (defmethod get_request_mode_vfunc ()
			(declare (values "Gtk::SizeRequestMode"))
			(return (Gtk--Widget--get_request_mode_vfunc)))
		      (defmethod get_preferred_width_vfunc (minimum_width natural_width)
			(declare (type int& minimum_width natural_width))
			(setf minimum_width 60
			      natural_width 100))
		      (defmethod get_preferred_height_for_width_vfunc (width minimum_height natural_height)
			(declare (type int& minimum_height natural_height)
				 (type int width))
			(setf minimum_height 50
			      natural_height 70))
		      (defmethod get_preferred_height_vfunc (minimum_height natural_height)
			(declare (type int& minimum_height natural_height))
			(setf minimum_height 50
			      natural_height 70))
		      (defmethod get_preferred_width_for_height_vfunc (height minimum_height natural_height)
			(declare (type int& minimum_height natural_height)
				 (type int height))
			(setf minimum_height 60
			      natural_height 100))
		      (defmethod on_size_allocate (allocation)
			(declare (type Gtk--Allocation& allocation))
			(set_allocation allocation)
			(when m_refGdkWindow
			  (m_refGdkWindow->move_resize (allocation.get_x)
						       (allocation.get_y)
						       (allocation.get_width)
						       (allocation.get_height))))
		      #+nil
		      (defmethod measure_vfunc (orientation for_size minimum natural minimum_baseline natural_baseline)
			(declare (type Gtk--Orientation orientation)
				 (type int for_size)
				 (type int& minimum natural minimum_baseline natural_baseline))
			(if (== Gtk--Orientation--HORIZONTAL orientation)
			    (setf minimum 60
				  natural 100)
			    (setf minimum 50
				  natural 70))
			(setf minimum_baseline -1
			      natural_baseline -1))
		      
		      (defmethod on_map ()
			(Gtk--Widget--on_map))
		      (defmethod on_unmap ()
			(Gtk--Widget--on_unmap))
		      (defmethod on_realize ()
					; (Gtk--Widget--on_realize) ;; only call this when set_has_window false
			(set_realized)
			(setf m_scale (m_scale_prop.get_value))
			,(logprint "" `(m_scale))
			(unless m_refGdkWindow
			  (let ((attr))
			    (declare (type GdkWindowAttr attr))
			    (memset &attr 0 (sizeof attr))
			    (let ((allocation (get_allocation)))
			      ,@(loop for e in `(x y width height) collect
				     `(setf ,(format nil "attr.~a" e)
					   (,(format nil "allocation.get_~a" e)))))
			    (setf attr.event_mask (logior (get_events)
								Gdk--EXPOSURE_MASK)
				  attr.window_type GDK_WINDOW_CHILD
				  attr.wclass GDK_INPUT_OUTPUT
				  m_refGdkWindow (Gdk--Window--create
						  (get_parent_window)
						  &attr
						  (logior GDK_WA_X
							  GDK_WA_Y)
						  ))
			    (set_window m_refGdkWindow)
			    (m_refGdkWindow->set_user_data (gobj))))
			)
		      (defmethod on_unrealize ()
			(when m_refGdkWindow
			 (m_refGdkWindow.reset))
			(Gtk--Widget--on_unrealize))
		      (defmethod on_draw (cr)
			(declare (type "const Cairo::RefPtr<Cairo::Context>&" cr)
				 (values bool))
			(let ((allocation (get_allocation))
			      (scale_x (/ (static_cast<double> (allocation.get_width))
					  m_scale))
			      (scale_y (/ (static_cast<double> (allocation.get_height))
					  m_scale))
			      (style (get_style_context)))
			  (style->render_background cr
						    ,@(loop for e in `(x y width height) collect
							   `(,(format nil "allocation.get_~a" e))))
			  (let ((state2 (style->get_state))
				)
			    (Gdk--Cairo--set_source_rgba cr (style->get_color state2))
			    ,@(loop for (cmd x y) in `((m 155 165)
						       (l 155 838)
						       (l 265 900)
						       (l 849 564)
						       (l 849 438)
						       (l 265 100)
						       (l 155 165)
						       (m 265 100)
						       (l 265 652)
						       (l 526 502)
						       (m 369 411)
						       (l 633 564)
						       (m 369 286)
						       (l 369 592)
						       (m 369 286)
						       (l 849 564)
						       (m 633 564)
						       (l
							155 838)
						       )
				 collect
				   `(,(format nil "cr->~a_to" (ecase cmd
								(m "move")
								(l "line")))
				      (* ,(coerce x 'double-float)
					 scale_x)
				      (* ,(coerce y 'double-float)
					 scale_y)))
			    ,@(loop for (x y) in `()
				 collect
				   `(,(format nil "cr->line_to" )
				      (* ,(coerce x 'double-float)
					 scale_x)
				      (* ,(coerce y 'double-float)
					 scale_y)))
			    (cr->stroke)))
			(return true)
			)
		      (defmethod on_parsing_error (section error)
			(declare (type "const Glib::RefPtr<const Gtk::CssSection>&" section)
				 (type "const Glib::Error&" error))
			,(logprint "parse" `((error.what) (-> section
							      (get_file)
							      (get_uri))
					     #+nil(-> section
						 (get_start_location))
					     #+nil (-> section
						 (get_end_location)))))
		      "Gtk::StyleProperty<int> m_scale_prop;"
		      "Glib::RefPtr<Gdk::Window> m_refGdkWindow;"
		      "Glib::RefPtr<Gtk::CssProvider> m_refCssProvider;"
		      "int m_scale;"
		      )
		    

		    
		    )))
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



