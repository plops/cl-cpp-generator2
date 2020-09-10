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
  (defparameter *source-dir* #P"example/40_gtk3_custom_widget/source/")
  
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
		      (include "vis_00_base.hpp")
		      " "
		      ))
		    #+nil (defclass PenroseExtraInit "public Glib::ExtraClassInit"
		      "public:"
		      (defmethod PenroseExtraInit (css_name)
			(declare (values :constructor)
				 (construct (Glib--ExtraClassInit ;class_init_function
					     (lambda (g_class class_data)
					       (declare (type void* g_class class_data)
							)
					       (g_return_if_fail (GTK_IS_WIDGET_CLASS g_class))
					       (let ((klass (static_cast<GtkWidgetClass*> g_class))
						     (css_name2 (static_cast<Glib--ustring*> class_data)))
						 (gtk_widget_class_set_css_name klass (css_name2->c_str))))
					     &m_css_name
					     ;; instance_init_function
					     (lambda (instance g_class)
					       (declare (type void* g_class)
							(type GTypeInstance* instance)
							)
					       (g_return_if_fail (GTK_IS_WIDGET instance)))
					     ))
				 (type "const Glib::ustring&" css_name)))
		      "private:"
		      "Glib::ustring m_css_name;")

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
			)
		      (defmethod on_parsing_error (section error)
			(declare (type "const Glib::RefPtr<Gtk::CssSection>&" section)
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
		    (defclass ExampleWindow "public Gtk::Window"
		      "public:"
		      (defmethod ExampleWindow ()
			(declare (values :constructor)
				 )
			(set_title (string "custom widget example"))
			(set_border_width 6)
			(set_default_size 600 400)
			(m_grid.set_margin 6)
			(m_grid.set_row_spacing 10)
			(m_grid.set_column_spacing 10)
			;;(Gtk--Container--add m_grid)
			;;(add m_penrose)
			(add m_grid)
			;;(this->add m_grid)
			(m_grid.attach m_penrose 0 0)
			(m_penrose.show)
			(show_all_children)
			
			)
		      (defmethod ~ExampleWindow ()
			(declare (values :constructor)
				 (virtual)))
		      ; "protected:"
		      #+nil (defmethod on_button_quit ()
			      (hide))
		      "Gtk::Grid m_grid;"
		      "PenroseWidget m_penrose;"
		      )

		    

		    
		    
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



