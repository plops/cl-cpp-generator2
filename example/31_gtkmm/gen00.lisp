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
  (defparameter *source-dir* #P"example/31_gtkmm/source/")
  
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
	       ;; https://developer.gnome.org/gtkmm-tutorial/stable/sec-basics-simple-example.html.en
	       ;; https://www.murrayc.com/permalink/2015/07/31/gtkmm-now-uses-c11/
		    (include <iostream>
			     <chrono>
			     <thread>
			     
			     )

		    ;(include <gtkmm.h>)
		    " "

		    
		    
		    (split-header-and-code
		     (do0
		      "// header"
		      #+nil 
		      (include <gtkmm/button.h>
			       <gtkmm/window.h>)
		      (include <gtkmm.h>)
		      " "
		      )
		     (do0
		      "// implementation"
		      (include "vis_00_base.hpp")
		      " "
		      ))

		    #+nil (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (let ((app (Gtk--Application--create argc argv
							   (string "org.gtkmm.examples.base")))
			    (win))
			(declare (type Gtk--Window win))
			(win.set_default_size 200 200)
			(app->run win)))


		    ;; https://github.com/GNOME/gtkmm/blob/master/demos/gtk-demo/example_treeview_liststore.cc
		    (defclass CellItem_Bug ()
			"public:"
		      (defmethod CellItem_Bug ()
			(declare
			 (construct (m_fixed false)
				    (m_number 0))
			 (values :constructor)))
		      (defmethod ~CellItem_Bug ()
			(declare (values :constructor)))
		      (defmethod CellItem_Bug (src)
			(declare (values :constructor)
				 (type "const CellItem_Bug&" src))
			(operator= src))
		      (defmethod CellItem_Bug (fixed number severity description)
			(declare (values :constructor)
				 (construct (m_fixed fixed)
					    (m_number number)
					    (m_severity severity)
					    (m_description description))
				 (type bool fixed)
				 (type guint number)
				 (type "const Glib::ustring&" severity)
				 (type "const Glib::ustring&" description)))
		      (defmethod operator= (src)
			(declare (values CellItem_Bug&)
				 (type "const CellItem_Bug&" src))
			,@(loop for e in `(m_fixed m_number m_severity m_description) collect
			       `(setf ,e (dot src ,e)))
			(return *this))
		      "bool m_fixed;"
		      "guint m_number;"
		      "Glib::ustring m_severity;"
		      "Glib::ustring m_description;")


		    (defclass Example_TreeView_ListStore "public Gtk::Window"
		      "public:"
		      (defmethod Example_TreeView_ListStore ()
			(declare (values :constructor)
				 (construct (m_VBox Gtk--ORIENTATION_VERTICAL 8)
					    (m_Label (string "This is the bug list."))))
			(set_title (string "Gtk::ListStore demo"))
			(set_border_width 8)
			(set_default_size 280 250)
			(add m_VBox)
			(m_VBox.pack_start m_Label Gtk--PACK_SHRINK)
			(m_ScrolledWindow.set_shadow_type Gtk--SHADOW_ETCHED_IN)
			(m_ScrolledWindow.set_policy Gtk--POLICY_NEVER Gtk--POLICY_AUTOMATIC)
			(m_VBox.pack_start m_ScrolledWindow)
			;-(create_model)
			)
		      (defmethod ~Example_TreeView_ListStore ()
			(declare (values :constructor)
				 ;; override
				 ))
		      "protected:"
		      (defmethod create_model ()
			(declare (virtual))
			(setf m_refListStore (Gtk--ListStore--create m_columns))
			(add_items)
			(std--for_each
			 (m_vecItems.begin)
			 (m_vecItems.end)
			 (sigc--mem_fun *this
					&Example_TreeView_ListStore--liststore_add_item)))
		      (defmethod add_columns ()
			(declare (virtual))
			(let ((cols_count (m_TreeView.append_column_editable (string "Fixed?")
									     m_columns.fixed))
			      (pColumn (m_TreeView.get_column (- cols_count 1))))
			  ;; set to fixed 50 pixel size
			  (pColumn->set_sizing Gtk--TREE_VIEW_COLUMN_FIXED)
			  (pColumn->set_fixed_width 50)
			  (pColumn->set_clickable))
			(m_TreeView.append_column (string "Bug Number")
						  m_columns.number)
			(m_TreeView.append_column (string "Severity")
						  m_columns.severity)
			(m_TreeView.append_column (string "Description")
						  m_columns.description)
			)
		      (defmethod add_items ()
			(declare (virtual)))
		      (defmethod liststore_add_item (foo)
			(declare (virtual)
				 (type "const CellItem_Bug&" foo))
			(let ((row (deref (m_refListStore->append))))
			  ,@(loop for e in `(fixed number severity description) collect
				 `(setf (aref row (dot m_columns ,e))
					(dot foo ,(format nil "m_~a" e))))))
		      "Gtk::Box m_VBox;"
		      "Gtk::ScrolledWindow m_ScrolledWindow;"
		      "Gtk::Label m_Label;"
		      "Gtk::TreeView m_TreeView;"
		      "Glib::RefPtr<Gtk::ListStore> m_refListStore;"

		      "typedef std::vector<CellItem_Bug> type_vecItems;"
		      "type_vecItems m_vecItems;"
		      (do0
		       ,(let ((l `((bool fixed)
				   ("unsigned int" number)
				   ("Glib::ustring" severity)
				   ("Glib::ustring" description))))
			  `(space "struct ModelColumns : public Gtk::TreeModelColumnRecord"
				  (progn
				    ,@(loop for (e f) in l collect
					   (format nil "Gtk::TreeModelColumn<~a> ~a;" e f))
				    (defun+ ModelColumns ()
				      (declare (values :constructor))
				      ,@(loop for (e f) in l collect
					     `(add ,f))))))
		       "const ModelColumns m_columns;")
		      
		      )


		    (defun do_treeview_liststore ()
		      (declare (values "Gtk::Window*"))
		      (return (new (Example_TreeView_ListStore))))
		    
		    (defclass HelloWorld "public Gtk::Window"
		      "public:"
		      (defmethod HelloWorld ()
			(declare (values :constructor)
				 (construct (m_button (string "_Hello World") true)))
			(set_border_width 10)
			(dot m_button
			     (signal_clicked)
			     (connect
			      (lambda ()
				,(logprint "button" `()))
			      #+nil
			      (sigc--mem_fun *this
					     &HelloWorld--on_button_clicked
					     ))
			     )
			(add m_button)
			(m_button.show))
		      (defmethod ~HelloWorld ()
			(declare (values :constructor)))

		      "protected:"
		      #+nil (defmethod on_button_clicked ()
			,(logprint "button" `()))
		      
		      "Gtk::Button m_button;"
		      
		      )

		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (let ((app (Gtk--Application--create argc argv
							   (string "org.gtkmm.example")))
			    (hw))
			(declare (type HelloWorld hw))
			;(win.set_default_size 200 200)
			(app->run hw)))

		    
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



