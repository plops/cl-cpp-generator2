(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/112_usb/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  #+nil
  (let ((name `Error))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> stdexcept))
     :implementation-preamble `(do0
				(include "Error.h"))
     :code `(do0
	     (defclass ,name "public std::runtime_error"
	       "public:"
	       (defmethod ,name ()
		 (declare
		  (construct
		   (runtime_error (libusb_error_name err_code))
		   (_code err_code))
		  (values :constructor)))
	       (defmethod code ()
		 (declare (const)
			  (values int))
		 (return _code))
	       "private:"
	       "int _code;"))))

  #+nil
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"fatheader.hpp"
		     *source-dir*))
   `(do0
     (include
      ,@(loop for e in `(file-dialog
			 push-button
			 label
			 check-box
			 combo-box
			 v-box-layout
			 h-box-layout
			 drag
			 mime-data
			 tool-button
			 frame
			 validator
			 action
			 button-group
			 header-view
			 line-edit
			 spacer-item
			 stacked-widget
			 radio-button
			 tab-widget
			 tool-tip
			 mouse-event
			 style
			 timer

			 application
			 variant
			 map
			 vector
			 string-list
			 dir
			 pointer
			 color

			 main-window
			 double-validator
			 group-box
			 )
	      collect
	      (format nil "<~a>"
		      (cl-change-case:pascal-case
		       (format nil "q-~a" e))))

      ,@(loop for e in `(string
			 set
			 map
			 memory
			 vector
			 unordered_map
			 array
			 bitset
			 initializer_list
			 functional
			 algorithm
			 numeric
			 iterator
			 type_traits
			 cmath
			 cassert
			 cfloat
			 complex
			 cstddef
			 cstdint
			 cstdlib
			 mutex
			 thread
			 condition_variable)
	      collect
	      (format nil "<~a>" e))
      )))
  

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"usbpp.hpp"
		     *source-dir*))
   `(do0
     "template<typename T, void(*del)(T*)> using Handle = std::unique_ptr<T,decltype([](T*x){del(x);})>;"
     "using context = Handle<libusb_context, libusb_exit>;"
     (defun check (err)
       (declare (type int err)
		)
       (when (< err 0)
	 (throw (Error err))))
     (defun init ()
       (declare (values context)
		(inline))
       (let ((*ctx (libusb_context* nullptr))
	     )
	 (check (libusb_init &ctx))
	 (return context{ctx})))))

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     ;"import std_mod;"

     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
      ))))


  
