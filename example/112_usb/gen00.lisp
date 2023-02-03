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
  (load "util.lisp")
  (let ((name `UsbError))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> stdexcept)
			)
     :implementation-preamble `(do0
				(include "UsbError.h")
				)
     :code `(do0
	     (defun check (err)
				  (declare (type int err)

					   )
				  (when (< err 0)
				    (throw (UsbError err))))
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

 
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"fatheader.hpp"
		     *source-dir*))
   `(do0
     
     (include
      "UsbError.h"
      
      ,@(loop for e in `(libusb-1.0/libusb.h
			 memory
			 ranges
			 vector
			 chrono
			 algorithm
			 iostream
			 exception
			 
			 
			 ;; string
			 ;; set
			 ;; map
			 
			 ;; vector
			 ;; unordered_map
			 ;; array
			 ;; bitset
			 ;; initializer_list
			 ;; functional
			 
			 ;; numeric
			 ;; iterator
			 ;; type_traits
			 ;; cmath
			 ;; cassert
			 ;; cfloat
			 ;; complex
			 ;; cstddef
			 ;; cstdint
			 ;; cstdlib
			 ;; mutex
			 ;; thread
			 ;; condition_variable
			 )
	      collect
	      (format nil "<~a>" e))
      )))
  

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"usbpp.hpp"
		     *source-dir*))
   `(do0
     "import fatheader;"
     "template<typename T, void(*del)(T*)> using Handle = std::unique_ptr<T,decltype([](T*x){del(x);})>;"
     "using context = Handle<libusb_context, libusb_exit>;"
     
     (defun init ()
       (declare (values context)
		(inline))
       (let ((ctx nullptr)
	     )
	 (declare (type libusb_context* ctx))
	 (check (libusb_init &ctx))
	 (return context{ctx})))))

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (include "usbpp.hpp")

     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
      ))))


  
