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
			(include<> exception
				   stdexcept
				   libusb-1.0/libusb.h)
			
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
	       (defmethod ,name (err_code)
		 (declare
		  (type int err_code)
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

     (do0
      "using context = Handle<libusb_context, libusb_exit>;"
      
      (defun init ()
	(declare (values context)
		 (inline))
	(let ((ctx nullptr)
	      )
	  (declare (type libusb_context* ctx))
	  (check (libusb_init &ctx))
	  (return context{ctx}))))

     (do0
      "using device = Handle<libusb_device, libusb_unref_device>;"
      (defun get_device_list (ctx)
	(declare (values "std::vector<device>")
		 (type context& ctx)
		 (inline))
	(let ((list nullptr)
	      (n (libusb_get_device_list (ctx.get)
					 &list)))
	  (declare (type libusb_device** list))
	  (check n)
	  (let ((ret (std--vector<device>)))
	    (dotimes (i n)
	      (ret.emplace_back (aref list i)))
	    (libusb_free_device_list list false)
	    (return ret)))))


     (do0
      "using device_handle = Handle<libusb_device_handle,libusb_close>;"
      (defun open (dev)
	(declare (type device& dev)
		 (values device_handle)
		 (inline))
	(let ((handle nullptr))
	  (declare (type libusb_device_handle* handle))
	  (let ((err (libusb_open (dev.get)
				  &handle)))
	    (check err)
	    (return "device_handle{handle}")))
	))

     
     (do0
      "using device_descriptor = libusb_device_descriptor;"
      (defun get_device_descriptor (dev)
	(declare (type "const  device&" dev)
		 (values device_descriptor))
	(let ((ret (device_descriptor)))
	  (check (libusb_get_device_descriptor (dev.get)
					       &ret))
	  (return ret))
	))

     (defun open_device_with_vid_pid (ctx vid pid)
       (declare (values device_handle)
		(inline)
		(type context& ctx)
		(type uint16_t vid pid))
       (let ((h (libusb_open_device_with_vid_pid (ctx.get)
						 vid pid))
	     )
	 "device_handle ret{h};"
	 (when (== nullptr ret)
	   (throw (UsbError LIBUSB_ERROR_NOT_FOUND)))
	 (return ret))
      )))

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (include "usbpp.hpp")
					;"import fatheader;"
     ;(include<> algorithm)
     ;; Bus 001 Device 039: ID 8087:0a2b Intel Corp. Bluetooth wireless interface

     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (let ((ctx (init))
	     (bt (open_device_with_vid_pid ctx
					   "0x8087"
					   "0x0a2b"
					   )))
	 #+nil
	 (let ((devices (get_device_list ctx))
		   (bt_devs (std--find_if devices
					  (lambda (dev)
					    (declare (type
						      "const auto&"
					;"const device&"
						      dev))
					    (let ((d (get_device_descriptor dev)))
					      (return (logand
						       (== "0x8087" d.idVendor )
						       (== "0x0a2b" d.idProduct))))))))
	   (when (== (devices.end)
		     bt_devs)
	     (<< std--cerr (string "no intel bluetooth device found"))
	     (return 1))
	   (let ((bt (open *bt_devs)))))
	 )
      ))))


  
