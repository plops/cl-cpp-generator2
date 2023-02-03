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
  #+nil
  (let ((name `UsbInterface))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include "UsbUsing.h"
				 "UsbError.h")
			(include<>
			 vector
				   libusb-1.0/libusb.h)
			
			)
     :implementation-preamble `(do0
				
				(include "UsbUsing.h"
					 "UsbError.h"
					 )
				(include<> vector
					  ; libusb-1.0/libusb.h
					  ; exception
					  ; stdexcept
					   )
				;"import fatheader;"
				)
     :code `(do0
	 
	     )))

  (let ((name `UsbInterface))
   (write-source
    (asdf:system-relative-pathname
     'cl-cpp-generator2
     (merge-pathnames #P"UsbInterface.hpp"
		      *source-dir*))
    `(do0
      "#pragma once"
      (include
       "UsbError.h"
       "UsbUsing.h"
      
      
       )
      (include<> libusb-1.0/libusb.h)
      (do0
       (defclass+ ,name ()
	 "static constexpr int Invalid = -1;"
	 "int handle = Invalid;"
	 "libusb_device_handle* dev = nullptr;"

	 (defun try_release ()
	   (unless (== Invalid handle)
	     (let ((h handle))
	       (setf handle Invalid)
	       (return (libusb_release_interface dev h)))))
	 
	 "public:"
	 (defmethod ,name (i dev)
	   (declare
	    (type int i)
	    (type device_handle& dev)
	    (construct (handle i)
		       (dev (dev.get))
		       )
	    (values :constructor)))
	 "Interface(const Interface&)=delete;"
	 "Interface& operator=(const Interface&)=delete;"
	 (defmethod ,name (from)
	   (declare
	    (type Interface&& from)
	    
	    (values :constructor))
	   (setf *this
		 (std--move from)))
	 (defun release_interface ()
	   (check (try_release)))
	 (defun operator= (from)
	   (declare
	    (type Interface&& from)
	    
	    (values Interfac&))
	   (release_interface)
	   (setf handle from.handle
		 dev from.dev
		 from.handle Invalid)
	   (return *this))
	 (defmethod ~Interface ()
	   (declare
	    (values :constructor))
	   (let ((e (libusb_release_interface dev handle)))
	     (unless (== e 0)
	       (<< std--cerr
		   (string "failed to release interface")
		   (UsbError e)))))
	   
	 "private:"
	 )
       ))))

  
  
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"Usbpp.hpp"
		     *source-dir*))
   `(do0
     "#pragma once"
     (include
      "UsbError.h"
      "UsbUsing.h"
      
      
      )
     (include<> libusb-1.0/libusb.h
		vector)
     (do0

	     ;"template<typename T, void(*del)(T*)> using Handle = std::unique_ptr<T,decltype([](T*x){del(x);})>;"

     
	     (do0
      ;"using context = Handle<libusb_context, libusb_exit>;"
      
      (defun init ()
	(declare (values context)
		 (inline)
					;(inline)
		 )
	(let ((ctx nullptr)
	      )
	  (declare (type libusb_context* ctx))
	  (check (libusb_init &ctx))
	  (return context{ctx}))))
	     
     (do0
      ;"using device = Handle<libusb_device, libusb_unref_device>;"
      (defun get_device_list (ctx)
	(declare (values "std::vector<device>")
		 (type context& ctx)
		 ;(inline)
		 )
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
      ;"using device_handle = Handle<libusb_device_handle,libusb_close>;"
      (defun open (dev)
	(declare (type device& dev)
		 (values device_handle)
		; (inline)
		 )
	(let ((handle nullptr))
	  (declare (type libusb_device_handle* handle))
	  (let ((err (libusb_open (dev.get)
				  &handle)))
	    (check err)
	    (return "device_handle{handle}")))
	))

     
     (do0
      ;"using device_descriptor = libusb_device_descriptor;"
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
		;(inline)
		(type context& ctx)
		(type uint16_t vid pid))
       (let ((h (libusb_open_device_with_vid_pid (ctx.get)
						 vid pid))
	     )
	 "device_handle ret{h};"
	 (when (== nullptr ret)
	   (throw (UsbError LIBUSB_ERROR_NOT_FOUND)))
	 (return ret))
       )
     )))

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"fatheader.hpp"
		     *source-dir*))
   `(do0
     "#pragma once"
     (include
      ;"UsbError.h"
      ;"UsbUsing.h"
      ,@(loop for e in `(libusb-1.0/libusb.h
			 memory
			 ranges
			 vector
			 chrono
			 algorithm
			 iostream
			 exception
			 )
	      collect
	      (format nil "<~a>" e))
      )))
  

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"UsbUsing.h"
		     *source-dir*))
   `(do0
      "#pragma once"
					;"import fatheader;"
      (include<> memory
		 libusb-1.0/libusb.h)
     "template<typename T, void(*del)(T*)> using Handle = std::unique_ptr<T,decltype([](T*x){del(x);})>;"

     "using context = Handle<libusb_context, libusb_exit>;"
     

     "using device = Handle<libusb_device, libusb_unref_device>;"
     


     "using device_handle = Handle<libusb_device_handle,libusb_close>;"
     

     
     "using device_descriptor = libusb_device_descriptor;"
      

     ))

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (include "UsbInterface.hpp"
	      ;"UsbUsing.h"
	      ;"UsbError.h"
	      "Usbpp.hpp"
	      )
     (include "fatheader.hpp")
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


  
