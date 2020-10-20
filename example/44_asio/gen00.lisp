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
  (defparameter *source-dir* #P"example/44_asio/source/")
  
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

		    (include <boost/asio.hpp>
			     <boost/asio/ts/buffer.hpp>
			     <boost/asio/ts/internet.hpp>)

		    (include "vis_01_message.hpp")
		    "using namespace std::chrono_literals;"
		    " "

		    (split-header-and-code
		     (do0
		      "// header"
		      )
		     (do0
		      "// implementation"
		      ))

		    "std::vector<char> buffer(20*1024);"


		    (space enum class CustomMsgTypes ":uint32_t"
			   (progn
			     "FireBullet,MovePlayer"
			  ))
		    
		    (defun grab_some_data (socket)
		      (declare (type "boost::asio::ip::tcp::socket&" socket))
		      (socket.async_read_some
		       (boost--asio--buffer (buffer.data)
				     (buffer.size)
				     )
		       (lambda (ec length)
			 (declare (type "std::error_code" ec)
				  (type "std::size_t" length)
				  (capture "&")) 
			 (unless ec
			   ,(logprint "read bytes:" `(length))
			   (for ("size_t i=0"
				 (< i length)
				 (incf i))
				(<< std--cout (aref buffer i)))
			   "// will wait until some data available"
			   (grab_some_data socket)))
		       ))
		    
		    (defun main (argc argv
				 )
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      ,(logprint "start" `(argc (aref argv 0)))
		      (let ((msg (message<CustomMsgTypes>)))
			(setf msg.header.id CustomMsgTypes--FireBullet)
			"int a=1;"
			"bool b = true;"
			"float c=3.14f;"
			(<< msg a b c)
			"int a2; bool b2; float c2;"
			(>> msg c2 b2 a2)
			,(logprint "out" `(a2 b2 c2)))
		      (let ((ec )
			    ;; this is where asio will do its work
			    (context)
			    ;; some fake work for context to prevent
			    ;; that asio exits early
			    (idle_work (boost--asio--io_context--work context))
			    ;; create asio thread so that it doesn't block the main thread while waiting
			    (context_thread (std--thread
					     (lambda ()
					       (declare (capture "&"))
					       (context.run)))))
			(declare (type "boost::system::error_code" ec)
				 (type "boost::asio::io_context" context))
			(let ((endpoint (boost--asio--ip--tcp--endpoint
					 (boost--asio--ip--make_address
					  (string ;"192.168.2.1"
					   "93.184.216.34"
					;"127.0.0.1"
					   ) ec)
					 80))
			      (socket (boost--asio--ip--tcp--socket context)))
			  (socket.connect endpoint ec)
			  (if ec
			      ,(logprint "failed to connect to address" `((ec.message)))
			      ,(logprint "connected" `()))
			  (when (socket.is_open)

			    ;; start waiting for data
			    (grab_some_data socket)
			    (let ((request ("std::string"
					    (string
					     ,(concatenate
					       'string
					       "GET /index.html HTTP/1.1\\r\\n"
					       "Host: example.com\\r\\n"
					       "Connection: close\\r\\n\\r\\n")))))
			      ;; issue the request
			      (socket.write_some
			       (boost--asio--buffer (request.data)
						    (request.size))
			       ec)

			      (std--this_thread--sleep_for 2000ms)

			      ;; message<T>
			      ;; header -> id (enumclass T), size (bytes)
			      ;; body (0 + bytes)
			      )))
			)
		      (return 0)))))

    (define-module
       `(message ()
	      (do0
	       
	       
		    (include <iostream>
			     <chrono>
			     <thread>
			     <vector>
			     )

		    (include <boost/asio.hpp>
			     <boost/asio/ts/buffer.hpp>
			     <boost/asio/ts/internet.hpp>)
	
		    " "

		    (split-header-and-code
		     (do0
		      "// header"
		      (defclass (message_header :template "typename T") ()
			"public:"
			"T id{};"
		      "uint32_t size = 0;")

		       (defclass+ (message :template "typename T") ()
			 "public:"
			 "message_header<T> header{};"
		       "std::vector<uint8_t> body;"
		       (defmethod size ()
			 (declare (const)
				  (values size_t))
			 (return (sizeof (+ (sizeof message_header<T>)
					    (body.size)))))
		       (defmethod operator<< (msg data)
			 (declare (values "template<typename DataType> friend message<T>&")
				  (type "message<T>&" msg)
				  (type "const DataType&" data))
			 ;; simple types can be pushed in
			 ;; some types can't be trivially serialized
			 ;; e.g. classes with static variables
			 ;; complex arrangements of pointers
			 (static_assert
			  "std::is_standard_layout<DataType>::value"
			  (string "data is too complicated"))
			 (let ((i (msg.body.size))
			       )
			   (msg.body.resize (+ (msg.body.size)
					       (sizeof DataType)))
			   (std--memcpy
			    (+ (msg.body.data) i)
			    &data
			    (sizeof DataType)
			    )
			   (setf msg.header.size (msg.size))
			   ;; arbitrary objects of arbitrary
			   ;; types can be chained and pushed
			   ;; into the vector
			   (return msg))
			 )
		       (defmethod operator>> (msg data)
			 (declare (values "template<typename DataType> friend message<T>&")
				  (type "message<T>&" msg)
				  (type "DataType&" data))
			 
			 (static_assert
			  "std::is_standard_layout<DataType>::value"
			  (string "data is too complicated"))
			 (let ((i (- (msg.body.size)
				     (sizeof DataType )))
			       )
			   ;; copy data from vector into users
			   ;; variable treat vector as a stack
			   ;; for performance of resize op
			   (std--memcpy
			    &data
			    
			    (+ (msg.body.data) i)
			    
			    (sizeof DataType)
			    )
			   (msg.body.resize i)
			   
			   (setf msg.header.size (msg.size))
			   
			   (return msg))
			 ))
		      )
		     (do0
		      "// implementation"
		      ))


		   
		    
		    
		    #+nil (do0
		     ;; https://github.com/OneLoneCoder/olcPixelGameEngine/blob/master/Videos/Networking/Parts1%262/net_message.h
		     ;; message<GAME> msg
		     ;; msg << x << y
		     ;; msg >> y >> x
		     
		     )
		    )))

        (define-module
       `(tsqueue ()
	      (do0
	       
	       
		    (include <iostream>
			     <chrono>
			     <thread>
			     <deque>
			     <mutex>
			     )

		    (include <boost/asio.hpp>
			     <boost/asio/ts/buffer.hpp>
			     <boost/asio/ts/internet.hpp>)
	
		    " "

		    (split-header-and-code
		     (do0
		      "// header"
		      (defclass+ (tsqueue :template "typename T") ()
			"public:"
			"tsqueue() = default;"
			"tsqueue(const tsqueue<T>&) = delete;"
			,@(loop for e in
				`((front "const T&")
				  (back "const T&")
				  (empty bool)
				  (size size_t)
				  (clear void)
				  (pop_front "T" :code (let ((el (std--move
								  (deq.front))))
							 (deq.pop_front)
							 (return el)))
				  (pop_back "T" :code (let ((el (std--move
								  (deq.back))))
							 (deq.pop_back)
							(return el)))
				  
				   (push_back "T"
					    :params ((item "const T&"))
					    :code (do0
						   (deq.emplace_back
						    (std--move item))
						   (std--))))
				collect
				(destructuring-bind (name type &key params code)
				    e
				 `(defmethod ,name ,(mapcar #'first params)
				    (declare (values ,type)
					     ,@(loop for (e f) in params
						     collect
						     `(type ,f ,e)))
				    (let ((lock (std--scoped_lock mux_deq)))
				      ,(if code
					   `(do0
					     ,@code)
					   `(return (dot deq (,name))))))))
			
			"protected:"
			"std::mutex mux_deq"
			"std::deque<T> deq;")
		      )
		     (do0
		      "// implementation"
		      ))
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



