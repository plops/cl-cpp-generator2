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
		      (include "vis_04_client.hpp")
		      " "
		      (space enum class CustomMsgTypes ":uint32_t"
			     (curly
			      ServerAccept
			      ServerDeny
			      ServerPing
			      MessagesAll
			      ServerMessage))
		      )
		     (do0
		      "// implementation"
		      (include "vis_00_base.hpp")
		      ))

		    "std::vector<char> buffer(20*1024);"


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
	       
		      #+nil (let ((msg (message<CustomMsgTypes>)))
			(setf msg.header.id CustomMsgTypes--FireBullet)
			"int a=1;"
			"bool b = true;"
			"float c=3.14f;"
			(<< msg a b c)
			"int a2; bool b2; float c2;"
			(>> msg c2 b2 a2)
			,(logprint "out" `(a2 b2 c2)))
		     #+nil  (let ((ec )
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

		       "template <typename T> class connection;"
		       
		       (defclass+ (owned_message :template "typename T") ()
			 "public:"
			 "std::shared_ptr<connection<T>> remote = nullptr;"
			 "message<T> msg;")
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
       `(tsqueue () ;; thread safe queue
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
			(defmethod ~tsqueue ()
			  (declare (virtual)
				   (values :constructor))
			  (clear))
			#+nil(defmethod empty ()
				    (declare (values bool)
					     (const))
				    (let ((lock (std--scoped_lock mux_deq)))
				      (return (dot deq (empty)))))
			,@(loop for e in
				`((front "const T&")
				  (back "const T&")
				  (empty bool)
				  ;(size size_t)
				  (clear void)
				  (pop_front "T" :code (let ((el (std--move
								  (deq.front))))
							 (deq.pop_front)
							 (return el)))
				  (pop_back "T" :code (let ((el (std--move
								  (deq.back))))
							 (deq.pop_back)
							(return el)))
				  
				  (push_back void
					    :params ((item "const T&"))
					    :code (do0
						   (deq.emplace_back
						    (std--move item))
						   (let ((ul (std--unique_lock<std--mutex> mux_blocking)))
						     (cv_blocking.notify_one))))
				  (push_front void
					    :params ((item "const T&"))
					    :code (do0
						   (deq.emplace_front
						    (std--move item))
						   (let ((ul (std--unique_lock<std--mutex> mux_blocking)))
						     (cv_blocking.notify_one))))
				  (wait void
					    :params ((item "const T&"))
					    :code (while (empty)
						   (let ((ul (std--unique_lock<std--mutex> mux_blocking)))
						     (cv_blocking.wait ul)))))
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
					     ,code)
					   `(return (dot deq (,name))))))))
			
			"protected:"
			"std::mutex mux_deq;"
			"std::mutex mux_blocking;"
			"std::condition_variable cv_blocking;"
			"std::deque<T> deq;")
		      )
		     (do0
		      "// implementation"
		      ))
		    )))

    (define-module
       `(connection ()
	      (do0
	       
	       
		    (include <iostream>
			     <chrono>
			     <thread>
			     
			     )
		    (include <boost/asio.hpp>
			     <boost/asio/ts/buffer.hpp>
			     <boost/asio/ts/internet.hpp>)
		    (include "vis_01_message.hpp"
			     "vis_02_tsqueue.hpp")
	
		    " "

		    (split-header-and-code
		     (do0
		      "// header"
		      ;; we can create shared pointer from within this obj
		      ;; like *this, but shared
		      (defclass+ (connection :template "typename T") "public std::enable_shared_from_this<connection<T>>"
			"public:"
			(space enum class owner
			       (progn
				 "server, client"))
			(defmethod connection (parent asio_context
					       socket q_in)
			  (declare (values :constructor)
				   (type owner parent)
				   (type boost--asio--io_context& asio_context)
				   (type boost--asio--ip--tcp--socket socket)
				   (type tsqueue<owned_message<T>>& q_in)
				   (virtual)
				   (construct 
					      (m_socket (std--move socket))
					      (m_asio_context asio_context)
					      (m_q_messages_in q_in)
					      (m_owner_type parent)
					      
					      ))
			  )
			(defmethod ~connection ()
			  (declare
			   (values :constructor)
			   (virtual))
			  )
			(defmethod get_id ()
			  (declare (const)
				   (values uint32_t))
			  (return id))
			(defmethod connect_to_client (uid=0)
			  (declare (type uint32_t uid=0))
			  (when (== owner--server
				    m_owner_type)
			    (when (m_socket.is_open)
			      (setf id uid)
			      (read_header))))
			(defmethod connect_to_server (endpoints)
			  (declare 
			   (type "const boost::asio::ip::tcp::resolver::results_type&" endpoints))
			  (when (== owner--client
				    m_owner_type)
			    (boost--asio--async_connect
			     m_socket endpoints
			     (lambda (ec endpoint)
			       (declare (capture this)
					(type std--error_code ec)
					(type boost--asio--ip--tcp--endpoint endpoint))
			       (unless ec
				 (read_header))
			       ))))
			(defmethod disconnect ()
			  (declare (values bool))
			  (return false))
			(defmethod is_connected ()
			  (declare (values bool)
				   (const))
			  (return (m_socket.is_open)))

			(defmethod send (msg)
			  (declare ;(values bool)
			   (type "const message<T>&"
			    ;"message<T>&"
				 msg)
				   ;(const)
			   )
			  (boost--asio--post
			   m_asio_context
			   (lambda ()
			     (declare (capture this msg))
			     ;; if message is currently being sent
			     ;; dont add second write_header work
			     (let ((idle (m_q_messages_out.empty)))
			       ;(declare (type "const auto" idle))
			       (m_q_messages_out.push_back msg)
			       (when idle
				 (write_header))))))
			"private:"
			;; async
			(defmethod read_header ()
			  (boost--asio--async_read
			   m_socket
			   (boost--asio--buffer
			    &m_msg_temporary_in.header
			    (sizeof message_header<T>))
			   (lambda (ec length)
			     (declare (capture this)
				      (type std--error_code ec)
				      (type std--size_t length))
			     (if ec
				 (do0
				  ,(logprint "read header fail"
					     `(id))
				  ;; this close will be detected by
				  ;; server and client
				  (m_socket.close))
				 (do0
				  (if (< 0 m_msg_temporary_in.header.size)
				      (do0
				       (m_msg_temporary_in.body.resize
					(m_msg_temporary_in.size))
				       (read_body)
				       )
				      (do0
				       ;; message is complete (no body)
				       (add_to_incoming_message_queue)
				       ))
				  )))
			 )
			  )
			(defmethod read_body ()
			  (boost--asio--async_read
			   m_socket
			   (boost--asio--buffer
			    (m_msg_temporary_in.body.data)
			    (m_msg_temporary_in.body.size))
			   (lambda (ec length)
			     (declare (capture this)
				      (type std--error_code ec)
				      (type std--size_t length))
			     (if ec
				 (do0
				  ,(logprint "read body fail"
					     `(id))
				  (m_socket.close))
				 (do0
				  (add_to_incoming_message_queue))))
			   )
			  )
			(defmethod write_header ()
			  (boost--asio--async_write
			   m_socket
			   (boost--asio--buffer
			    (dot &m_q_messages_out
				 (front)
				 header)
			    (sizeof message_header<T>))
			   (lambda (ec length)
			     (declare (capture this)
				      (type std--error_code ec)
				      (type std--size_t length))
			     (if ec
				 (do0
				  ,(logprint "write header fail"
					     `(id))
				  (m_socket.close))
				 (do0
				  (if (< 0 (dot m_q_messages_out
						(front)
 						body
						(size)))
				      (write_body)
				      (do0
				       (m_q_messages_out.pop_front)
				       ;; if q not empty we need to
				       ;; send more messages by
				       ;; issueing the next header
				       (unless (m_q_messages_out.empty)
					 (write_header))))))))
			  )
			(defmethod write_body ()
			  (boost--asio--async_write
			   m_socket
			   (boost--asio--buffer
			    (dot m_q_messages_out
				 (front)
				 body
				 (data))
			    (dot m_q_messages_out
				 (front)
				 body
				 (size)))
			   (lambda (ec length)
			     (declare (capture this)
				      (type std--error_code ec)
				      (type std--size_t length))
			     (if ec
				 (do0
				  ,(logprint "write body fail"
					     `(id))
				  (m_socket.close))
				 (do0
				  (m_q_messages_out.pop_front)
				  (unless (m_q_messages_out.empty)
				    (write_header))))))
			  )

			(defmethod add_to_incoming_message_queue ()
			  (if (== owner--server
				  m_owner_type)
			      (m_q_messages_in.push_back
			       (curly (this->shared_from_this)
				      m_msg_temporary_in))
			      (m_q_messages_in.push_back
			       (curly nullptr
				      m_msg_temporary_in)))
			  (read_header)
			  )
			
			"protected:"
			;; each connection has a socket
			"boost::asio::ip::tcp::socket m_socket;"
			;; shared context, server uses only one
			"boost::asio::io_context& m_asio_context;"
			"tsqueue<message<T>> m_q_messages_out;"

			"message<T> m_msg_temporary_in;"
			;; input queue is owned by server or client 
			"tsqueue<owned_message<T>>& m_q_messages_in;"
			"owner m_owner_type = owner::server;"
			"uint32_t id=0;"
			
			)
		      )
		     (do0
		      "// implementation"
		      ))
		    )))

    (define-module
       `(client ()
	      (do0
	       
	       
		    (include <iostream>
			     <chrono>
			     <thread>
			     
			     )
		    (include "vis_01_message.hpp"
			     "vis_02_tsqueue.hpp"
			     "vis_03_connection.hpp")
	
		    " "

		    (split-header-and-code
		     (do0
		      "// header"
		      ;; we can create shared pointer from within this obj
		      ;; like *this, but shared
		      (defclass+ (client_interface :template "typename T") ()
			"public:"
			
			(defmethod client_interface ()
			  (declare (values :constructor)
				  
				   (construct (m_socket m_asio_context))
				   (virtual))
			  )
			(defmethod ~client_interface ()
			  (declare
			   (values :constructor)
			   (virtual))
			  (disconnect)
			  )

			(defmethod connect (host port)
			  (declare (values bool)
				   (type "const std::string&" host)
				   (type "const uint16_t" port))
			  (handler-case
			      (progn
				
				(let ((resolver (boost--asio--ip--tcp--resolver
						 m_asio_context))
				      (endpoints (resolver.resolve
						     host
						     (std--to_string port))))
				  
				  (setf m_connection
					(std--make_unique<connection<T>>
					 connection<T>--owner--client
					 m_asio_context
					 (boost--asio--ip--tcp--socket
					  m_asio_context)
					 m_q_messages_in))
				  (m_connection->connect_to_server
				   endpoints)
				  (setf m_thread_asio
					(std--thread (lambda ()
						       (declare (capture this))
						       (m_asio_context.run)))))
			       )
			    ("std::exception&" (e)
			      ,(logprint "client exception" `((e.what)))
			      (return false))
			   )
			  (return true))
			(defmethod disconnect ()
			  (when (is_connected)
			    (m_connection->disconnect))
			  (m_asio_context.stop)
			  (when (m_thread_asio.joinable)
			    (m_thread_asio.join))

			  (m_connection.release)
			  )

			(defmethod is_connected ()
			  (declare (values bool)
				   )
			  (if m_connection
			      (return (m_connection->is_connected))
			      (return false)))


			(defmethod send (msg)
			  (declare (type "const message<T>&"
				    ;"message<T>&"
					 msg))
			  (when (is_connected)
			    (m_connection->send msg)))
			(defmethod incoming ()
			  (declare (values "tsqueue<owned_message<T>>&"))
			  (return m_q_messages_in))

			"protected:"
			"boost::asio::io_context m_asio_context;"
			"std::thread m_thread_asio;"
			;; initial socket, when connected this is
			;; handed over as unique pointer to connection
			"boost::asio::ip::tcp::socket m_socket;"
			"std::unique_ptr<connection<T>> m_connection;"
			
			"private:"
			
			"tsqueue<owned_message<T>> m_q_messages_in;"
			)
		      )
		     (do0
		      "// implementation"
		      ))
		    )))

    (define-module
       `(server ()
	      (do0
	       
	       
		    (include <iostream>
			     <chrono>
			     <thread>
			     
			     )
		    (include "vis_01_message.hpp"
			     "vis_02_tsqueue.hpp"
			     "vis_03_connection.hpp")
	
		    " "

		    (split-header-and-code
		     (do0
		      "// header"
		      ;; we can create shared pointer from within this obj
		      ;; like *this, but shared
		      (defclass+ (server_interface :template "typename T") ()
			"public:"
			
			(defmethod server_interface (port)
			  (declare (values :constructor)
				   (type uint16_t port)
				  (construct (m_asio_acceptor
					       m_asio_context
					       (boost--asio--ip--tcp--endpoint
						(boost--asio--ip--tcp--v4)
						port)))
				   (virtual))
			  )
			(defmethod ~server_interface ()
			  (declare
			   (values :constructor)
			   (virtual))
			  (stop)
			  )

			(defmethod start ()
			  (declare (values bool))
			  (handler-case
			      (progn
				;; make sure asio is having before
				;; context starts
				(wait_for_client_connection)
				(setf m_thread_context
				      (std--thread
				       (lambda ()
					 (declare (capture this))
					 (m_asio_context.run)))))
			    ("std::exception&" (e)
			      ,(logprint "server exception"
					 `((e.what)))
			      (return false)))
			  ,(logprint "server started")
			  (return true))

			(defmethod stop ()
			  (m_asio_context.stop)
			  (when (m_thread_context.joinable)
			    (m_thread_context.join)
			    )
			  ,(logprint "server stopped")
			  )

			;; async
			(defmethod wait_for_client_connection ()
			  (m_asio_acceptor.async_accept
			   (lambda (ec socket)
			     (declare (capture this)
				      (type std--error_code ec)
				      (type boost--asio--ip--tcp--socket socket))
			     (if ec
			       ,(logprint "server connection error"
					  `((ec.message)))
			       (do0
				,(logprint "server new connection"
					   `((socket.remote_endpoint)))
				(let ((newconn (std--make_shared<connection<T>>
						connection<T>--owner--server
						m_asio_context
						(std--move socket)
						m_q_messages_in)))
				  (if (on_client_connect newconn)
				      (do0
				       (m_deq_connections.push_back (std--move newconn))
				       (incf n_id_counter)
				       (-> (m_deq_connections.back)
					   (connect_to_client n_id_counter))
				       ,(logprint "server connection approved"
						  `((-> (m_deq_connections.back)
							(get_id))))
				       )
				      (do0
				       ,(logprint "server connection denied"))))))
			     ;; keep working
			     (wait_for_client_connection)

			     )))
			(defmethod message_client (client msg)
			  (declare (type std--shared_ptr<connection<T>> client)
				   (type "const message<T>&" msg))
			  (if (and client (client->is_connected))
			      (do0 (client->send msg ))
			      (do0
			       (on_client_disconnect client )
			       (client.reset)
			       ;; following can be expensive for many clients
			       (m_deq_connections.erase
				(std--remove (m_deq_connections.begin)
					     (m_deq_connections.end)
					     client)
				(m_deq_connections.end)))
			    ))
			(defmethod message_all_clients (msg
							ignore_client=nullptr)
			  (declare (type std--shared_ptr<connection<T>> ignore_client=nullptr)
				   (type "const message<T>&" msg))
			  (let ((invalid_client_exists false))
			   (for-range (&client m_deq_connections)
				      (if (and client (client->is_connected))
					  (unless (== client ignore_client)
					    (client->send msg ))
					  (do0
					   (on_client_disconnect client )
					   (client.reset)
					   (setf invalid_client_exists true)
					   )
					  ))
			    ;; faster and not changing the deq in iteration
			    (when invalid_client_exists
			      (m_deq_connections.erase
			       (std--remove (m_deq_connections.begin)
					    (m_deq_connections.end)
					    nullptr)
			       (m_deq_connections.end)))))

			(defmethod update (,(intern
					     (string-upcase
					      (format nil "n_max_messages=0x~x"
						      (- (expt 2 64) 1))))
					   wait=false)
			  (declare (type size_t ,(intern
						  (string-upcase
						   (format nil "n_max_messages=0x~x"
							   (- (expt 2 64) 1)))))
				   (type bool wait=false)
				   )
			  #+nil(when wait
			    (m_q_messages_in.wait))
			  (let ((n_message_count (size_t 0)))
			    (while (and
				    (< n_message_count
				       n_max_messages)
				    (not (m_q_messages_in.empty)))
				   (let ((msg (m_q_messages_in.pop_front)))
				     (on_message msg.remote
						 msg.msg)
				     (incf n_message_count))))
			  )
			
			"protected:"
			(defmethod on_client_connect (client)
			  (declare (type std--shared_ptr<connection<T>> client)
				   (virtual)
				   (values bool))
			  (return false))
			(defmethod on_client_disconnect (client)
			  (declare (type std--shared_ptr<connection<T>> client)
				   (virtual)
				   ))
			(defmethod on_message (client msg)
			  (declare (type std--shared_ptr<connection<T>> client)
				   (type message<T>& msg)
				   (virtual)
				   ))
			
			"tsqueue<owned_message<T>> m_q_messages_in;"
			"std::deque<std::shared_ptr<connection<T>>> m_deq_connections;"
			"boost::asio::io_context m_asio_context;"
			"std::thread m_thread_context;"
			"boost::asio::ip::tcp::acceptor m_asio_acceptor;"
			;; consistent id in system (client knows)
			"uint32_t n_id_counter=10000;"
			)
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
			(file-h (string-upcase (format nil "~a_H" file)))
			(fn-h (asdf:system-relative-pathname 'cl-cpp-generator2
								      (format nil "~a/~a.hpp"
									      *source-dir* file))))
		   (with-open-file (sh fn-h
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
		     )
		   (sb-ext:run-program "/usr/bin/clang-format"
				       (list "-i"  (namestring fn-h)
				   
				   ))
		   )
		 

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



 
