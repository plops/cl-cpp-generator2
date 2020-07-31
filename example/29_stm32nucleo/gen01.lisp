;(setf *features* (set-difference *features* '(:generic-c)))
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


;; qmake -project
;; qmake source1
(progn
  (defparameter *source-dir* #P"example/29_stm32nucleo/source1/")
  
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
       `(main ((_main_version :type "std::string")
	       (_code_repository :type "std::string")
	       (_code_generation_time :type "std::string")
	       )
	      (do0
	       (comments "https://doc.qt.io/qt-5/qtserialport-blockingmaster-example.html")
	       (include <iostream>
			<chrono>
			<cstdio>
			<cassert>
					;<unordered_map>
			;<string>
					;     <fstream>
			<thread>
			;     <vector>
			;     <experimental/iterator>
			;     <algorithm>
			)
	       ;" "
		    (include
			
		     <QtWidgets/QApplication>
		     <QtWidgets/QDialog>
		     <QtSerialPort/QSerialPort>
		     <QTime>
		     
		     )
		;    " "
					;(include <yacx/main.hpp>)
		    (split-header-and-code
		     (do0
		      (include <QMutex>
			       <QThread>
			       <QWaitCondition>))
		     ;(include "vis_01_serial.hpp")
		     (include "vis_02_dialog.hpp"))
		    
		    "using namespace std::chrono_literals;"
		    (let ((state ,(emit-globals :init t)))
		      (declare (type "State" state)))
		    
		    
		    		    
		    (defun main (argc argv)
		      (declare (values int)
			       (type int argc)
			       (type ; "char const *const *const"
				"char**"
				     argv))
		      (setf ,(g `_main_version)
			    (string ,(let ((str (with-output-to-string (s)
						  (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
				       (subseq str 0 (1- (length str))))))

		      (setf
		       ,(g `_code_repository) (string ,(format nil "https://github.com/plops/cl-cpp-generator2/tree/master/example/27_sparse_eigen_hydrogen"))
		       ,(g `_code_generation_time) 
                       (string ,(multiple-value-bind
                                      (second minute hour date month year day-of-week dst-p tz)
				    (get-decoded-time)
				  (declare (ignorable dst-p))
				  (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
					  hour
					  minute
					  second
					  (nth day-of-week *day-names*)
					  year
					  month
					  date
					  (- tz)))))

		      (setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
						   (time_since_epoch)
						   (count)))
		      ,(logprint "start main" `(,(g `_main_version)))
		      ,(logprint "" `(,(g `_code_repository)))
		      ,(logprint "" `(,(g `_code_generation_time)))

		      (let (((app argc argv)))
			(declare (type QApplication (app argc argv)))
			(let ((dialog))
			  (declare (type Dialog dialog))
			  (dialog.show)
			  (return (app.exec))))

		      ,(logprint "end main" `())
		      (return 0)))))

    (define-module
       `(serial ()
	      (do0
	       (comments "https://doc.qt.io/qt-5/qtserialport-blockingmaster-example.html")
	       
	       
		    (include
			
		     <QtWidgets/QApplication>
		     <QtWidgets/QDialog>
		     <QtSerialPort/QSerialPort>
		     <QTime>
		     
		     )

		    (split-header-and-code
		     (do0
		      (include <QMutex>
			       <QThread>
			       <QWaitCondition>))
		     (include "vis_01_serial.hpp"))
		    		    
		   (defclass SerialReaderThread "public QThread"
		     "Q_OBJECT"
		     "public:"
		     (defmethod SerialReaderThread (&key (parent nullptr))
		       (declare (type QObject* parent)
				(values :constructor)
				(construct (QThread parent))
				(explicit)))
		     (defmethod ~SerialReaderThread ()
		       (declare (values :constructor)
				)
		       (m_mutex.lock)
		       (setf m_quit true)
		       (m_mutex.unlock)
		       (wait))
		     (defmethod startReader (portName waitTimeout response)
		       (declare (type "const QString&" portName)
				(type int waitTimeout)
				(type "const QString&" response))
		       "const QMutexLocker locker(&m_mutex);"
		       ,@(loop for e in `(portName waitTimeout response)
			    collect
			      `(setf ,(format nil "m_~a" e)
				     ,e))
		       (unless (isRunning)
			 (start)))
		     (do0 "signals:"
			  #+nil ,@(loop for e in `(request error timeout)
			       collect
				 (format nil "void ~a(const QString &s);" e))
			  (do0
			   #+nil (defmethod request (s)
			     (declare (type "const QString&" s)))
			   (defmethod error (s)
			     (declare (type "const QString&" s)))
			   (defmethod timeout (s)
			     (declare (type "const QString&" s)))))
		     "private:"
		     (defmethod run ()
		       ,(logprint "" `())
		       (let ((currentPortNameChanged false))
			 (declare (type bool currentPortNameChanged))
			 (m_mutex.lock)
			 (let ((currentPortName))
			   (declare (type QString currentPortName))
			   (unless (== currentPortName m_portName)
			     (setf currentPortName m_portName
				   currentPortNameChanged true))
			   (let ((currentWaitTimeout m_waitTimeout)
				 (currentResponse m_response))
			     (m_mutex.unlock)
			     (let ((serial))
			       (declare (type QSerialPort serial))
			       (while (not m_quit)
				 (when currentPortNameChanged
				   ,(logprint "new port name" `())
				   (serial.close)
				   (serial.setPortName currentPortName)
				   ,@(loop for (e f) in `((BaudRate Baud115200)
							  (Parity NoParity)
							  (DataBits Data8)
							  (StopBits OneStop)
							  (FlowControl NoFlowControl))
					collect
					  `(do0
					    ,(logprint (format nil "~a = ~a" e f) `())
					    ((dot serial ,(format nil "set~a" e)) ,(format nil "QSerialPort::~a" f))))
				   (unless (serial.open QIODevice--ReadWrite)
				     (space emit (error (dot (tr (string "Cant open %1, error code %2"))
							     (arg m_portName)
							     (arg (serial.error))))) 
				     (return))
				   ,(logprint "open" `())
				   (if (serial.waitForReadyRead currentWaitTimeout)
				       (let ((requestData (serial.readAll)))
					 ,(logprint "readAll" `())
				       (while (serial.waitForReadyRead 10)
					 (incf requestData (serial.readAll)))
				       #+nil
				       (let ((responseData (currentResponse.toUtf8)))
					 (serial.write responseData)
					 (if (serial.waitForBytesWritten m_waitTimeout)
					     (let ((request (QString--fromUtf8 requestData)))
					       (space emit (this->request request)))
					     (space emit (timeout (dot (tr (string "Wait write response timeout %1"))
								       (arg (dot (QTime--currentTime)
										 (toString)))))))))
				       (space emit (timeout (dot (tr (string "Wait read request timeout %1"))
								       (arg (dot (QTime--currentTime)
										 (toString))))))
				       )
				   (m_mutex.lock)
				   (if (== currentPortName m_portName)
				       (do0
					(setf currentPortNameChanged false))
				       (do0
					(setf currentPortName m_portName
					      currentPortNameChanged true)))
				   (setf currentWaitTimeout m_waitTimeout
					 currentResponse m_response)
				   (m_mutex.unlock))))))))
		     "QString m_portName;"
		     "QString m_response;"
		     "int m_waitTimeout = 0;"
		     "QMutex m_mutex;"
		     "bool m_quit = false;"
		     )
		    		    
		   )))

    (define-module
       `(dialog ()
		(do0

		 (include "vis_01_serial.hpp")
	       (split-header-and-code
		(do0
		 (include <QtWidgets/QDialog>)
		 QT_BEGIN_NAMESPACE
		 ,@(loop for e in `(Label LineEdit ComboBox SpinBox PushButton) collect
			(format nil "class Q~a;" e))
		 QT_END_NAMESPACE)
		(do0
		 
		 (include "vis_02_dialog.hpp")
		      (include <QComboBox>
			       <QGridLayout>
			       <QLabel>
			       <QLineEdit>
			       <QPushButton>
			       <QSerialPortInfo>
			       <QSpinBox>))
		     )
	       ,(let ((l `((transactionCount int :value 0)
			  (serialPortLabel QLabel* :init (space new (QLabel (tr (string "Serial port:")))))
			  (serialPortComboBox QComboBox* :init (space new QComboBox))
			  (waitRequestLabel QLabel* :init (space new (QLabel (tr (string "Wait request, msec:")))))
			   (waitRequestSpinBox QSpinBox* :init (space new QSpinBox))
			  (responseLabel QLabel* :init (space new (QLabel (tr (string "Response:")))))
			  (responseLineEdit QLineEdit* :init (space new (QLineEdit (tr (string "hello ... ")))))
			  (trafficLabel QLabel* :init (space new (QLabel (tr (string "No traffic.")))))
			  (statusLabel QLabel* :init (space new (QLabel (tr (string "Status: Not running.")))))
			  (runButton QPushButton* :init (space new (QPushButton (tr (string "Start"))))))))
		 `(defclass Dialog "public QDialog"

		  Q_OBJECT
		  "public:"
		  (defmethod Dialog (&key (parent nullptr))
		    (declare (type QWidget* parent)
			     (values :constructor)
			     (construct (QDialog parent)
					,@(remove-if #'null
						     (loop for e in l
							collect
							  (destructuring-bind (name type &key (value 'nullptr) init) e
							    (when init
							      `(,(format nil "m_~a" name)
								 ,init))))))
			     (explicit))
		    (m_waitRequestSpinBox->setRange 0 10000)
		    (m_waitRequestSpinBox->setValue 10000)
		    (let ((infos (QSerialPortInfo--availablePorts)))
		      (for-range (&info infos)
				 (m_serialPortComboBox->addItem (info.portName))))
		    (let ((mainLayout (new QGridLayout)))
		      ,@(loop for e in `((serialPortLabel 0 0)
					 (serialPortComboBox 0 1)
					 (waitRequestLabel 1 0)
					 (waitRequestSpinBox 1 1)
					 (runButton 0 2 2 1)
					 (responseLabel 2 0)
					 (responseLineEdit 2 1 1 3)
					 (trafficLabel 3 0 1 4)
					 (statusLabel 4 0 1 5)
					 )
			   collect
			     (destructuring-bind (name &rest rest) e
				 `(mainLayout->addWidget ,(format nil "m_~a" name)
							 ,@rest)))
		      (setLayout mainLayout)

		      (setWindowTitle (tr (string "Blocking Serial Reader")))
		      (m_serialPortComboBox->setFocus)
		      ;; receiver is always this
		      ,@(loop for e in `((m_runButton &QPushButton--clicked &Dialog--startReader)
					 ;(&m_thread &SerialReaderThread--request &Dialog--showRequest)
					 ;(&m_thread &SerialReaderThread--error &Dialog--processError)
					 ;(&m_thread &SerialReaderThread--timeout &Dialog--processTimeout)
					 (m_serialPortComboBox ("QOverload<const QString&>::of"
								 &QComboBox--currentIndexChanged)
								&Dialog--activateRunButton)
					 (m_waitRequestSpinBox &QSpinBox--textChanged &Dialog--activateRunButton)
					 (m_responseLineEdit &QLineEdit--textChanged &Dialog--activateRunButton)
					 )
			   collect
			     (destructuring-bind (sender signal method) e
			      `(connect ,sender ,signal this ,method)))))
		  "private slots:"
		  (defmethod startReader ()
		    (m_runButton->setEnabled false)
		    (m_statusLabel->setText (dot (tr (string "Status: Running, connected to port %1.")
						     )
						 (arg (m_serialPortComboBox->currentText))))
		    (m_thread.startReader (m_serialPortComboBox->currentText)
					 (m_waitRequestSpinBox->value)
					 (m_responseLineEdit->text)))
		  (defmethod showRequest (s)
		    (declare (type QString& s))
		    (m_trafficLabel->setText (dot (tr (string "Traffic, transaction #%1:\\n\\r-request: %2\\n\\r-response: %3"))
						  (arg (incf m_transactionCount))
						  (arg s)
						  (arg (m_responseLineEdit->text)))))
		  (defmethod processError (s)
		    (declare (type QString& s))
		    (activateRunButton)
		    (do0
		     (m_statusLabel->setText (dot (tr (string "Status: Not running, %1."))
						 
						  (arg s)
						  ))
		     (m_trafficLabel->setText (tr (string "No traffic."))
					      )))
		  (defmethod processTimeout (s)
		    (declare (type QString& s))
		    (do0
		     (m_statusLabel->setText (dot (tr (string "Status: Running, %1."))
						  (arg s)))
		     (m_trafficLabel->setText (tr (string "No traffic."))
					      )))
		  (defmethod activateRunButton ()
		    (m_runButton->setEnabled true))
		  "private:"
		  ,@(loop for e in l
		       collect
			 (destructuring-bind (name type &key (value 'nullptr) init) e
			   (format nil "~a m_~a~@[=~a~];" type name value)))
		  "SerialReaderThread m_thread;"))
	       
		    ))))
  
  (progn
    (progn ;with-open-file
      #+nil (s (asdf:system-relative-pathname 'cl-cpp-generator2
					(merge-pathnames #P"proto2.h"
							 *source-dir*))
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



