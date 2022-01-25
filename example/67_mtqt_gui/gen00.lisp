(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     ;(ql:quickload "cl-ppcre")
     ;(ql:quickload "cl-change-case")
     ) 
(in-package :cl-cpp-generator2)
(progn
  (defparameter *source-dir* #P"example/67_mtqt_gui/source/")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (msg &optional rest)
      `(progn				;do0
	 " "
	 #-nolog
	 (do0 ;let
	  #+nil ((lock (std--unique_lock<std--mutex> ,(g `_stdout_mutex)))
	   )
	  
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
	       ;(string " ")
	       ;(string ,msg)
	       (string " ")
	       ,@(loop for e in rest appending
		       `(("std::setw" 8)
					;("std::width" 8)
			 (string ,(format nil " ~a='" (emit-c :code e)))
			 ,e
			 (string "'")))
	       "std::endl"
	       "std::flush")))))
  (let ((type-definitions
	  `(do0

	    (defclass AnyQAppLambda ()
	      "public:"
	      (defmethod run ()
		(declare (virtual)
			 (pure)))
	      (defmethod ~AnyQAppLambda ()
		(declare (virtual)
			 (values :constructor))))
	   
	    


	    (defclass AnyQAppLambdaEvent "public QEvent"
		      "public:"
		      "AnyQAppLambda* al=nullptr;"
		      (defmethod AnyQAppLambdaEvent (al)
			(declare (type AnyQAppLambda* al)
				 ;; event id a random number between 1000 and ushort max
				 (construct (QEvent (QEvent--Type 48301))
					    (al al))
				 (values :constructor)))
		      (defmethod ~AnyQAppLambdaEvent ()
			(declare
			 (virtual)
			 (values :constructor)
			 )
			(unless (== nullptr al)
			  (-> al (run)))
			(delete al)))

	    (defclass BlockingEvent "public AnyQAppLambdaEvent"
	      "public:"
	      "std::atomic<bool>* done;"
	      (defmethod BlockingEvent (al  done)
		(declare (type AnyQAppLambda* al)
			 (type std--atomic<bool>* done)
			 (constr (AnyQAppLmabdaEvent al)
				 (done done))
			 (values :constructor)))
	      (defmethod ~BlockingEvent ()
		(declare  
		 (values :constructor))
		(unless (== nullptr this->al)
		  (-> this
		      al
		      (run)))
		(delete al)
		(setf al nullptr)
		(-> done (store true))))

	    (defclass QApplicationManager ()
	      "public:"
	      "std::shared_ptr<std::atomic<bool>> done = std::make_shared<std::atomic<bool>>(false);"
	      "bool we_own_app = true;"
	      "std::thread thr;"
	      "QCoreApplication* app = nullptr;"
	      (defmethod QApplicationManager ()
		(declare (values :constructor))
		(when we_own_app
		  (quit)
		  (when (thr.joinable)
		    (thr.join))))
	      (defmethod create (argc argv)
		(declare (static)
			 (type int argc)
			 (type char** argv)
			 (values "std::shared_ptr<QApplicationManager>"))
		(let ((qm (std--make_shared<QApplicationManager>)))
		  (unless (== nullptr (QApplication--instance))
		    (setf qm->we_own_app false
			  qm->app (QCoreApplication--instance))
		    ,(lprint "we are not managing this qapp instance.")
		    (-> qm
			app
			(postEvent qm->app
				   (new (AnyQAppLambdaEvent
					 (new QAppLambda
					      (lambda ()
						(declare (capture qm))
						(QObject--connect qm->app
								  &QApplication--aboutToQuit
								  qm->app
								  (lambda ()
								    (declare (capture qm))
								    (setf *qm->done true)))))))))
		    (return qm))
		  (let ((ready false))
		    (declare (type std--atomic<bool> ready))
		    (setf qm->thr
			  (std--thread
			   (lambda ()
			     (declare (capture "&"))
			     (setf qm->app (new ("class QApplication" argc argv)))
			     ;; qm captured by copy, continues to exist till closure is finished (QApplication is deleted)
			     (QObject--connect qm->app
					       &QApplication--aboutToQuit
					       qm->app
					       (lambda ()
						 (declare (capture qm))
						 (setf *qm->done true)))
			     (setf ready true)
			     (qm->app->exec)
			     )))
		    (while (not ready)
			   (std--this_thread--sleep_for
			    (std--chrono--milliseconds 50)))
		    (return qm))))
	      (defmethod wait_for_finished ()
		(while (not *done)
		       (std--this_thread--sleep_for
			    (std--chrono--milliseconds 50))
		       ))
	      )
	    (do0
	       "std::mutex QApp_mtx;"
	       "std::shared_ptr<QApplicationManager> qm = nullptr;"
	       (defmethod qapplication_manager (&key (argc 0) (argv nullptr))
		 (declare (type int argc)
			  (type char** argv)
			  (values 
			   "std::shared_ptr<QApplicationManager>"))
		 "std::unique_lock<std::mutex> ul(QApp_mtx);"
		 (when (== nullptr qm)
		   (setf qm (QApplicationManager--create argc argv))
		   (return qm))))
	    
	    (defun qapplication (&key (argc 0) (argv nullptr))
	      (declare (type int argc)
		       (type char** argv)
		       (values QCoreApplication*))
	      (return (-> (qapplication_manager argc argv)
			  app)))
	    (defun wait_for_qapp_to_finish ()
	      (-> (qapplication_manager)
		  (wait_for_finished)))
	    (defun run_in_gui_thread (re)
	      (declare (type AnyQAppLambda* re))
	      (let ((qm (qapplication)))
		(qm->postEvent qm (new (AnyQAppLambdaEvent re)))))

	    (defun run_in_gui_thread_blocking (re)
	      (declare (type AnyQAppLambda* re))
	      
	      (let ((done false)
		    (qm (qapplication)))
		(declare (type std--atomic<bool> done))
		(qm->postEvent qm (new (BlockingEvent re &done)))
		(while (not done)
		       (std--this_thread--sleep_for
			    (std--chrono--milliseconds 50)))))
	    
	    (defun quit ()
	      (let ((app (-> (qapplication_manager)
			     app)))
		(run_in_gui_thread_blocking
		 (new (QAppLambda (lambda ()
				    (declare (capture app))
				    (-> app (quit))))))))
	    (defun wait_key ()
	      (declare (values "unsigned char")))
	    )))

    (let ((fn-h (asdf:system-relative-pathname
			  'cl-cpp-generator2
			  (merge-pathnames #P"mtgui.h"
					   *source-dir*))))
     (with-open-file (sh fn-h
			 :direction :output
			 :if-exists :supersede
			 :if-does-not-exist :create)
       (emit-c :code
	       `(do0
		 "#pragma once"
		 (include <tuple>
			  <mutex>
		     <thread>
		     <QEvent>
		     <QApplication>
		     <iostream>
		     <iomanip>
		     <chrono>
		     <memory>
			  )
		 "class AnyQAppLambda;"
		 "class QCoreApplication;"
		 ,type-definitions)
	       :hook-defun #'(lambda (str)
                               (format sh "~a~%" str))
               :hook-defclass #'(lambda (str)
                                  (format sh "~a;~%" str))
	       :header-only t))
     (sb-ext:run-program "/usr/bin/clang-format"
                         (list "-i"  (namestring fn-h))))
    
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"mtgui_template.h"
				    *source-dir*))
		  `(do0
		 
		  #+nil  (include <mtgui.h>
			     )

		    (defclass+ (QAppLambda :template "class Lambda, class... Args") AnyQAppLambda
	      "public:"
	      ,@(loop for (e f) in `((Lambda lambda)
				     (std--tuple<Args...> args)
				     )
		      collect
		      (format nil "~a ~a;" (emit-c :code  e) f))
	      (defmethod QAppLambda (lambda args)
		(declare (type Lambda lambda)
			 (type Args... args)
			 (constr (AnyQAppLambda)
				 (lambda lambda)
				 (args (std--make_tuple args...))
				 )
			 (values :constructor)))
	      (defmethod run ()
		(declare (override))
		(run_impl ("std::make_index_sequence<sizeof...(Args)>")))
	      (defmethod run_impl (
				   <I...>)
		(declare (type "std::index_sequence" <I...>)
			 (values "template<std::size_t... I> void"))
		("lambda" (space (std--get<I> args)
			 "...")))
	      )))

    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"mtgui.cpp"
				    *source-dir*))
		  `(do0
		 
		    (include <mtgui.h>
			     <mtgui_template.h>
			     )
		    
		    
		    ,type-definitions

		    

		    ))
    (write-source (asdf:system-relative-pathname
		  'cl-cpp-generator2
		  (merge-pathnames #P"mtgui_driver.cpp"
				   *source-dir*))
		  
		  `(do0
		    (include
		     <mtgui.h>
		     )
		    
		   
		   (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      )
		   )))
  
  (with-open-file (s "source/CMakeLists.txt" :direction :output
					     :if-exists :supersede
					     :if-does-not-exist :create)
    (macrolet ((out (fmt &rest rest)
		 `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
      (out "cmake_minimum_required( VERSION 3.4 )")
      (out "project( mytest LANGUAGES CXX )")
      (out "set( CMAKE_CXX_COMPILER clang++ )")
      (out "set( CMAKE_CXX_FLAGS \"\"  )")
      (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
      ;(out "set( CMAKE_CXX_STANDARD 23 )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")
      
					;(out "set( CMAKE_CXX_FLAGS )")
      (out "find_package( Qt5 5.9 REQUIRED Core Gui Widgets )")
      (out "set( SRCS ~{~a~^~%~} )"
	   (directory "source/*.cpp"))
      (out "add_executable( mytest ${SRCS} )")
      (out "target_compile_features( mytest PUBLIC cxx_std_20 )")
      (out "target_include_directories( mytest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )")
      
      (out "target_link_libraries( mytest PRIVAT Qt5::Core Qt5::Gui Qt5::Widgets )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
      )
    ))

 

