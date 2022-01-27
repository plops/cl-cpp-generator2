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
  (defun lprint (&key (msg "") (vars nil))
    #+nil `(comments ,msg)
    #-nil`(progn				;do0
	 " "
	 (do0				;let
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
	       (string " ")
	       (string ,msg)
	       (string " ")
	       ,@(loop for e in vars appending
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
			 (pure))
		,(lprint))
	      (defmethod ~AnyQAppLambda ()
		(declare (virtual)
			 (values :constructor))
		,(lprint)))
	    (defclass AnyQAppLambdaEvent "public QEvent" 
		      "public:"
		      "AnyQAppLambda* al=nullptr;"
		      (defmethod AnyQAppLambdaEvent (al)
			(declare (type AnyQAppLambda* al)
				 ;; event id a random number between 1000 and ushort max
				 (construct (QEvent (QEvent--Type 48301))
					    (al al))
				 (values :constructor))
			)
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
			 (construct (AnyQAppLambdaEvent al)
				 (done done))
			 (values :constructor))
		,(lprint))
	      (defmethod ~BlockingEvent ()
		(declare  
		 (values :constructor))
		,(lprint)
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
	      (defmethod ~QApplicationManager ()
		(declare (values :constructor))
		;,(lprint :vars `(we_own_app))
		(when we_own_app
		  (quit)
		  (when (thr.joinable)
		    (thr.join))))
	      (defmethod create (argc argv)
		(declare (static)
			 (type int argc)
			 (type char** argv)
			 (values "std::shared_ptr<QApplicationManager>"))
		;,(lprint)
		(let ((qm (std--make_shared<QApplicationManager>)))
		  (unless (== nullptr (QApplication--instance))
		    (setf qm->we_own_app false
			  qm->app (QCoreApplication--instance))
		    ,(lprint :msg "we are not managing this qapp instance.")
		    (-> qm
			app
			(postEvent qm->app
				   (new (AnyQAppLambdaEvent
					 (new (QAppLambda
					       (lambda ()
						 (declare (capture qm))
						 (QObject--connect qm->app
								   &QApplication--aboutToQuit
								   qm->app
								   (lambda ()
								     (declare (capture qm))
								     (setf *qm->done true))))))))))
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
	     (do0
	      "std::mutex QApp_mtx;"
	      "std::shared_ptr<QApplicationManager> qm = nullptr;"
	      (defun qapplication_manager (&key (argc 0) (argv nullptr))
		(declare (type int argc)
			 (type char** argv)
			 (values 
			  "std::shared_ptr<QApplicationManager>"))
		,(lprint :msg "request lock" )
		"std::unique_lock<std::mutex> ul(QApp_mtx);"
		,(lprint :msg "have lock")
		(when (== nullptr qm)
		  (setf qm (QApplicationManager--create argc argv))
		  )
		(return qm))))
	    
	    (defun qapplication (&key (argc 0) (argv nullptr))
	      (declare (type int argc)
		       (type char** argv)
		       (values QCoreApplication*))
	      ,(lprint)
	      (return (-> (qapplication_manager argc argv)
			  app)))
	    (defun wait_for_qapp_to_finish ()
	      ,(lprint)
	      (-> (qapplication_manager)
		  (wait_for_finished)))
	    (defun run_in_gui_thread (re)
	      (declare (type AnyQAppLambda* re))
	      ,(lprint)
	      (let ((qm (qapplication)))
		(qm->postEvent qm (new (AnyQAppLambdaEvent re)))))

	    (defun run_in_gui_thread_blocking (re)
	      (declare (type AnyQAppLambda* re))
	      ,(lprint)
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
	      (declare (values "unsigned char"))
	      (return 0))
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
		 (pragma once)
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
		    (pragma once)
		    (defclass+ (QAppLambda :template "class Lambda, class... Args") "public AnyQAppLambda"
	      "public:"
	      ,@(loop for (e f) in `((Lambda lambda)
				     (std--tuple<Args...> args))
		      collect
		      (format nil "~a ~a;" (emit-c :code  e) f))
	      (defmethod QAppLambda (lambda args)
		(declare (type Lambda lambda)
			 (type Args... args)
			 (construct (AnyQAppLambda)
				 ("lambda" lambda)
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
			     <mtgui_template.h>)
		    ,type-definitions))

    (let ((class-defs `(do0
			(include ,@(loop for e in `(mutex thread memory)
				       collect
				       (format nil "<~a>" e)))
		       (defclass Figure () 
			 "public:"
			 "JKQTPlotter plot;"
			 "std::map<std::string,JKQTPXYLineGraph*> graphs;"
			 (defmethod labeled_graph (label)
			   (declare (type std--string label)
				    (values JKQTPXYLineGraph*))
			   
			   (let ((it (graphs.find label)))
			     (unless (== (graphs.end)
					 it)
			       ,(lprint :msg "found" :vars `(label))
			       (return it->second))
			     ,(lprint :msg "create" :vars `(label))
			     (let ((graph (new (JKQTPXYLineGraph &plot))))
			       (graph->setTitle (QObject--tr (label.c_str)))
			       (setf (aref graphs label)
				     graph)
			       (return graph))))
			 (defmethod set (xs ys label)
			   (declare (type "const std::vector<double>&" xs ys)
				    (type std--string label)
				    )
			   ,(lprint)
			   (let ((ds (plot.getDatastore))
				 (X (QVector<double> (xs.size)))
				 (Y (QVector<double> (ys.size)))
				 (contains_nan false))
			     (dotimes (i (std--min (xs.size)
						   (ys.size)))
			       (when (std--isnan (+ (aref xs i)
						    (aref ys i)))
				 (setf contains_nan true)
				 continue)
			       (<< X (aref xs i))
			       (<< Y (aref ys i)))
			     (let ((columnX (ds->addCopiedColumn X (string "x")))
				   (columnY (ds->addCopiedColumn Y (string "y")))
				   (graph (labeled_graph label)))
			       (graph->setXColumn columnX)
			       (graph->setYColumn columnY)
			       (plot.addGraph graph)
			       (plot.zoomToFit)))))
		       (defclass Plotter ()
			 "public:"
			 (defmethod clear_plot (title)
			   (declare (type std--string title))
			   ,(lprint)
			   (let ((plotter (self.lock)))
			     (run_in_gui_thread
			      (new (QAppLambda
				    (lambda ()
				      (declare (capture plotter title))
				      (plotter->clear_plot_internal title)))))))
			 (defmethod delete_plot (title)
			   (declare (type std--string title))
			   ,(lprint)
			   (let ((plotter (self.lock)))
			     (run_in_gui_thread
			      (new (QAppLambda
				    (lambda ()
				      (declare (capture plotter title))
				      (plotter->delete_plot_internal title)))))))
			 (defmethod plot (xs ys title label)
			   (declare (type "const std::vector<double>&" xs ys)
				    (type std--string title label))
			   ,(lprint)
			   (call_plot xs ys title label))
			 (defmethod create ()
			   (declare (values "std::shared_ptr<Plotter>")
				    (static))
			   ,(lprint)
			   (let ((plotter (std--shared_ptr<Plotter> (new Plotter))))
			     (setf plotter->self plotter)
			     (return plotter)))
			 (defmethod Plotter ()
			   (declare (values :constructor)))
			 "std::weak_ptr<Plotter> self;"
			 "std::mutex call_plot_mutex;"
			 (defmethod call_plot (xs ys title label)
			   (declare (type "const std::vector<double>&" xs ys)
				    (type std--string title label))
			   ,(lprint)
			   (let ((ul (std--unique_lock<std--mutex> call_plot_mutex))
				 (plotter (self.lock)))
			     ;; fixme: something about blocking and problem with opencv
			     (run_in_gui_thread
			      (new (QAppLambda
				    (lambda ()
				      (declare (capture plotter xs ys title label))
				      (plotter->plot_internal xs ys title label))) ))
			     ))
			 "std::map<std::string, std::shared_ptr<Figure>> plots_;"
			 (defmethod named_plot (name)
			   (declare (type std--string name)
				    (values "std::shared_ptr<Figure>"))
			   
			   (let ((it (plots_.find name)))
			     (unless (== it (plots_.end))
			       ,(lprint :msg "found existing" :vars `(name))
			       (return it->second))
			     ,(lprint :msg "create new figure" :vars `(name))
			     (let ((figure (std--make_shared<Figure>)))
			       ,(lprint :msg "store figure")
			       (setf (aref plots_ name) figure)
			       (let ((&plot figure->plot))
				 ,(lprint :msg "set title..")
				 (plot.setWindowTitle (QString (name.c_str)))
				 ,(lprint :msg "show..")
				 (plot.show)
				 ,(lprint :msg "resize..")
				 (plot.resize 600 400)
				 (return figure)))))
			 (defmethod clear_plot_internal (title)
			   (declare (type std--string title))
			   ,(lprint)
			   (let ((it (plots_.find title)))
			     (unless (== it (plots_.end))
			       (-> it
				   second
				   (plot.clearGraphs))
			       
			       (let ((plot_use_count (dot (-> it
							  second
							 
							  )
					       		  (use_count))))
				 ,(lprint :msg "\\033[1;31m CLEAR \\033[0m" :vars `((-> it first)
										    plot_use_count))))))
			 (defmethod delete_plot_internal (title)
			   (declare (type std--string title))
			   ,(lprint)
			   (let ((it (plots_.find title)))
			     (unless (== it (plots_.end))
			       
			       (setf (-> it second) nullptr)
			       (let ((plot_use_count (dot (-> it
							  second)
					       		  (use_count))))
				 ,(lprint :msg "\\033[1;31m CLEAR \\033[0m" :vars `((-> it first)
						  plot_use_count))))))
			 "std::mutex plot_internal_mtx;"
			 (defmethod plot_internal (xs ys title label)
			   (declare (type "const std::vector<double>&" xs ys)
				    (type std--string title label))
			   ,(lprint)
			   (let ((ul (std--unique_lock<std--mutex> plot_internal_mtx))
				 (figure (std--shared_ptr<Figure> (named_plot title))))
			     (figure->set xs ys label)
			     )
			   ))
		       (do0
		       "std::mutex plotter_mtx;"
		       "std::shared_ptr<Plotter> plotter_ = nullptr;"
		       (defun plotter ()
			 (declare (values "std::shared_ptr<Plotter>"))
			 ,(lprint)
			 (let ((ul (std--unique_lock<std--mutex> plotter_mtx)))
			   (when (== nullptr plotter_)
			     (setf plotter_ (Plotter--create)))
			   (return plotter_)))
		       (defun clear_plot (title)
			 (declare (type std--string title))
			 ,(lprint)
			 (-> (plotter)
			     (clear_plot title)))
		       (defun delete_plot (title)
			 (declare (type std--string title))
			 ,(lprint)
			 (-> (plotter)
			     (delete_plot title)))
		       (defun plot (ys title label )
			 (declare (type "const std::vector<double>&"  ys)
				  (type std--string title label))
			 ,(lprint)
			 (let ((indexes (std--vector<double> (ys.size))))
			   (dotimes (i (ys.size))
			     (indexes.push_back i))
			   (-> (plotter)
			       (plot indexes ys title label))))
		       (defun plot (xs ys title label )
			 (declare (type "const std::vector<double>&" xs ys)
				  (type std--string title label))
			 ,(lprint)
			 (-> (plotter)
			     (plot xs ys title label)))
		       (defun initialize_plotter ()
			 ,(lprint)
			 (plotter))))
		      )
	    (source-name "plot")
	    (includes `(mtgui.h mtgui_template.h
				jkqtplotter/jkqtplotter.h
				jkqtplotter/graphs/jkqtpscatter.h)))
      (write-source (asdf:system-relative-pathname
		       'cl-cpp-generator2
		       (merge-pathnames (format nil "~a.cpp" source-name)
					*source-dir*))
		    `(do0
		      (include ,(format nil "<~a.h>" source-name))
		      ,class-defs
		      
		      
		      
		      ))
      (let ((fn-h (asdf:system-relative-pathname
		    'cl-cpp-generator2
		    (merge-pathnames
		     (format nil "~a.h" source-name)
		     *source-dir*))))
	 (with-open-file (sh fn-h
			     :direction :output
			     :if-exists :supersede
			     :if-does-not-exist :create)
	   (emit-c :code
		   `(do0
		     (pragma once)
		     (include ,@(loop for e in includes
				      collect
				      (format nil "<~a>" e)))
		     ,class-defs
		     )
		   :hook-defun #'(lambda (str)
				   (format sh "~a~%" str))
		   :hook-defclass #'(lambda (str)
                                      (format sh "~a;~%" str))
		   :header-only t))
	 (sb-ext:run-program "/usr/bin/clang-format"
                             (list "-i"  (namestring fn-h)))))
    
    
    (write-source (asdf:system-relative-pathname
		  'cl-cpp-generator2
		  (merge-pathnames #P"main.cpp"
				   *source-dir*))
		  
		  `(do0
		    (include
		     <thread>
		     ;<iostream>
		     <chrono>
		     ;<mtgui.h>
		     ;<mtgui_template.h>
		     <QApplication>
		     <QMainWindow>
		     <plot.h>
		     <cmath>
		     )
		    
		    #+nil (defun typical_qt_gui_app ()
		      (let ((i 0)
			    (qapp (QApplication i nullptr))
			    (window (QMainWindow)))
			,(lprint)
			(window.show)
			(qapp.exec)
			))
		    (defun thread_independent_qt_gui_app ()
		      (comments "no need to initialize qt")
		      ,(lprint :msg "first window")
		      #+nil (run_in_gui_thread
		       (new (QAppLambda (lambda ()
					  (let ((*window (new QMainWindow))
						)
					    ,(lprint :msg "show first window")
					    (window->show))))))
		      #+nil (do0,(lprint :msg "second window")
			  (run_in_gui_thread
			   (new (QAppLambda (lambda ()
					      (let ((*window (new QMainWindow))
						    )
						,(lprint :msg "show second window")
						(window->show)))))))
		      #+nil (do0 ,(lprint :msg "third window in its own thread")
			   (let ((thr (std--thread (lambda ()
						     (run_in_gui_thread
						      (new (QAppLambda (lambda ()
									 (let ((*window (new QMainWindow))
									       )
									   ,(lprint :msg "show third window")
									   (window->show))))))))))
			     ,(lprint :msg "wait for thread of third window")
			     (thr.join)))
		      #+nil(std--this_thread--sleep_for (std--chrono--milliseconds 30))
		      ;(wait_for_qapp_to_finish)
		      )
		    #+nil (defun external_app_gui ()
		      ,(lprint)
		      (let ((i 0)
			    (qapp (QApplication i nullptr))
			    (thr (std--thread (lambda ()
						(run_in_gui_thread
						 (new (QAppLambda
						       (lambda ()
							 (let ((*window (new QMainWindow)))
							   (window->show))))))))))
			(thr.join)
			(qapp.exec)
			))
		   (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		     ;(typical_qt_gui_app)
		     #+nil(thread_independent_qt_gui_app)
		     #-nil(do0
		     ;; (initialize_plotter)
		      ,(let ((n-points 100))
			 `(let (
				(xs (std--vector<double> ,n-points))
				(ys (std--vector<double> ,n-points))
				)
			    (dotimes (q 100)
			     (do0
			      (dotimes (i ,n-points)
				(setf (aref xs i) i
				      (aref ys i) (* (exp (* -.01 i)) (sin (* .4 (+ i q))))))
			      (plot xs ys (string "bla1") (string "bla2"))
			      (std--this_thread--sleep_for (std--chrono--milliseconds 30))
			      (clear_plot (string "bla1"))))
			    )))
					;(external_app_gui)
		     (delete_plot (string "bla1"))
		     (wait_for_qapp_to_finish)
		     
		     (return 0)
		      )
		   ))) 
  
  (with-open-file (s "source/CMakeLists.txt" :direction :output
					     :if-exists :supersede
					     :if-does-not-exist :create)
    ;;https://clang.llvm.org/docs/AddressSanitizer.html
    ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
    (let ((dbg "-ggdb -O1 -fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope "))
     (macrolet ((out (fmt &rest rest)
		  `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
       (out "cmake_minimum_required( VERSION 3.4 )")
       (out "project( mytest LANGUAGES CXX )")
       (out "set( CMAKE_CXX_COMPILER clang++ )")
					;(out "set( CMAKE_CXX_FLAGS \"\"  )")
       (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
       (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a \")" dbg)
       (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a \")" dbg )
					;(out "set( CMAKE_CXX_STANDARD 23 )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")
      
		 			;(out "set( CMAKE_CXX_FLAGS )")
       (out "find_package( Qt5 5.9 REQUIRED Core Gui Widgets PrintSupport )")
       (out "set( SRCS ~{~a~^~%~} )"
	    (directory "source/*.cpp"))
       (out "add_executable( mytest ${SRCS} )")
       (out "target_compile_features( mytest PUBLIC cxx_std_20 )")
       (out "target_include_directories( mytest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} /usr/local/include  )")
       (out "target_link_directories( mytest PUBLIC /usr/local/lib )")

       (out "find_package (Threads) ")
       ;; Core Gui Widgets PrintSupport Svg Xml OpenGL
       (out "target_link_libraries( mytest PRIVATE Qt5::Core Qt5::Gui Qt5::PrintSupport Qt5::Widgets Threads::Threads JKQTPlotterSharedLib_ )")
      
					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
       ))
    ))

 

