(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  ;; for classes with templates use write-source and defclass+
  ;; for cpp files without header use write-source
  ;; for class definitions and implementation in separate h and cpp file
  (defparameter *source-dir* #P"example/99_qt_pch/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  #+nil
  (let ((name `AGameCharacter))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include "bla.h"))
     :implementation-preamble `(do0
				(include "bah.h"))
     :code `(do0
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name ()
		 (declare
		  (construct
		   (Camera 3))
		  (values :constructor)))))))

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
      )

     (do0
      (include <spdlog/spdlog.h>)
					;(include <popl.hpp>)

      (comments "precompiled header is 154MB")
      )))


  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (include
      ,@(loop for e in `(main-window
			 double-validator
			 group-box)
	      collect
	      (format nil "<~a>"
		      (cl-change-case:pascal-case
		       (format nil "q-~a" e))))

      
      )

     (defun convertTemperature (outputLineEdit
				outputComboBox
				inputUnit
				inputTemp)
       (declare (type QLineEdit* outputLineEdit)
		(type QComboBox* outputComboBox)
		(type "const QString&" inputUnit)
		(type double inputTemp))
      (let ((outputTemp (double 0d0)))
	(if (== (string "Celsius")
		inputUnit)
	    (do0
	     (setf outputTemp inputTemp))
	    (if (== (string "Fahrenheit")
		inputUnit)
	    (do0
	     (setf outputTemp (/ (* 5 (- inputTemp 32))
				 9)))
	    (if (== (string "Kelvin")
		inputUnit)
	    (do0
	     (setf outputTemp (- inputTemp 273.15d0)))
	    )))
	(let ((outputUnit (outputComboBox->currentText)))
	  (if (== (string "Celsius")
		outputUnit)
	    (do0
	     (setf outputTemp outputTemp))
	    (if (== (string "Fahrenheit")
		outputUnit)
	    (do0
	     (setf outputTemp (+ 32 (/ (* 9 outputTemp) 5))))
	    (if (== (string "Kelvin")
		inputUnit)
	    (do0
	     (setf outputTemp (+ outputTemp 273.15d0)))
	    )))
	  )
	(-> outputLineEdit
	    (setText (QString--number outputTemp))))
       )
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       "(void)argv;"
       ,(lprint :msg "start" :vars `(argc))
       (let ((app (QApplication argc argv))
	     (window (QMainWindow)))
	 (window.setWindowTitle (string "Temperature converter"))
	 ,(flet ((add-widgets (dst-srcs)
		   (destructuring-bind (dst srcs) dst-srcs
		     `(do0
		       ,@(loop for src in srcs
			       collect
			       `(-> ,dst
				    (addWidget ,src)))))))
	    `(do0

	      (let ((*mainLayout (new QVBoxLayout))
		    (*inputGroupBox (new (QGroupBox (string "Input"))))
		    (*inputLayout (new QHBoxLayout)))
		(let ((*inputLineEdit (new QLineEdit))
		      (*inputValidator (new QDoubleValidator)))
		  (inputLineEdit->setValidator inputValidator))

		(let ((*inputComboBox (new QComboBox)))
		  ,@(loop for e in `(Celsius
				     Fahrenheit
				     Kelvin)
			  collect
			  `(-> inputComboBox
			       (addItem (string ,e)))))

		,(add-widgets `(inputLayout (inputLineEdit inputComboBox)))
		(-> inputGroupBox
		    (setLayout inputLayout))
		,(add-widgets `(mainLayout (inputGroupBox)))
	     
		)

	      (do0 (let ((*outputGroupBox (new (QGroupBox (string "Output"))))
			 (*outputLayout (new QHBoxLayout)))
		     (let ((*outputLineEdit (new QLineEdit))
			   )
		       (-> outputLineEdit (setReadOnly true)))
		     )
		   (let ((*outputValidator (new QDoubleValidator)))
		     (-> outputLineEdit (setValidator outputValidator))))

	      (do0
	       (let ((*outputComboBox (new  QComboBox)))
		 ,@(loop for e in `(Celsius
				    Fahrenheit
				    Kelvin)
			 collect
			 `(-> outputComboBox
			      (addItem (string ,e))))))

	      ,(add-widgets `(outputLayout (outputLineEdit
					    outputComboBox)))
	      (-> outputGroupBox (setLayout outputLayout))
	      ,(add-widgets `(mainLayout (outputGroupBox)))

	      (let ((*convertButton (new (QPushButton (string "Convert")))))
		(QObject--connect
		 convertButton
		 &QPushButton--clicked
		 (lambda ()
		   (declare (capture "="))
		   (let ((inputTemp (dot (-> inputLineEdit
					     (text))
					 (toDouble)))
			 (inputUnit (-> inputComboBox
					(currentText))))
		     (convertTemperature
		      outputLineEdit
		      outputComboBox
		      inputUnit
		      inputTemp
		      ))))
		)

	      ,(add-widgets `(mainLayout
			      (convertButton)))
	      (window.setLayout mainLayout)
	      (window.show)
	      (return (app.exec))

	    
	    

	      ))
	 )
       )))

  (with-open-file (s (format nil "~a/CMakeLists.txt" *full-source-dir*)
		     :direction :output
		     :if-exists :supersede
		     :if-does-not-exist :create)
		  (let ((l-dep `(spdlog Qt5Gui Qt5Core Qt5Widgets)))
		    (macrolet ((out (fmt &rest rest)
				    `(format s ,(format nil "~&~a~%" fmt) ,@rest))
			       )
			      (out "cmake_minimum_required( VERSION 3.16 FATAL_ERROR )")
			      (out "project( mytest )")

			      ;;(out "set( CMAKE_CXX_COMPILER clang++ )")
			      ;;(out "set( CMAKE_CXX_COMPILER g++ )")
			      (out "set( CMAKE_VERBOSE_MAKEFILE ON )")

			      (out "set( SRCS ~{~a~^~%~} )"
				   (append
				    (directory (format nil "~a/*.cpp" *full-source-dir*))
				    ))

			      (out "add_executable( mytest ${SRCS} )")
			      (out "set_property( TARGET mytest PROPERTY CXX_STANDARD 20 )")

		      ;;(out "target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )")

			      (out "find_package( PkgConfig REQUIRED )")
			      (loop for e in l-dep
				    do
				    (out "pkg_check_modules( ~a REQUIRED ~a )" e e))

			      (out "target_include_directories( mytest PUBLIC ~{${~a_INCLUDE_DIRS}~^ ~} )" l-dep)
			      (out "target_compile_options( mytest PUBLIC ~{${~a_CFLAGS_OTHER}~^ ~} )" l-dep)

		      ;; (out "set_property( TARGET mytest PROPERTY POSITION_INDEPENDENT_CODE ON )")
		      ;(out "set( CMAKE_POSITION_INDEPENDENT_CODE ON )")
		      #+nil
			      (progn
				(out "add_library( libnc SHARED IMPORTED )")
				(out "set_target_properties( libnc PROPERTIES IMPORTED_LOCATION /home/martin/stage/cl-cpp-generator2/example/88_libnc/dep/libnc-2021-04-24/libnc.so )"))
			      (out "target_link_libraries( mytest PRIVATE ~{${~a_LIBRARIES}~^ ~} )"
				   l-dep)
			      (out "target_precompile_headers( mytest PRIVATE fatheader.hpp )")))))

