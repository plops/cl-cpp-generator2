(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/109_clang_qt_mod_portable/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
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
    (merge-pathnames #P"std_mod.hpp"
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
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
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
	      (let ((*centralWidget (new QWidget)))
		(centralWidget->setLayout mainLayout)
		(window.setCentralWidget centralWidget))
	      
					;(window.setLayout mainLayout)
	      (window.show)
	      (return (app.exec))))))))  )


  
