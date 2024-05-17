(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ()
  (defparameter *source-dir* #P"example/150_flatbuffers/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)


  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "image.fbs"
		     *source-dir*))
   `(do0
     (space namespace MyImage)
     (space-n table Image
	    (progn
	      (space "width:uint")
	      (space "height:uint")
	      (aref "data:" ubyte)))
     (space root_type Image))
   :omit-parens t
   :format nil
   :tidy nil)
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
      fstream
      )
     (include image_generated.h)

     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))

       (let ((width 256)
	     (height 371)
	     (imageData (std--vector<uint8_t> (* width height)
					      128))
	     (builder (flatbuffers--FlatBufferBuilder))
	     (image (MyImage--CreateImageDirect builder width height &imageData)))
	 (builder.Finish image)
	 (let ((output (std--ofstream (string "image.bin")
				      std--ios--binary)))
	   (output.write ("reinterpret_cast<const char*>"
			  (builder.data))
			 (builder.GetSize))
	   (output.close))
	 )
       
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))
