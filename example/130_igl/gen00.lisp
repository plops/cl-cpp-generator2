(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/130_igl/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
    
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     ,@(loop for e in `((GLFW_INCLUDE_NONE)
			(GLFW_EXPOSE_NATIVE_X11)
			(GLFW_EXPOSE_NATIVE_GLX)
			(USE_OPENGL_BACKEND 1)
			(ENABLE_MULTIPLE_COLOR_ATTACHMENTS 0)
			(IGL_FORMAT "fmt::format"))
	     collect
	     (destructuring-bind (name &optional (value "")) e
	       (format nil 
		       "#define ~a ~a" name value)))
     
     (include<>
      GLFW/glfw3.h
      GLFW/glfw3native.h
      cassert
      regex
      iostream
      igl/IGL.h
      igl/opengl/glx/Context.h
      igl/opengl/glx/Device.h
      igl/opengl/glx/HWDevice.h
      igl/opengl/glx/PlatformDevice.h
      )

     (include<> fmt/core.h)
     
     
     "static const uint32_t kNumColorAttachments = 1;"


     (setf "std::string codeVS"
	   (string-r ,(emit-c
		       :code
		       `(do0 "#version 460"
			     "layout (location=0) out vec3 color;"
			     (let ((pos ("vec2[3]" (vec2 -.6 -.4)
						   (vec2 .6 -.4)
						   (vec2 .0 .6))))
			       (declare (type (array "const vec2" 3) pos)))
			     ))))

     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"
              
       (return 0)))))


