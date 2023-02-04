(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/114_modern_sdl_av/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  
  
  
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"c_resource.hpp"
		     *source-dir*))
   `(do0
     "#pragma once"
     (include<> concepts
		cstring
		type_traits)
     (do0
      "template <typename T> constexpr inline T *c_resource_null_value=nullptr;"
      (comments "two api schemas for destructors and constructor"
		"1) thing* construct();      void destruct(thing*)"
		"2) void construct(thing**); void destruct(thing**)")
      ))))


  
