(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-change-case")
  (ql:quickload "cl-cpp-generator2")
  )

(defpackage #:my-cpp-project
  (:use #:cl #:cl-cpp-generator2)) 

(in-package #:my-cpp-project)

;; wget https://www.lua.org/ftp/lua-5.4.8.tar.gz; tar xaf lua-5.4.8.tar.gz


(let ()
  (defparameter *source-dir* #P"example/190_lua/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))
   (write-source
   "src/main.cpp"
   `(do0
     (include<> cstdio vector array cstdint)
     (comments "check luaconf.h, LUA_32BITS LUA_NUMBER=float LUA_INT_TYPE=long LUA_IDSIZE reduce debug info to save memory")
     (space extern "\"C\"" (progn
			     (include lua.h lualib.h lauxlib.h)))

     (do0
      "constexpr size_t LUA_HEAP_SIZE = 16 * 1024;"
      "static std::array<uint8_t, LUA_HEAP_SIZE> lua_memory_pool;"
      "static size_t lua_mem_used = 0;")
     (defun custom_alloc (ud ptr osize nsize)
       (declare (type void* ud ptr)
		(type size_t osize nsize)
		(values "static void*"))
       (when (== 0 nsize)
	 (return nullptr))
       (when (< LUA_HEAP_SIZE (+ lua_mem_used nsize))
	 (return nullptr))
       (setf "void* new_ptr"
	     (aref &lua_memory_pool lua_mem_used))
       (incf lua_mem_used nsize)
       (return new_ptr))
     (defun main ()
       (declare (values int))
       (return 0)))
   :omit-parens t
   :dir *full-source-dir*))
