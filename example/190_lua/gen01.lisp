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
  (defun lprint (&key (msg "")
		 (vars nil)
		 )
    `(<< std--cout
       (string ,(format nil "~a"
			msg
			
			))
       ,@(loop for e in vars
	       appending
	       `((string ,(format nil " ~a='" (emit-c :code e :omit-redundant-parentheses t)))
		 ,e
		 (string "' ")))   
       std--endl))
  (defparameter *source-dir* #P"example/190_lua/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))
   (write-source
   "src/main.cpp"
   `(do0
     (include<> cstdio vector array cstdint iostream)
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

     (defun panic_handler (L)
       (declare (type lua_State* L)
		(values "static int"))
       (let ((*msg (lua_tostring L -1))))
       ,(lprint :vars `(msg))
       (while true)
       (return 0))

     (defun native_add (L)
       (declare (type lua_State* L)
		(values "static int"))
       (let ((a (luaL_checknumber L 1))
	     (b (luaL_checknumber L 2)))
	 (lua_pushnumber L (+ a b))
	 (return 1)))
     
     (defun main ()
       (declare (values int))
       (let ((*L (lua_newstate custom_alloc nullptr)))
	 (when (== nullptr L)
	   ,(lprint :msg "Failed to init Lua state (OOM)")
	   (return -1)))
       (lua_atpanic L panic_handler)
       (comments "lade nur was noetig ist, kein luaL_openlibs")
       (luaL_requiref L (string "_G")
		      luaopen_base 1)
       (lua_pop L 1)
       (lua_pushcfunction L native_add)
       (lua_setglobal L (string "cpp_add"))
       (let ((*lua_script (string-r "local x = 10
local y = 20
local sum = cpp_add(x,y)
print('Sum: ' .. sum)
lua_status = 'OK'"))))
       (let ((status (luaL_dostring L lua_script))))
       (if (== LUA_OK status)
	   (do0
	    (lua_getglobal L (string lua_status))
	    (when (lua_isstring L -1)
	      (let ((stat (lua_tostring L -1)))
		,(lprint :vars `(stat))))
	    (lua_pop L 1))
	   (do0
	    (when (lua_isstring L -1)
	      (let ((err (lua_tostring L -1)))
		,(lprint :vars `(err))))
	    (lua_pop L 1))
	   )
       (lua_close L)
       (return 0)))
   :omit-parens t
   :dir *full-source-dir*))
