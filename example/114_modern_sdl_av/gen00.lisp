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
  
  (load "util.lisp")
  
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
		"2) void construct(thing**); void destruct(thing**)"
		""
		"modifiers like replace(..) exist only for schema 2) act like auto construct(thing**)")
      (space
       "template <typename T, auto* ConstructFunction, auto* DestructFunction>"
       (defclass+ c_resource ()
	 "public:"
	 (using pointer T*
		const_pointer "std::add_const_t<T>*"
		element_type T)
	 "private:"
	 (using Constructor (decltype ConstructFunction)
		Destructor (decltype DestructFunction))
	 (let ((construct ConstructFunction)
	       (destruct DestructFunction)
	       (null c_resource_null_value<T>))
	   (declare (type "static constexpr Constructor" construct)
		    (type "static constexpr Destructor" destruct)
		    (type "static constexpr T*" null))
	   "struct construct_t {};")
	 "public:"
	 "static constexpr construct_t constructed = {};"
	 "[[nodiscard]] constexpr c_resource() noexcept = default;"
	 (space "[[nodiscard]] constexpr explicit c_resource(construct_t) requires std::is_invocable_r_v<T*,Constructor>:ptr_{construct()}"
		(progn
		  (<< std--cout (string "construct75")
		      std--endl)))
	 (space "template <typename... Ts> requires(sizeof...(Ts) > 0 && requires(T*p, Ts... Args) { { construct(&p, Args...) } -> std::same_as<void>; }) [[nodiscard]] constexpr explicit(sizeof...(Ts) == 1) c_resource(Ts &&... Args) noexcept : ptr_{ null }"
		(progn
		  
		  (construct &ptr_
			     "static_cast<Ts &&> (Args)... ")
		  (<< std--cout (string "construct83")
		      std--endl)))
	 ;; WTF! greater than inside a template
	 (space "template <typename... Ts> requires(sizeof...(Ts) > 0 && requires(T*p, Ts... Args) { { construct(&p, Args...) } -> std::same_as<void>; }) [[nodiscard]] constexpr explicit(sizeof...(Ts) == 1) c_resource(Ts &&... Args) noexcept : ptr_{ null }"
		(progn
		  (construct &ptr_
			     "static_cast<Ts &&> (Args)...")
		  (<< std--cout (string "construct93")
		      std--endl)))

	 (space "template <typename... Ts>"
		(requires "std::is_invocable_v<Constructor, T**, Ts... >")
		"[[nodiscard]]"
		constexpr auto
		(emplace "Ts &&... Args")
		noexcept
		(progn
		  (_destruct ptr_)
		  (setf ptr_ null)
		  (<< std--cout (string "emplace")
		      std--endl)
		  (return (construct &ptr_
				     "static_cast<Ts &&>(Args)..."))))
	 (space "[[nodiscard]]"
		constexpr
		(c_resource "c_resource&& other")
		noexcept
		(progn
		  (setf ptr_ other.ptr_
			other.ptr_ null)
		  (<< std--cout (string "copy104")
		      std--endl)))
	 (space constexpr
		c_resource&
		(operator= "c_resource&& rhs")
		noexcept
		(progn
		  (unless (== this &rhs)
		    (_destruct ptr_)
		    (setf ptr_ rhs.ptr_
			  rhs.ptr_ null)
		    (<< std--cout (string "operator=")
			std--endl)
		    )
		  (return *this)))
	 (space constexpr
		void
		(swap "c_resource& other")
		noexcept
		(progn
		  (let ((ptr ptr_))
		    (setf ptr_ other.ptr_
			  other.ptr_ ptr)
		    (<< std--cout (string "swap")
			std--endl)
		    )))
	 (space static constexpr bool
		(setf destructible
		      (or "std::is_invocable_v<Destructor, T*>"
			  "std::is_invocable_v<Destructor, T**>")))

	 (space constexpr (~c_resource) noexcept =delete)
	 (space constexpr (~c_resource) noexcept
		requires destructible
		(progn
		  (_destruct ptr_)
		  (<< std--cout (string "destruct129")
		      std--endl)))
	 (space constexpr void (clear) noexcept
		requires destructible
		(progn
		  (_destruct ptr_)
		  (setf ptr_ null)
		  (<< std--cout (string "clear")
		      std--endl)))
	 (space constexpr c_resource&
		(operator= "std::nullptr_t") noexcept
		requires destructible
		(progn
		  (clear)
		  (<< std--cout (string "operator=137")
		      std--endl)
		  (return *this)))
	 (space "[[nodiscard]]" constexpr explicit operator
		(bool) const noexcept
		(progn
		  (<< std--cout (string "bool")
			std--endl)
		  (return (!= ptr_ null))
		  ))
	 (space "[[nodiscard]]"
		constexpr bool
		(empty)
		const noexcept
		(progn
		  (<< std--cout (string "empty")
			std--endl)
		  (return (== ptr_ null))))
	 (space "[[nodiscard]]"
		constexpr friend bool
		(have "const c_resource& r")
		noexcept
		(progn
		  (<< std--cout (string "have")
			std--endl)
		  (return (!= r.ptr_ null))))

	 (space auto ("operator<=>" "const c_resource&") =delete)
	 
	 (space "[[nodiscard]]"
		bool
		(operator== "const c_resource& rhs")
		const noexcept
		(progn
		  (<< std--cout (string "operator==")
			std--endl)
		  (return (== 0 (std--memcmp ptr_
					     rhs.ptr_
					     (sizeof T))))))
	 (comments "here should be some get and pointer methods whose implementation depends on __cpp_explicit_this_parameter")
	 "public:"
	 (space constexpr void
		(reset "pointer ptr=null")
		noexcept
		(progn
		  (_destruct ptr_)
		  (setf ptr_ ptr)
		   (<< std--cout (string "reset")
		       std--endl)))
	 (space constexpr pointer
		(release)
		noexcept
		(progn
		  (let ((ptr ptr_))
		    (setf ptr_ null)
		   (<< std--cout (string "release")
			std--endl) 
		    (return ptr))
		  ))

	 (space template "<auto* CleanupFunction>"
		struct guard
		(progn
		  "public:"
		  (using cleaner (decltype CleanupFunction))
		  (space constexpr
			 (guard "c_resource& Obj")
			 noexcept
			 ": ptr_{ Obj.ptr_ }"
			 (progn
			   (<< std--cout (string "guard")
			       std--endl)))
		  (space constexpr
			 (~guard )
			 noexcept
			 
			 (progn
			   (unless (== ptr_ null)
			     (CleanupFunction ptr_))
			   (<< std--cout (string "~guard")
			       std--endl)))
		  "private:"
		   "pointer ptr_;"))
	 "private:"
	 (space constexpr static void
		(_destruct "pointer& p")
		noexcept
		requires "std::is_invocable_v<Destructor, T*>"
		(progn
		  (unless (== p null)
		    (<< std--cout (string "_destruct224 T*")
			       std--endl)
		    (destruct p))))
	 (space constexpr static void
		(_destruct "pointer& p")
		noexcept
		requires "std::is_invocable_v<Destructor, T**>"
		(progn
		  (unless (== p null)
		    (<< std--cout (string "_destruct230 T**")
			       std--endl)
		    (destruct &p))))
	 
	 "pointer ptr_ = null;"
	 
	 ))
      )))

  (let ((name `FancyWindow))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> SDL2/SDL.h)
			(using Window ("c_resource<SDL_Window,SDL_CreateWindow,SDL_DestroyWindow>")
			       Renderer ("c_resource<SDL_Renderer,SDL_CreateRenderer,SDL_DestroyRenderer>")
			       Texture ("c_resource<SDL_Texture,SDL_CreateTexture,SDL_DestroyTexture>"))
			(space struct
			       tDimensions
			       (progn
				 "uint16_t Width;"
				 "uint16_t Height;"))
					
			)
     :implementation-preamble `(do0
				
				
				
				
				)
     :code `(do0
	     (setf "static const auto initializedSDL" (SDL_Init SDL_INIT_VIDEO)
		   "static constexpr auto TexttureFormat" SDL_PIXEL_FORMAT_ARGB8888)
	     (defun successful (Code)
				  (declare (type int Code)
					   (values "static constexpr bool"))
				  (return (== 0 Code)))
	     (defun centeredBox (Dimensions
				 &key (Monitor (SDL_GetNumVideoDisplays)) 
				 ;|Monitor=SDL_GetNumVideoDisplays()|
				   )
	       (declare (type tDimensions Dimensions)
			(type int Monitor ;|Monitor=SDL_GetNumVideoDisplays()|
			      )))
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name (Dimensions)
		 (declare
		  (type tDimensions Dimensions)
		  (construct
		   )
		  (explicit)
		  (noexcept)
		  (values :constructor))
		 )
	       
	       "private:"
	       "Window Window_;"
	       "Renderer Renderer_;"
	       "Texture Texture_;"
	       "int Width_, Height_, PixelsPitch_, SourceFormat_;")
	     (defun isAlive ()
	       (declare (values bool)
			(noexcept))
	       "SDL_Event event;"
	       (while (SDL_PollEvent &event)
		 (when (== SDL_QUIT
			   event.type)
		   (return false)))
	       (return true)
	       ))))
  )


  
