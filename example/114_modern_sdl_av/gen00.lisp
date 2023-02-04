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

	 
	 ))
      ))))


  
