(in-package :cl-cpp-generator2)

#+nil
(defclass+ ForthVM ()
  "static constexpr auto MAX_STACK =  256;"
  "static constexpr auto MAX_DICT =  64;"
  "static constexpr auto FUEL_LIMIT =  10'000;"
  "std::vector<int> stack;"
  "std::unordered_map<std::string, int> variables;"
  "std::unordered_map<std::string, void(*)()> dictionary;"
  "int fuel = 0;"
  "public:"
  (defmethod push (val)
    (declare (type int val)
	     (values void))
    (when (<= MAX_STACK (stack.size))
      (throw Error--Stack_Error))
    (stack.push_back val))

  (defmethod pop ()
    (declare (values int))
    (when (stack.empty)
      (throw Error--Stack_Error))
    (let ((val (stack.back)))
      (stack.pop_back)
      (return val)))

  (defmethod consume_fuel ()
    (when (< FUEL_LIMIT
	     "++fuel")
      (throw Error--Stack_Error)))

  (defmethod dot ()
    (<< std--cout (pop) (string " ")))

  (defmethod dup ()
    (let ((v (pop)))
      (push v)
      (push v)))

  (defmethod drop ()
    (pop))

  (defmethod swap ()
    (let ((a (pop))
	  (b (pop)))
      (push a)
      (push b))))
