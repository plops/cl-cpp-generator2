  ;; ---------------------------------------------------------------- logging
  ;; A var is either an expression (the emitted code doubles as its label)
  ;; or (:as "label" expression).  One std::print call, so every value
  ;; appears exactly once in the generated line.  An optional format spec
  ;; (e.g. ":.3f") can be attached with (:as "label" expr "spec").
  (defun lprint (&key (msg "") (vars nil))
    (let ((entries (mapcar (lambda (v)
			     (if (and (consp v) (eq (car v) :as))
				 (list (second v) (third v) (fourth v))
				 (list (emit-c :code v) v nil)))
			   vars)))
      ;; requires C++23
      `(std--println (string ,(format nil "~a~{ ~a='{~a}'~}"
				      msg
				      (loop for e in entries
					    append (list (first e)
							 (or (third e) "")))))
		     ,@(mapcar #'second entries))))
