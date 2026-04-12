(in-package :cl-cpp-generator2)

(defmacro only-write-when-hash-changed (filename code)
  `(let* ((f ,filename)
          (new-code ,code)
          (new-hash (sxhash new-code))
          (old-hash (gethash f *file-hashes*)))
     (unless (and old-hash (equal new-hash old-hash))
       (setf (gethash f *file-hashes*) new-hash)
       (format t "WRITING TO FILE: ~a~%" f)
       (with-open-file (s f :direction :output :if-exists :supersede :if-does-not-exist :create)
         (write-string new-code s))
       (sb-ext:run-program "clang-format" (list "-i" f) :search t))))

(defun set-class-name (name)
  (setf (gethash "current-class" *file-hashes*) name))

(defun find-class-body (node)
  (if (listp node)
      (if (eq (car node) 'defclass)
          (nthcdr 3 node)
          (loop for e in node
                for res = (find-class-body e)
                when res return res))
      nil))

(defun prefix-method (node class-name)
  "Prefix defmethod name with Class::"
  (if (and (listp node) (eq (car node) 'defmethod))
      (let ((name (second node))
            (params (third node))
            (rest (nthcdr 3 node)))
        (if (and (symbolp name) (not (search "::" (format nil "~a" name))))
            `(defmethod ,(intern (format nil "~a::~a" class-name name))
               ,params
               ,@rest)
            node))
      node))

(defun write-class-runtime (dir name headers header-preamble implementation-preamble code)
  (let* ((header-file (format nil "~a/~a.h" dir name))
         (cpp-file (format nil "~a/~a.cpp" dir name))
         (class-bodies (find-class-body code)))
    (set-class-name name)
    (format t "Processing class ~a~%" name)
    ;; Header
    (only-write-when-hash-changed header-file
				  (emit-c :code `(do0 (do0 ,@header-preamble) ,@headers ,code)))
    ;; Implementation
    (only-write-when-hash-changed cpp-file
				  (emit-c :code `(do0 (do0 ,@implementation-preamble)
						      (include ,(file-namestring header-file))
						      ,@(loop for item in class-bodies
							      collect (prefix-method item name)))))))

(defmacro write-class (&key dir name headers header-preamble implementation-preamble code (format t))
  `(write-class-runtime ,dir ,name ',headers ',header-preamble ',implementation-preamble ,code))
