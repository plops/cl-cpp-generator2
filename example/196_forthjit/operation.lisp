(in-package :cl-cpp-generator2)

(let* ((class-name `Operation)
       (members0 `((:name kind   :type OperationKind :initform OperationKind--Literal)
		   (:name value :type  int  :initform 0)
		   (:name primitive   :type Primitive :initform Primitive--Add)
		   (:name true_branch :type "std::vector<Operation>")
		   (:name false_branch :type "std::vector<Operation>")))
       (members (loop for e in members0
		      collect
		      (destructuring-bind (&key name type param doc initform) e
			`(:name ,name
			  :type ,type
			  :param ,param
			  :doc ,doc
			  :initform ,initform
			  :member-name ,(intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))
			  :param-name ,(when param
					 (intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))))))))
     
  (write-class
   :dir *full-source-dir*
   :name class-name
   :headers `()
   :header-preamble `(do0 (comments "header") (include "helpers.h") (include<> vector))
   :implementation-preamble `(do0 (comments "implementation"))
   :code `(do0
	   (defclass ,class-name ()
	     "public:"
		
	     (defmethod ,class-name (&key ,@(remove-if
					     #'null
					     (loop for e in members
						   collect
						   (destructuring-bind (&key name type param doc initform param-name member-name) e
						     (when param
						       `(,param-name ,(if initform initform 0)))))))
	       (declare
		,@(remove-if #'null
			     (loop for e in members
				   collect
				   (destructuring-bind (&key name type param doc initform param-name member-name) e
				     (let ((const-p (let* ((s  (format nil "~a" type))
							   (last-char (aref s (- (length s)
										 1))))
						      (not (eq #\* last-char)))))
				       (when param
					 (if (eq name 'callback)
					     `(type "std::function<void(const uint8_t*, const size_t)>"
						    #+nil PacketReceivedCallback ,param-name)
					     `(type ,(if const-p
							 (format nil "const ~a&" type)
							 type)
						    ,param-name))
					 )))))
		(construct
		 ,@(remove-if #'null
			      (loop for e in members
				    collect
				    (destructuring-bind (&key name type param doc initform param-name member-name) e
				      (cond
					(param
					 (if (eq name 'callback)
					     `(,member-name (std--move ,param-name))
					     `(,member-name ,param-name))) 
					(initform
					 `(,member-name ,initform)))))))
					;(explicit)	    
		(values :constructor)))

	     (defmethod ,(format nil "~~~a" class-name) ()
	       (declare
		(values :constructor)))


	     (defmethod literal (value)
	       (declare (type int value)
			(values Operation))
	       (let ((op (space Operation (curly ))))
		 (do0 (setf op.kind OperationKind--Literal)
		      (setf op.value value))
		 (return op)))
		
	     #+nil
	     ,@(remove-if
		#'null
		(loop for e in members
		      appending
		      (destructuring-bind (&key name type param doc initform param-name member-name) e
			(let ((get (cl-change-case:camel-case (format nil "get-~a" name)))
			      (set (cl-change-case:camel-case (format nil "set-~a" name)))
			      (const-p (let* ((s  (format nil "~a" type))
					      (last-char (aref s (- (length s)
								    1))))
					 (not (eq #\* last-char)))))
			  `(,(if doc
				 `(doc ,doc)
				 "")
			    (defmethod ,get ()
			      (declare ,@(if const-p
					     `((const)
					       (values ,(format nil "const ~a&" type)))
					     `((values ,type))))
			      (return ,member-name))
			    (defmethod ,set (,member-name)
			      (declare (type ,type ,member-name))
			      (setf (-> this ,member-name)
				    ,member-name)))))))
	     "private:"
		
	     ,@(remove-if #'null
			  (loop for e in members
				collect
				(destructuring-bind (&key name type param doc initform param-name member-name) e
				  (if initform
				      `(space ,type ,member-name (curly ,initform))
				      `(space ,type ,member-name)))))))
   :format t))
