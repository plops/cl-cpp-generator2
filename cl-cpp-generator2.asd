(asdf:defsystem cl-cpp-generator2
    :version "0"
    :description "Emit C/C++/Cuda Language code"
    :maintainer " <kielhorn.martin@gmail.com>"
    :author " <kielhorn.martin@gmail.com>"
    :licence "MIT"
    :depends-on ("alexandria" "cl-ppcre" "jonathan")
    :serial t
    :components ((:file "package")
		 (:file "c")) )
