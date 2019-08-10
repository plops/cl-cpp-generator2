(asdf:defsystem cl-cpp-generator2
    :version "0"
    :description "Emit C/C++/Cuda Language code"
    :maintainer " <kielhorn.martin@gmail.com>"
    :author " <kielhorn.martin@gmail.com>"
    :licence "GPL"
    :depends-on ("alexandria")
    :serial t
    :components ((:file "package")
		 (:file "c")) )
