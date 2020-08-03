(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria")
  )
(in-package :cl-py-generator)

(setf *features* (union *features* '()))



(progn
  (defparameter *path* "/home/martin/stage/cl-cpp-generator2/example/29_stm32nucleo/")
  (defparameter *code-file* "run_00_uart")
  (defparameter *source* (format nil "~a/source2/~a" *path* *code-file*))
  
  (defparameter *inspection-facts*
    `((10 "")))

  
    
  (let* ((code
	  `(do0
	    (comments "sudo emerge pyserial")
	    
	    #+nil(do0
	     (imports (matplotlib))
			      ;(matplotlib.use (string "Agg"))
			      (imports ((plt matplotlib.pyplot)))
			 (plt.ion))
	    
	    (imports (			;os
					sys
					time
					;docopt
					;pathlib
		      (np numpy)
					serial
					;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
					
					))
	    (sys.path.append (string "/home/martin/src/nanopb/b/"))
	    "import simple_pb2 as pb"
	    

	       (do0 "# %%"
                    (setf con (serial.Serial
                               :port (string "/dev/ttyACM0")
                               :baudrate 115200 ;1000000
                               :bytesize serial.EIGHTBITS
                            :parity serial.PARITY_NONE
                            :stopbits serial.STOPBITS_ONE
                            :timeout .5 ;; seconds
                            :xonxoff False
                            :rtscts False
                            :writeTimeout .05 ;; seconds
                            :dsrdtr False
                            :interCharTimeout .05)))
	       (setf msg (pb.SimpleMessage))
	       (time.sleep 1)
	       (setf d (con.read_all))
	       (print d)
	       #+nil (setf pbr (msg.ParseDelimitedFromString d))
	       (setf pbr (msg.ParseFromString d ;(aref d "1:")
					      ))
	    #-nil
	    (class Uart ()
		   (def __init__ (self        connection
				       &key (debug False) 
				  )
		     (do0 (setf self._con connection)
			  (setf self._debug debug)
			  )
		     
		     )
		   
		   (do0
		    (def _write (self cmd)
		      #+nil (time.sleep .01)
		      #+nil (print (dot (string " Q: {}") (format cmd)))
					;(self._con.reset_input_buffer)
		      (do0
		      #+nil (setf old_reply (self._con.read self._con.in_waiting))
		       #+nil (unless (== "b''" old_reply)
			 (print old_reply)))
		      (self._con.write (dot (string "{}\\n")
					    (format cmd)
					    (encode (string "utf-8")))))
		    
		    
		    (def _read (self)
		      "# read all response lines from uart connections"
		      (try
		       (do0
			#+nil (time.sleep .01)
			(setf line (self._con.read_until))
			
			(setf res (dot line
				       (decode (string ;"utf-8"
						"ISO-8859-1"
						))))
			
			(while self._con.in_waiting
			
					(print res)
			  
			  (setf line (self._con.read_until))
			  #-nil (print (dot (string "AW: {}") (format line)))
			  
			  (setf res (dot line
					 (decode (string ;"utf-8"
						  "ISO-8859-1"
						  )))))
			)
		       ("Exception as e"
			(print (dot (string "warning in _read: {}. discarding the remaining input buffer {}.")
				    (format  e (self._con.read :size self._con.in_waiting))))
			(self._con.reset_input_buffer)
			(return np.nan)))
		      (return res))

		    (def close (self)
		      (self._con.close))
		    )
		   )
	    (setf u (Uart con))))
	 
	 )


    (write-source (format nil "~a/source2/~a" *path* *code-file*) code)
    ))

