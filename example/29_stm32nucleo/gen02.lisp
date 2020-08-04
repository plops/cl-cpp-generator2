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
	    
	    (do0
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
					(pd pandas)
					(xr xarray)
					(xrp xarray.plot)
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
                               :baudrate ; 500000
					115200 ;1000000
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
	       (setf d0 (con.read (* 30 180)))
	       (setf d d0
		     d1 d0)
	       (setf res (list))
	       (setf starting_point_found False
		     starting_point_found_again False)
	       (setf count 0)
	       (while (not starting_point_found_again)
		(try
		 (do0
		  ;; we search for the end of one packet and the start of the next
		  (setf pattern (string-b "\\xff\\xff\\xff\\xff\\xff\\x55\\x55\\x55\\x55\\x55"))
		  ;; find the packet boundary
		  (setf start_idx (d.find pattern))
		  ;; throw away the partial packet in the beginning and the UUUUU start sequence of the first complete packet
		  ;; (setf d (aref d (slice (+ start_idx (len pattern)) "")))
		  ;; keep the start sequence 
		  (setf d (aref d (slice (+ 5 start_idx) "")))
		  ;; find the next packet boundary
		  (setf end_idx (dot d (find pattern)))
		  (setf diff_idx (- end_idx start_idx))
		  ;; cut out the first complet packet
		  (setf dh1 (aref d "0:end_idx+5"))
		  (setf d1 (aref d "5:end_idx"))
		  ;; parse the protobuf stream
		  (setf pbr (msg.ParseFromString d1
					;(aref d "1:")
						 ))
		  (when (and (not starting_point_found)
			     (== msg.phase 0))
		    (setf starting_point_found True))
		  (when (and (not starting_point_found_again)
			     (== msg.phase 0))
		    (setf starting_point_found_again True))
		  (when (and starting_point_found
			     (not starting_point_found_again))
		   (do0
		    ,@(loop for i below 40 collect
			   `(res.append
			     (dict
			      ((string "sample_nr")
			       ,i)
			      ((string "sample")
			       ,(format nil "msg.sample~2,'0d" i)
			       )
			      ((string "phase")
			       msg.phase)
			      ))
			   )))
		  (setf count (+ count 1))
		  ;(print msg)
		  )
		 ("Exception as e"
                  (print (dot (string "exception while processing packet {}: {}")
                              (format count e)))
		  
		  ,(let ((l `(start_idx end_idx diff_idx d (len d) dh1 (len dh1) d1 (len d1))))
		     `(do0
		       
		       (print (dot (string3 ,(format nil "~{~a={}~^~%~}" l))
				  (format ,@l)))))
                   (setf f (open (string  ,(format nil "~a/source2/~a.py" *path* *code-file*)))
                         content (f.readlines))
                   (f.close)
                   (setf lineno (dot (aref (sys.exc_info) -1)
                                     tb_lineno))
                   (for (l (range (- lineno 3) (+ lineno 2)))
                        (print (dot (string "{} {}")
                                    (format l (aref (aref content l) (slice 0 -1))))))
                   (print (dot (string "Error in line {}: {} '{}'")
                               (format lineno
                                       (dot (type e)
                                            __name__)
                                       e)))
                   
                   pass)
		 #+nil ("Exception as e"
		  (print e)
		  pass)))
	       (setf last_len (msg.ByteSize))
	       

	       (setf df (pd.DataFrame res))
	       (setf dfi (df.set_index (list (string "sample_nr")
					     (string "phase"))))
	       (setf xs (dfi.to_xarray))
	       (xrp.imshow (np.log xs.sample))
	       #+nil (setf data1 (list
			    ,@(loop for i below 40 collect
				   (format nil "msg.sample~2,'0d" i))))
	       #+nil (setf data2 (list
				  ,@(loop for i below 40 collect
				   (format nil "msg2.sample~2,'0d" i))))
	    #+nil
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

