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

  

  (defun define-automaton (name states)
    `(do0
      (class State_FSM (Enum)
	     ,@(loop for e in `(START
				START_CHAR0
				START_CHAR1
				START_CHAR2
				START_CHAR3
				START_CHAR4
				PACKET_LEN_LSB
				PACKET_LEN_MSB
				PAYLOAD
				END_CHAR0
				END_CHAR1
				END_CHAR2
				END_CHAR3
				END_CHAR4
				STOP
				ERROR
				)
		    and i from 0
		  collect
		    `(setf ,e ,i)))
      "#  http://www.findinglisp.com/blog/2004/06/basic-automaton-macro.html"
      (setf state State_FSM.START)
      (def ,(format nil "~a_reset" name) ()
        "global state"
        (setf state State_FSM.START))
      (def ,name (con)
        "# returns tuple with 3 values (val, result, comment). If val==1 call again, if val==0 then fsm is in finish state. If val==-1 then FSM is in error state. Collect the comments if you want them. They can contain multiple lines. The result on the other hand will be exactly one line."
        "global state"
        (setf
              result (string "")
              result_comment (string ""))
        ,@(loop for (new-state code) in states collect
             `(if (== state (dot State_FSM ,new-state))
                  (do0
                   ,@code)))
        (return (tuple 1 result result_comment)))))

    
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
	    "from enum import Enum"
	    ,(define-automaton 'parse_serial_packet
                `((START ((setf current_char (dot (con.read)
                                                    (decode)))
                            #+nil (do0 (print current_char)
                                       (print state))
                            (if (== current_char (string "U"))
                                (do0
                                 (setf result (+ current_char
                                                 (dot (con.read)
                                                      (decode))))
                                 (setf state State_FSM.START_CHAR0)
                                 )
				(do0
				 (setf state State_FSM.ERROR)
				 ))))
                    (FINISH (#+nil (print state)
                             (return (tuple 0 result result_comment))))
                    (ERROR (#+nil (print state)
                                  (raise (Exception (dot (string "error in parse_module_response"))))
                                  ;(return (tuple -1 result result_comment))
                                  ))))
	    
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
	       (setf d0 (con.read (* 10 180)))
	       (setf d d0
		     d1 d0)
	       (setf res (list))
	       (setf starting_point_found False
		     starting_point_found_again False)
	       (setf count 0)
	       (while (and (not starting_point_found_again)
			   ;(< count 10)
			   ;(< 100 (len d))
			   )
		(try
		 (do0
		  (when (< 200 (len d))
		    (setf d (con.read (* 10 180))))
		  ;; we search for the end of one packet and the start of the next
		  (setf pattern (string-b "\\xff\\xff\\xff\\xff\\xff\\x55\\x55\\x55\\x55\\x55"))
		  ;; find the packet boundary
		  (setf start_idx (d.find pattern))
		  ;; throw away the partial packet in the beginning and the UUUUU start sequence of the first complete packet
		  ;; (setf d (aref d (slice (+ start_idx (len pattern)) "")))
		  ;; keep the start sequence
		  (setf pkt_len_lsb (aref d (+ 5 5 0 start_idx))
			pkt_len_msb (aref d (+ 5 5 1 start_idx))
			pkt_len (+ pkt_len_lsb (* 256 pkt_len_msb)))

		  (setf pktfront (aref d (slice start_idx (+ start_idx 5 5 2))))
		  
		  (setf d (aref d (slice (+ 5 5 2 start_idx) "")))
		  (setf count (+ count 1))
		  
		  (setf dpkt (aref d "0:pkt_len"))
		  (setf pktend  (aref d (slice pkt_len (+ pkt_len 5))))
		  ;; parse the protobuf stream
		  (setf pbr (msg.ParseFromString dpkt
					;(aref d "1:")
						 ))
		  (when (and (not starting_point_found)
			     (== msg.phase 3))
		    (setf starting_point_found True))
		  (when (and (not starting_point_found_again)
			     (== msg.phase 3))
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
		  (print (dot (string "count={} msg.phase={}")
			      (format count msg.phase)))
		  ;(print msg)
		  )
		 ("Exception as e"
                  (print (dot (string "exception while processing packet {}: {}")
                              (format count e)))
		  
		  ,(let ((l `(start_idx pktfront pktend dpkt ;d (len d) dpkt (len dpkt) pkt_len_msb pkt_len_lsb
					(len dpkt)
					pkt_len ;msg
					)))
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
	   #+nil  (setf u (Uart con))))
	 
	 )


    (write-source (format nil "~a/source2/~a" *path* *code-file*) code)
    ))

