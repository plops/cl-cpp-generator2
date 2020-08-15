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
					;START_CHAR4
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
				FINISH
				)
		    and i from 0
		  collect
		    `(setf ,e ,i)))
      "#  http://www.findinglisp.com/blog/2004/06/basic-automaton-macro.html"
      (setf state State_FSM.START)
      (def ,(format nil "~a_reset" name) ()
        "global state"
        (setf state State_FSM.START))
      (def ,name (con &key (accum "{}") (debug False))
        "# returns tuple with 3 values (val, result, comment). If val==1 call again, if val==0 then fsm is in finish state. If val==-1 then FSM is in error state."
        "global state"
        (setf
              result (string "")
              result_comment accum)
        ,@(loop for (new-state code) in states collect
             `(if (== state (dot State_FSM ,new-state))
                  (do0
                   ,@code)))
        (return (tuple 1 result result_comment)))))

    
  (let*
      ((n-channels 60)
       (code
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
                 `(,@(loop for init-state in `(START
					       START_CHAR0
					        START_CHAR1
					START_CHAR2
					START_CHAR3
					;START_CHAR4
					       )
			       and
				 next-state in `(
						 START_CHAR0
						 START_CHAR1
					START_CHAR2
					START_CHAR3
					;START_CHAR4
						 PACKET_LEN_LSB
						 )
			        collect
			  `(,init-state
			    ((setf current_char (con.read)
                                   )
			     
			     
			     ,(if (eq init-state 'START)
				  `(do0
				    (setf (aref result_comment (string "parsed_bytes")) 0)
				    (setf (aref result_comment (string "non_starting_bytes_seen"))
					  (+ 1 
					   (aref result_comment (string "non_starting_bytes_seen")))))
				  `(setf (aref result_comment (string "parsed_bytes"))
					 (+ 1 (aref result_comment (string "parsed_bytes")))))
			     (when debug
			       (print (dot (string ,(format nil "{} current_state=~a next-state=~a char={} non_starting_bytes={}" init-state next-state))
					   (format (aref result_comment (string "parsed_bytes")) current_char
						   (aref result_comment (string "non_starting_bytes_seen"))))))
			     (if (== current_char "b'U'" ;#x55 ; (string "U")
				     )
				 (do0
				  
				  #+nil (setf result (+ current_char
						  (con.read)))
				  (setf state (dot State_FSM ,next-state)))
				 (do0
				  (setf state State_FSM.START ;ERROR
					)
				  )))))

		     (PACKET_LEN_LSB ((setf current_char (con.read))
				      (setf (aref result_comment (string "parsed_bytes"))
					 (+ 1 (aref result_comment (string "parsed_bytes"))))
				      (when debug
					(print (dot (string ,(format nil "{} current_state=PACKET_LEN_LSB char={}" ))
						   (format (aref result_comment (string "parsed_bytes")) current_char))))
				      (setf (aref result_comment (string "packet_len")) (aref  current_char 0))
				      (setf state (dot State_FSM PACKET_LEN_MSB))))
		     (PACKET_LEN_MSB ((setf current_char (con.read))
				      (setf (aref result_comment (string "parsed_bytes"))
					 (+ 1 (aref result_comment (string "parsed_bytes"))))
				      (setf (aref result_comment (string "packet_len"))
					    (+  (aref result_comment (string "packet_len"))
					       (* 256 (aref  current_char 0))))
				      (setf (aref result_comment (string "packet_payload_bytes_read"))
					    0)
				      (setf (aref result_comment (string "payload"))
					    (np.zeros (aref result_comment (string "packet_len"))
						      :dtype np.uint8)
					    )
				      (when debug
					(print (dot (string ,(format nil "{} current_state=PACKET_LEN_MSB char={} packet_len={}" ))
						   (format
						    (aref result_comment (string "parsed_bytes"))
						    current_char
						    (aref result_comment (string "packet_len"))))))
				      (setf state (dot State_FSM PAYLOAD))))
		     (PAYLOAD ((setf current_char (con.read))
			       (when debug
				 (print (dot (string ,(format nil "{} current_state=PAYLOAD char={} packet_payload_bytes_read={}" ))
					    (format (aref result_comment (string "parsed_bytes")) current_char
						    (aref result_comment (string "packet_payload_bytes_read"))))))
			       (setf (aref result_comment (string "parsed_bytes"))
				     (+ 1 (aref result_comment (string "parsed_bytes"))))
			       (setf (aref (aref result_comment (string "payload")) (aref result_comment (string "packet_payload_bytes_read")))
				     (aref current_char 0))
			       (setf (aref result_comment (string "packet_payload_bytes_read"))
				     (+ (aref result_comment (string "packet_payload_bytes_read"))
					1))
			       (if (< (aref result_comment (string "packet_payload_bytes_read"))
				      (aref result_comment (string "packet_len"))
					 )
				   (setf state (dot State_FSM PAYLOAD))
				   (setf state (dot State_FSM END_CHAR0)))))
		     ,@(loop for init-state in `(END_CHAR0
						 END_CHAR1
						 END_CHAR2
						 END_CHAR3
						 END_CHAR4
						 )
			       and
			  next-state in `(END_CHAR1
					  END_CHAR2
					  END_CHAR3
					  END_CHAR4
					  FINISH)
			        collect
			  `(,init-state
			    ((setf current_char (con.read)
                                   )
			     
			     (setf (aref result_comment (string "parsed_bytes"))
					 (+ 1 (aref result_comment (string "parsed_bytes"))))
			     (when debug
			       (print (dot (string ,(format nil "{} current_state=~a next-state=~a char={}" init-state next-state))
					  (format (aref result_comment (string "parsed_bytes")) current_char))))
			     (if (== (aref current_char 0) #xff ; "b'\xff'" ;#x55 ; (string "U")
				     )
				 (do0
				  
				  #+nil (setf result (+ current_char
						  (con.read)))
				  (setf state (dot State_FSM ,next-state)))
				 (do0
				  (setf state State_FSM.ERROR
					)
				  )))))
                     (FINISH (#+nil (print state)
				    (setf (aref result_comment (string "non_starting_bytes_seen")) 0)
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
					;2000000
                               :bytesize serial.EIGHTBITS
                               :parity serial.PARITY_NONE
                               :stopbits serial.STOPBITS_ONE
                               :timeout .5 ;; seconds
                               :xonxoff False
                               :rtscts False
                               :writeTimeout .05 ;; seconds
                               :dsrdtr False
                               :interCharTimeout .05)))

	       (class Listener ()
		      (def __init__ (self connection)
			(setf self._con connection))
		      (def _fsm_read (self)
			(parse_serial_packet_reset)
			(setf res (tuple 1 (string "") (dict ((string "non_starting_bytes_seen") 0))))
			(while (== 1 (aref res 0))
			  (setf res (parse_serial_packet self._con :accum (aref res 2)
							 :debug False))
			  ;(print res)
			  )
			(setf response (aref res 1))
			(return res)))
	       

	       (setf l (Listener con))
	       (setf msgs (list))
	       
	       (for (i (range 320))
		    (try
		     (do0
		      (setf res (l._fsm_read))
		
		 
		      (unless (== 0 (len (aref res 2)))
			(setf msg (pb.SimpleMessage))
			(setf pbr (msg.ParseFromString (aref (aref res 2)
							     (string "payload"))))
			(setf d (dict
				 ,@(loop for e in `(id timestamp phase ,@(loop for i below n-channels collect (format nil "sample~2,'0d" i))) collect
					`((string ,e) (dot msg ,e)))))
			(msgs.append d)
			(print msg)))
		     ("Exception as e"
		      (print e)
		      pass)))
	       (setf df (pd.DataFrame msgs))
	       (do0
		(setf xdat (np.array (list ,@(loop for i below n-channels appending
						  (loop for phase below 80 collect
						       (+ (* 80 i) phase)))))
		      ydat (np.zeros (* n-channels 80))		      
		      )
		,@(loop for phase below 80 collect
		       `(try
			 (do0
			  (setf df1 (aref df (== df.phase ,phase)))
		     	  ,@(loop for i below n-channels collect
			       `(setf (aref ydat (+ (* 80 ,i) ,phase))
				      (aref (aref df1.iloc 0)
					    (string ,(format nil "sample~2,'0d" i))))))
			 ("Exception as e"
			  (print e)
			  pass)))
		)
	       (do0
		(plt.plot xdat ydat)
		(plt.grid))
	       
	       
	       #+nil (do
		"# %%"
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
		 (xrp.imshow (np.log xs.sample)))
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

