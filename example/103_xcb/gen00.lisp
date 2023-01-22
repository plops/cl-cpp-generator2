(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/103_xcb/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (include <xcb/xcb.h>)
     (include<> cstring
		cstdlib
		vector)
     
					;(include <iostream>)

     (defun main ()
       (declare (values int))
       (let ((*conn (xcb_connect nullptr
				 nullptr)))
	 (when (xcb_connection_has_error conn)
	   (return 1))
	 #+nil (xcb_log_enable conn XCB_LOG_ERROR)
	 (let ((*screen (dot (xcb_setup_roots_iterator (xcb_get_setup conn))
			     data))
               (win (xcb_generate_id conn))
	       (mask (or XCB_CW_BACK_PIXEL
			 XCB_CW_EVENT_MASK))
	       (values (std--vector<uint32_t> (curly screen->white_pixel
						(or XCB_EVENT_MASK_EXPOSURE
						    XCB_EVENT_MASK_KEY_PRESS
						    XCB_EVENT_MASK_BUTTON_PRESS)))))
					;(declare (type xcb_window_t win))
           (xcb_create_window conn XCB_COPY_FROM_PARENT
			      win
			      screen->root 0 0
			      600 400
			      2
			      XCB_WINDOW_CLASS_INPUT_OUTPUT
			      screen->root_visual
			      mask
			      (values.data))
	   (let ((gc (xcb_generate_id conn))
		 (font (xcb_generate_id conn))
		 (fontName (string "-*-terminal-medium-*-*-*-14-*-*-*-*-*-iso8859-*"))
		 (fontMask (or XCB_GC_FOREGROUND
			       XCB_GC_BACKGROUND
			       XCB_GC_FONT))
		 (fontValues (std--vector<uint32_t> (curly screen->black_pixel
						      screen->white_pixel
						      font))))
					;(declare (type xcb_gcontext_t gc))
	     (xcb_open_font conn font (strlen fontName) fontName)
	     
	     (xcb_create_gc conn gc win fontMask (fontValues.data))
             (xcb_map_window conn win)
             (xcb_flush conn)))
	 (let ((done true)
	       (helloString (string "Hello")))
	  (while done
		 (let ((*event (xcb_wait_for_event conn)))
		   (unless event
		     break)
		   (case (-> event response_type)
		     #+nil (and (-> event response_type)
		      (bitwise-not 0x80)
		      )
		     (XCB_EXPOSE
		      (xcb_clear_area conn 0 win 0 0 0 0)
		      (xcb_image_text_8 conn (strlen helloString) win
					gc 50 50 helloString)
		      (xcb_flush conn))
		     (XCB_MAPPING_NOTIFY
		      )
		     (XCB_BUTTON_PRESS
		      (let ((ev (reinterpret_cast<xcb_button_press_event_t*> event))
			    (x (-> ev
				   event_x))
			    (y (-> ev
				   event_y)))
			
			(xcb_flush conn)))
		     (XCB_KEY_PRESS
		      (let ( ;(*key (reinterpret_cast<xcb_key_press_event_t*> event))
			    (geom_c (xcb_get_geometry conn win))
			    (*geom (xcb_get_geometry_reply conn
							   geom_c
							   nullptr)))
			(xcb_change_gc conn gc
				       XCB_GC_FOREGROUND &screen->white_pixel)
			(let ((s (string "Hello World")))
			  (declare (type "const char*" s))
			  (xcb_image_text_8 conn
					    (strlen s)
					    win
					    gc
					    geom->x geom->y s))
			(xcb_flush conn)
			(free geom))))
		   #+nil ((lambda ()
			    (declare (capture "&conn"))
			    (while true
				   (let ((*msg (xcb_log_get_message conn)))
				     (unless msg
				       return)
			 	     (<< std--cout
					 (xcb_log_get_level_label msg->level)
					 (string " ")
					 msg->message
					 std--endl)))))
		   (free event)
		   )))
	 (do0
	  (xcb_disconnect conn)
	  (return 0)))))))

;; https://xcb.freedesktop.org/tutorial/

;; 2017 maloney x window
