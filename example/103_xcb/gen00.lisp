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
     (include <cstring>)
     (include <iostream>)

     (defun main ()
       (declare (values int))
       (let ((*conn (xcb_connect nullptr
				 nullptr)))
	 (when (xcb_connection_has_error conn)
	   (return 1))
	 
	 (let ((*screen (dot (xcb_setup_roots_iterator (xcb_get_setup conn))
			     data))
               (win (xcb_generate_id conn)))
	   ;(declare (type xcb_window_t win))
           (xcb_create_window conn XCB_COPY_FROM_PARENT win screen->root 0 0
			      600 400 0 XCB_WINDOW_CLASS_INPUT_OUTPUT
			      screen->root_visual 0 nullptr)
	   (let ((gc (xcb_generate_id conn))
		 )
	     (declare (type xcb_gcontext_t gc))
	     (xcb_create_gc conn gc win XCB_GC_FOREGROUND &screen->black_pixel)
             (xcb_map_window conn win)
             (xcb_flush conn)))
	 (while true
	  (let ((*event (xcb_wait_for_event conn)))
            (unless event
	      break)
	    (case (and (-> event response_type)
		       (bitwise-not 0x80))
	      (XCB_EXPOSE
	       (xcb_clear_area conn 0 win 0 0 0 0))
	      (XCB_KEY_PRESS
	       (let (;(*key (reinterpret_cast<xcb_key_press_event_t*> event))
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
	    (free event)
	    ))
	 (do0
	     (xcb_disconnect conn)
	     (return 0)))))))

