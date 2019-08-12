(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *path* "/home/martin/quicklisp/local-projects/cl-golang-generator/examples/01_gopl_ch1_lissajous")
  (defparameter *code-file* "lissajous")
  (defparameter *source* (format nil "~a/source/~a" *path*  *code-file*))
  (let* ((code
	  `(do0
	    (package main)
	    (import image image/color image/gif io math math/rand os)
	    (let ((palette (curly "[]color.Color" color.White color.Black)))
	      (const witeIndex 0
		     blackIndex 1)
	      (defun main ()
		(lissajous os.Stdout))
	      (defun lissajous (out)
		(declare (type io.Writer out))
		
		(const cycles 5
		       res .001
		       size 100
		       nframes 64
		       delay 8)
		(assign
		 freq (* 3.0 (rand.Float64))
		 anim (curly gif.GIF
			     :LoopCount nframes)
		 phase 0.0)
		(dotimes (i nframes)
		  (assign
		   rect (image.Rect 0 0
				    (+ 1 (* 2 size))
				    (+ 1 (* 2 size)))
		   img (image.NewPaletted rect palette))
		  (for ((:= t 0.0) (< t (* cycles 2 math.Pi)) (incf t res))
		       (assign
			x (math.Sin t)
			y (math.Sin (+ (* t freq)
				       phase)))
		       (img.SetColorIndex
			(+ size (int (+ .5 (* x size))))
			(+ size (int (+ .5 (* y size))))
			blackIndex))
		  (incf phase .1)
		  (setf
		   anim.Delay (append anim.Delay delay)
		   anim.Image (append anim.Image img)))
		(gif.EncodeAll out &anim))))))
    (write-source *source* code)))


;; go build echo.go
