all: vis

vis: *.cpp
	g++ $^ -o $@ `pkg-config gtkmm-3.0 --cflags --libs` 

clean:
	rm vis
