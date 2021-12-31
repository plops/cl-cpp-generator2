all: generated/materials/bakedColor.filamat
generated/materials/bakedColor.filamat: bakedColor.mat
	/home/martin/src/filament/bin/matc --api all \
     -o generated/materials/bakedColor.filamat \
     bakedColor.mat

clean:
	rm generated/materials/bakedColor.filamat
