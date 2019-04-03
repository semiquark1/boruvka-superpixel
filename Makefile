
SRC = src
BLD = pybuild

default: all


CXXFLAGS=-std=c++11

tocopy=setup.py boruvka_superpixel_wrap.pxd boruvka_superpixel_wrap.pyx \
       boruvka_superpixel.h boruvka_superpixel.cpp

all: module

builddir:
	mkdir -p $(BLD)

module: builddir
	cp $(addprefix $(SRC)/,$(tocopy)) $(BLD)
	cd $(BLD); python3 setup.py build_ext --inplace

$(BLD)/boruvka_superpixel.o: builddir \
		$(SRC)/boruvka_superpixel.h $(SRC)/boruvka_superpixel.cpp
	$(CXX) $(CXXFLAGS) -c $(SRC)/boruvka_superpixel.cpp -o $@

clean:
	rm -rf $(BLD)

help:
	@echo make '{ module | clean }'

