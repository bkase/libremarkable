CC = arm-linux-gnueabihf-gcc
CXX = arm-linux-gnueabihf-g++
AR = arm-linux-gnueabihf-ar

CCFLAGS = -fPIC -g

all: poc spy.so

libremarkable/%:
	make -C libremarkable

FREETYPE_PATH = ./libfreetype/install
freetype:
	rm -rvf libfreetype || true
	git clone https://github.com/ricardoquesada/libfreetype
	mkdir -p $(FREETYPE_PATH)
	cd libfreetype && CXX=$(CXX) CC=$(CC) ./configure --host=arm-linux-gnueabihf --without-zlib --without-png --enable-static=yes --enable-shared=no             
	cd libfreetype && make -j4  
	cd libfreetype && DESTDIR=$(shell readlink -f libfreetype/install) make install

poc: libremarkable/libremarkable.a
	$(CXX) -c poc.cc -o poc.o -I$(FREETYPE_PATH)/usr/local/include -I$(FREETYPE_PATH)/usr/local/include/freetype2
	$(CXX) poc.o libremarkable/libremarkable.a $(FREETYPE_PATH)/usr/local/lib/libfreetype.a -o poc 

spy.so: libremarkable/libremarkable.a
	$(CC) -c spy.c -o spy.o
	$(CC) -shared spy.o libremarkable/libremarkable.a -ldl -o spy.so

clean:
	rm -rvf *.so || true
	rm -rvf *.o || true
	rm -rvf poc || true
	make -C libremarkable clean
