# Compiler choice:
CC    = gcc
MPICC = mpicc
CFLAGS = -O2 -g -ansi -Wno-unused-result
MPICFLAGS = -O2 -g -ansi

MLIBS	=   -lm

EXES = jozov-grid

jozov-grid: jozov-grid.o
	$(CC) $(CFLAGS) -o jozov-grid jozov-grid.o $(MLIBS)

all: $(EXES)
	mv $(EXES) ../bin

install: $(EXES)
	mv $(EXES) ../bin

.c.o:
	$(CC) -c -o $@ $(CFLAGS) $(QINC) $*.c

clean:
	rm -f $(EXES) *.o *~
