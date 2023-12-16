CC = nvcc
CFLAG = -std=c++17

SRC = ${wildcard *.cu}
EXE = ${patsubst %.cu,%,$(SRC)}

all: $(EXE)

clean:
	rm -rf bin/*
	rm -rf output/*

%:%.cu
	$(CC) $(CFLAG) -o bin/$@ $<