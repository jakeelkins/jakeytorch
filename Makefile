CC=gcc
CXX=g++
NVCC=nvcc

BLASPATH=/opt/asn/apps/blas_gcc610_avx
CUDAPATH=/opt/asn/apps/cuda_11.7.0

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70

TIMERINCPATH=-I$(CUDAPATH)/include -ITimer/include
INCPATH=-Ijakeytorch/include -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(GTESTPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64 -L$(GTESTPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(GTESTPATH)/lib64 -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart -lcublas

.PHONY: clean

all: build/lib/libTimer.so build/lib/libjake_kernels.so build/bin/project_test

# all the mkdir -p commands make directories if they're not already there.
# @ <cmd> hides printing the command itself when make is called

build/lib/libTimer.so: Timer/src/Timer.cpp
	@mkdir -p build/.objects/Timer
 # here, use .os extension to indicate an obj that can be used in a shared library
	$(CXX) $(CXXFLAGS) -c -fPIC -ITimer/include \
		-I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc\
		-o build/.objects/Timer/Timer.os Timer/src/Timer.cpp
	@mkdir -p build/lib
 # then, make a shared library with that file, link against cuda runtime static (for cuda dependencies later on)
 #-L<where the libs are> -l<lib i want>
	$(CXX) -shared -o build/lib/libTimer.so build/.objects/Timer/* \
		-L$(CUDAPATH)/lib64 -lcudart_static -lcublas_static
	@mkdir -p build/include
	@ln -sf ../../Timer/include/Timer.hpp build/include/Timer.hpp

build/lib/libjake_kernels.so: jakeytorch/src/jake_kernels.cu
	@mkdir -p build/.objects/jake_kernels
 # dr. wise's step 1: compile with -dc
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-Ijakeytorch/include -I$(CUDAPATH)/samples/common/inc \
		-dc -o build/.objects/jake_kernels/jake_kernels.o \
		jakeytorch/src/jake_kernels.cu
  # step 2: link object files with relocatable device code with -dlink.
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/jake_kernels/jake_kernels-dlink.o build/.objects/jake_kernels/jake_kernels.o
	mkdir -p build/lib
 # step 3: use gcc to make final shared library (see cuda docs: separate compilation and linking of cuda c++ device code.
	$(CC) -shared -o build/lib/libjake_kernels.so build/.objects/jake_kernels/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart -lcublas
	@mkdir -p build/include
	@ln -sf ../../jakeytorch/include/jake_kernels.h build/include/jake_kernels.h

build/bin/project_test: build/lib/libTimer.so build/lib/libjake_kernels.so \
	jakeytorch/test/src/test.cpp
	@mkdir -p build/bin
 # now, when we build, we can link our own libraries we built.
	$(CXX) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
		-o build/bin/project_test jakeytorch/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -ljake_kernels -lcudart -lblas -lgfortran -lcublas

run: build/bin/project_test
	@rm -f *.nsys-rep project_test.i* project_test.o* core.*
	@echo -ne "class\n1\n\n10gb\n1\nampere\nproject_test\n" | \
		run_gpu .runTests.sh > /dev/null
	@sleep 5
	@tail -f project_test.o*

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f project_test.*
