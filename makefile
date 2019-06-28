.SUFFIXES : .cu .f .c .cpp .o_s .o_d .o
# CUDA path
CUDA_INSTALL_PATH ?= /usr/local/cuda-8.0
# Compilers
NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc 
CC   := gcc
CXX  := g++
FORT := gfortran
LINK := $(CXX)
# Includes
INCLUDES  := -I. -I$(CUDA_INSTALL_PATH)/include
# Common flags
FLAGS := -O2 -DUNIX 
# NVCC flags
NVCCFLAGS := -gencode arch=compute_50,"code=sm_50"
# Libs
LIB := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse -fopenmp
# Object files
OBJS1 = ../SpMV/io.o_s testlusol.o_s ../SpMV/mmio.o auxil.o_s ../SpMV/skit_s.o \
	level.o_s syncfree.o_s lusol.o_s lusolDYNC.o_s lusolDYNR.o_s lusolLEVC.o_s \
	lusolLEVR.o_s lusolCPU.o_s format.o_s level_gpu_new.o_s topo_counter2.o_s

OBJS2 = ../SpMV/io.o_d testlusol.o_d ../SpMV/mmio.o auxil.o_d ../SpMV/skit_d.o \
	level.o_d syncfree.o_d lusol.o_d lusolDYNC.o_d lusolDYNR.o_d lusolLEVC.o_d \
	lusolLEVR.o_d lusolCPU.o_d format.o_d level_gpu_new.o_d topo_counter2.o_d

DEC_OBJ = dec.o_d

# Rules
.f.o:
	$(FORT) $(FLAGS) $(INCLUDES) -o $@ -c $<
.c.o:
	$(CC) $(FLAGS) $(INCLUDES) -o $@ -c $<
.cpp.o_s:
	$(CXX) $(FLAGS) -fopenmp -DDOUBLEPRECISION=0 $(INCLUDES) -o $@ -c $<
.cpp.o_d:
	$(CXX) $(FLAGS) -fopenmp -DDOUBLEPRECISION=1 $(INCLUDES) -o $@ -c $<
.cu.o_s:
	$(NVCC) $(NVCCFLAGS) $(FLAGS) -DDOUBLEPRECISION=0 $(INCLUDES) -o $@ -c $<
.cu.o_d:
	$(NVCC) $(NVCCFLAGS) $(FLAGS) -DDOUBLEPRECISION=1 $(INCLUDES) -o $@ -c $<

default: lusolS.ex lusolD.ex
lusolS.ex: $(OBJS1)
	$(LINK) -o lusolS.ex $(OBJS1) $(LIB)
lusolD.ex: $(OBJS2)
	$(LINK) -o lusolD.ex $(OBJS2) $(LIB)

dec: $(DEC_OBJ)
	$(LINK) -o dec.ex $(DEC_OBJ) $(LIB)

clean:
	find . | egrep "#" | xargs rm -f
	find . | egrep "\~" | xargs rm -f
	rm -f *.o *.o_s *.o_d
	rm -f *.ex

