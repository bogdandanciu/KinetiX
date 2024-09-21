CC = gcc
CXX = mpic++

CFLAGS = -std=c++17 -O3 -march=native -mtune=native -ffast-math
CFLAGS += -include mechanisms/gri/mechanism.H
CFLAGS += -include mechanisms/gri/mechanism.cpp

# Include directories for AMReX headers
AMREX_INCLUDE = -I$(HOME)/amrex/tmp_install_dir/include

# Linker flags for AMReX library
AMREX_LIB = -L$(HOME)/amrex/tmp_install_dir/lib -lamrex

SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
EXECUTABLE = pele_test

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CFLAGS) $(AMREX_INCLUDE) -o $@ $^ $(AMREX_LIB)

%.o: %.cpp
	$(CXX) $(CFLAGS) $(AMREX_LIB) $(AMREX_INCLUDE) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(EXECUTABLE) *.o
 
