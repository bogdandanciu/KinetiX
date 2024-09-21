CC=gcc
#CFLAGS= -std=c++11 -g -Ofast -march=native -mtune=native -fopt-info-vec-all
#CFLAGS= -std=c++11 -g -Ofast -march=native -mtune=native -fopt-info-vec-missed
CFLAGS= -std=c++17 -O3 -march=native -mtune=native -ffast-math
# Function specifiers
CFLAGS += -D__KINETIX_DEVICE__=
CFLAGS += -D__KINETIX_CONST__=const
CFLAGS += -D__KINETIX_INLINE__='static inline'
# FP64 
CFLAGS += -Dcfloat=double
CFLAGS += -Ddfloat=double
CFLAGS += -D__KINETIX_EXP__=exp
CFLAGS += -D__KINETIX_LOG10__=log10
CFLAGS += -D__KINETIX_LOG__=log
CFLAGS += -D__KINETIX_POW__=pow
CFLAGS += -DCFLOAT_MAX=1e300
CFLAGS += -DCFLOAT_MIN=1e-300
CFLAGS += -DKINETIX_MIN_CFLOAT=fmin
CFLAGS += -DKINETIX_MAX=fmax
CFLAGS += -include mechanisms/gri/mech.h
CFLAGS += -include mechanisms/gri/rates.cpp
CFLAGS += -include mechanisms/gri/enthalpy_RT.cpp
CFLAGS += -include mechanisms/gri/heat_capacity_R.cpp
CFLAGS += -include mechanisms/gri/conductivity.cpp
CFLAGS += -include mechanisms/gri/viscosity.cpp
CFLAGS += -include mechanisms/gri/diffusivity.cpp
# Include kinetix files

CXX=mpic++
CC=$(CXX)

all: kinetix_test

kinetix_test: kinetix_test.cpp kinetix_test.hpp
	$(CXX) -g $(CFLAGS) $(CF64FLAGS) $^ -o kinetix_test

.PHONY: clean kinetix
clean: 
	rm -f *.o kinetix_test


