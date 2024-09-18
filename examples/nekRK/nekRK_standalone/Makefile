CC=gcc
#CFLAGS= -std=c++11 -g -O3 -march=native -mtune=native -fopt-info-vec-missed
#CFLAGS= -std=c++11 -g -Ofast -march=native -mtune=native 
#CFLAGS= -std=c++11 -g -Ofast -march=native -mtune=native -fopt-info-vec-all
#CFLAGS= -std=c++11 -g -Ofast -march=native -mtune=native -fopt-info-vec-missed
CFLAGS= -std=c++17 -O3 -march=native -mtune=native -ffast-math
# External libraries
CFLAGS += -include "math.h"
# Function arguments
CFLAGS += -D__NEKRK_DEVICE__=
CFLAGS += -D__NEKRK_CONST__=const
CFLAGS += -D__NEKRK_INLINE__='static inline'
# F64 
CFLAGS += -Dcfloat=double
CFLAGS += -Ddfloat=double
CFLAGS += -D__NEKRK_EXP__=exp
CFLAGS += -D__NEKRK_LOG10__=log10
CFLAGS += -D__NEKRK_LOG__=log
CFLAGS += -D__NEKRK_POW__=pow
CFLAGS += -DCFLOAT_MAX=1e300
CFLAGS += -DCFLOAT_MIN=1e-300
CFLAGS += -DNEKRK_MIN_CFLOAT=fmin
CFLAGS += -DNEKRK_MAX=fmax
CFLAGS += -include files/rates.inc
CFLAGS += -include files/enthalpy_RT.inc
CFLAGS += -include files/heat_capacity_R.inc
CFLAGS += -include files/conductivity.inc
CFLAGS += -include files/viscosity.inc
CFLAGS += -include files/diffusivity.inc
CFLAGS += -include files/mech.h
# Include nekRK files

CXX=mpic++
CC=$(CXX)

all: nekRK

nekRK: nekRK_sa.cpp nekRK_sa.hpp
	$(CXX) -g $(CFLAGS) $(CF64FLAGS) $^ -o nekRK

.PHONY: clean nekRK nekRK32
clean: 
	rm -f *.o nekRK nekRK32


