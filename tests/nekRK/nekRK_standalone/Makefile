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
CF64FLAGS += -Dcfloat=double
CF64FLAGS += -Ddfloat=double
CF64FLAGS += -D__NEKRK_EXP__=exp
CF64FLAGS += -D__NEKRK_LOG10__=log10
CF64FLAGS += -D__NEKRK_LOG__=log
CF64FLAGS += -D__NEKRK_POW__=pow
CF64FLAGS += -DCFLOAT_MAX=1e300
CF64FLAGS += -DCFLOAT_MIN=1e-300
CF64FLAGS += -DNEKRK_MIN_CFLOAT=fmin
CF64FLAGS += -DNEKRK_MAX=fmax
# F32 
CF32FLAGS += -Dcfloat=float
CF32FLAGS += -Ddfloat=double
CF32FLAGS += -D__NEKRK_EXP__=expf
CF32FLAGS += -D__NEKRK_LOG10__=log10f
CF32FLAGS += -D__NEKRK_LOG__=logf
CF32FLAGS += -D__NEKRK_POW__=pow
CF32FLAGS += -DCFLOAT_MAX=1e37f
CF32FLAGS += -DCFLOAT_MIN=1e-37f
CF32FLAGS += -DNEKRK_MIN_CFLOAT=fminf
CF32FLAGS += -DNEKRK_MAX=fmaxf
# Include nekRK files
CF64FLAGS += -include files/rates.inc
CF32FLAGS += -include files/frates.inc
CFLAGS += -include files/fenthalpy_RT.inc
CFLAGS += -include files/fheat_capacity_R.inc
CFLAGS += -include files/fconductivity.inc
CFLAGS += -include files/fviscosity.inc
CFLAGS += -include files/fdiffusivity.inc
CFLAGS += -include files/mech.h

CXX=mpic++
CC=$(CXX)

all: nekRK nekRK32

nekRK: main.c nekRK.h
	$(CXX) -g $(CFLAGS) $(CF64FLAGS) $^ -o nekRK

nekRK32: main.c nekRK.h
	$(CXX) -g -DFP32 $(CFLAGS) $(CF32FLAGS) $^ -o nekRK32

.PHONY: clean nekRK nekRK32
clean: 
	rm -f *.o nekRK nekRK32


