CC = gcc
CXX = mpic++

CFLAGS += -std=c++17 -O3 -march=native -mtune=native -ffast-math
CFLAGS += $(shell pkg-config --cflags cantera)
LIBS += $(shell pkg-config --libs cantera)

SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
EXECUTABLE = cantera_test

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CFLAGS) $(LIBS) -o $@ $^ -lcantera_shared -lpthread

%.o: %.cpp
	$(CXX) $(CFLAGS) $(LIBS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)


