CC = gcc
CXX = mpic++

CFLAGS += -std=c++17 -O3 -march=native -mtune=native -ffast-math

SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
EXECUTABLE = pele_test

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CFLAGS) $(LIBS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)


