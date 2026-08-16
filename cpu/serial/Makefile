CXX := g++
CXXFLAGS ?= -O3 -std=c++14 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?=

TARGET := my_program
SOURCES := main.cpp
HEADERS := Graph.h Heuristic.h Intialize.h ListLinearHeap.h SBS.h Timer.h Utility.h helpers.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES) $(LDLIBS)

clean:
	rm -f $(TARGET)
