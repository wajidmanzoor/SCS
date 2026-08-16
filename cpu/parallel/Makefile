CXX := mpic++
CXXFLAGS ?= -O3 -std=c++14 -Wall -Wextra
TARGET := cpu_hybrid

HEADERS := Graph.h Heuristic.h Intialize.h ListLinearHeap.h SBS.h Timer.h Utility.h helpers.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ main.cpp

clean:
	rm -f $(TARGET)
