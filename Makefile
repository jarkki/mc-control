appname := none

CXX := clang++
# Header directories
ARMADILLO_INCLUDE_DIR := /usr/local/Cellar/armadillo/6.400.3_1/include
PROJECT_INCLUDE_DIR := ./include
BOOST_INLUDE_DIR := /usr/local/Cellar/boost/1.60.0_1/include
INCLUDE_DIRS := -I$(ARMADILLO_INCLUDE_DIR)  -I$(PROJECT_INCLUDE_DIR) -I$(BOOST_INLUDE_DIR)
CXXFLAGS := -Wall -g -std=c++11

# Libraries to link and library directories
LDLIBS := -larmadillo
ARMADILLO_LIB_DIR := /usr/local/Cellar/armadillo/6.400.3_1/lib
PROJECT_LIB_DIR := ./src
LDFLAGS := -L$(PROJECT_LIB_DIR) -L$(ARMADILLO_LIB_DIR)

# Sources
libsrcfiles :=
optgrowthfiles := $(libsrcfiles) ./src/examples/optgrowth.cpp
objects  := $(patsubst %.cpp, %.o, $(optgrowthfiles))

CXXFLAGS := $(CXXFLAGS) $(INCLUDE_DIRS) $(LDFLAGS)

all: optgrowth

# $(appname): $(objects)
# 	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -o $(appname) $(objects) $(LDLIBS)

optgrowth: $(objects)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -o optgrowth  $(objects) $(LDLIBS)

depend: .depend

.depend: $(optgrowthfiles)
	rm -f ./.depend
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -MM $^>>./.depend;

clean:
	rm -f $(objects)
	rm -rf optgrowth.dSYM

dist-clean: clean
	rm -f *~ .depend

include .depend
