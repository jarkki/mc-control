CXX := clang++
CXXFLAGS := -DNDEBUG -O2 -std=c++11
DEBUGFLAGS := -Wall -g -std=c++11

# Header directories
PROJECT_INCLUDE_DIR := ./
ARMADILLO_INCLUDE_DIR := /usr/local/Cellar/armadillo/6.400.3_1/include
BOOST_INLUDE_DIR := /usr/local/Cellar/boost/1.60.0_1/include
INCLUDE_DIRS := -I$(PROJECT_INCLUDE_DIR) -I$(ARMADILLO_INCLUDE_DIR) -I$(BOOST_INLUDE_DIR)

# Libraries to link and library directories
LDLIBS := -larmadillo
ARMADILLO_LIB_DIR := /usr/local/Cellar/armadillo/6.400.3_1/lib
LDFLAGS := -L$(ARMADILLO_LIB_DIR)

# This is header only library
DEPS := mc-control/utils.hpp mc-control/distribution.hpp mc-control/algorithms.hpp mc-control/model.hpp mc-control/plot.hpp

all: optgrowth

optgrowth: $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -o optgrowth examples/optgrowth.cpp $(LDLIBS)

optgrowth_debug: $(DEPS)
	$(CXX) $(DEBUGFLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -o optgrowth examples/optgrowth.cpp $(LDLIBS)

clean_og:
	rm optgrowth
	rm -rf optgrowth.dSYM
