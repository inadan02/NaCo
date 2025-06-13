all : contiguous-fa-lang contiguous-fa countpaths mjc-fa contiguous-negative-selection-lang makerep-contiguous-fa makerep-contiguous-fa-lang contiguous-negative-selection

CXXFLAGS = -O3 -std=c++0x
FSTINCLUDE = -I$(HOME)/openfst-1.6.3-install/include
FSTLIB = -L$(HOME)/openfst-1.6.3-install/lib -lfst

# Generic FA targets
%-fa : %-fa.cpp dfa.hpp idfa.hpp refinable-partition.hpp aminoacids.hpp miyazawa-jernigan-matrix.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Optional executable
printlang: printlang.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

countpaths: countpaths.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

contiguous-negative-selection-lang : contiguous-negative-selection-lang.cpp latinalphabet.hpp
	$(CXX) $(CXXFLAGS) $(FSTINCLUDE) $(FSTLIB) -o $@ $<

contiguous-negative-selection : contiguous-negative-selection.cpp aminoacids.hpp
	$(CXX) $(CXXFLAGS) $(FSTINCLUDE) $(FSTLIB) -o $@ $<

clean :
	rm -f contiguous-fa-lang contiguous-fa countpaths mjc-fa pattern-fa contiguous-negative-selection-lang pattern-exact-fa makerep-contiguous-fa makerep-contiguous-fa-lang makerep-pattern-exact-fa contiguous-negative-selection