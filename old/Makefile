# Compiler and flags
NVCC = nvcc
CXXFLAGS = -O0 -std=c++14

# Source files
SOURCES = prefetched.cu contention.cu hostm.cu contention4.cu

# Object files (just strip the .cu extension)
BINS = $(SOURCES:.cu=)

# Default target: build everything
all: $(BINS)

# Compile each .cu file into a binary
%: %.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

# Clean up generated binaries
clean:
	rm -f $(BINS)

cache: cache.cu
	nvcc -O0 -std=c++14 -o cache cache.cu