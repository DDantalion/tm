# Compiler and flags
NVCC = nvcc
CXXFLAGS = -O3 -std=c++14

# Source files
SOURCES = prefetch_latency_test.cu nvlink_contention_test.cu host_migration_parallelism_test.cu

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
