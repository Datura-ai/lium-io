NVCC = nvcc
CUDA_HOME ?= /usr/local/cuda
CFLAGS = -I$(CUDA_HOME)/include
LDFLAGS = -L$(CUDA_HOME)/lib64 -lcuda

# Get CUDA version
CUDA_VERSION := $(shell $(NVCC) --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
CUDA_MAJOR := $(shell echo $(CUDA_VERSION) | cut -d'.' -f1)
CUDA_MINOR := $(shell echo $(CUDA_VERSION) | cut -d'.' -f2)

# Set architecture based on CUDA version
ifeq ($(shell expr $(CUDA_MAJOR) \>= 11), 1)
    ARCH = -arch=sm_86
else
    ARCH = -arch=sm_75
endif

# Add version-specific flags
CFLAGS += -DCUDA_VERSION=$(CUDA_MAJOR)$(CUDA_MINOR)0

all: H100Verifier

H100Verifier: H100Verifier.cu
	$(NVCC) $(CFLAGS) $^ $(LDFLAGS) -o $@ -DDEBUG

# Print CUDA information
cuda-info:
	@echo "CUDA Version: $(CUDA_VERSION)"
	@echo "Using architecture: $(ARCH)"
	@echo "CUDA_VERSION define: $(CUDA_MAJOR)$(CUDA_MINOR)0"

clean:
	rm -f H100Verifier