NVCC = nvcc
CUDA_HOME ?= /usr/local/cuda
CFLAGS = -I$(CUDA_HOME)/include
LDFLAGS = -L$(CUDA_HOME)/lib64 -lcuda

# Get CUDA version
CUDA_VERSION := $(shell $(NVCC) --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
CUDA_MAJOR := $(shell echo $(CUDA_VERSION) | cut -d'.' -f1)
CUDA_MINOR := $(shell echo $(CUDA_VERSION) | cut -d'.' -f2)
SOURCE_FILE := $(CURDIR)/src/miner_jobs/H100Verifier.cu

# Set architecture based on CUDA version
ifeq ($(CUDA_MAJOR), 11)
    ifeq ($(CUDA_MINOR), 0)
        ARCH = -gencode=arch=compute_80,code=sm_80
    else
        ARCH = -gencode=arch=compute_80,code=sm_80
    endif
else ifeq ($(CUDA_MAJOR), 10)
    ARCH = -gencode=arch=compute_70,code=sm_70
else
    # Default to the full list of architectures if CUDA version is unknown
    ARCH = -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120
endif

# Add version-specific flags
CFLAGS += -DCUDA_VERSION=$(CUDA_MAJOR)$(CUDA_MINOR)0

all: H100Verifier

H100Verifier: $(SOURCE_FILE)
	$(NVCC) $(CFLAGS) $^ $(LDFLAGS) $(ARCH) -o $@ -DDEBUG

# Print CUDA information
cuda-info:
	@echo "CUDA Version: $(CUDA_VERSION)"
	@echo "Using architecture: $(ARCH)"
	@echo "CUDA_VERSION define: $(CUDA_MAJOR)$(CUDA_MINOR)0"

clean:
	rm -f H100Verifier
