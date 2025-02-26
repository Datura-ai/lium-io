#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <getopt.h> // For command-line argument parsing
#include <sys/stat.h>  // For mkdir
#include <unistd.h>  // For gethostname() on Linux
#ifdef _WIN32
#include <windows.h> // For GetComputerName() on Windows
#endif

const size_t MEMORY_TEST_SIZE = 1000 * 1024 * 1024;
const uint32_t VALIDATION_VALUE = 0x76543210;

__device__ double lcgRandDevice(unsigned long long seed) {
    // Sophisticated Linear Congruential Generator (LCG) parameters
    const unsigned long long A = 6364136223846793005ULL;  // Multiplier (64-bit)
    const unsigned long long C = 1442695040888963407ULL;  // Increment (64-bit)
    const unsigned long long M = 9223372036854775807ULL;  // Modulus (2^63 - 1)

    // Update the seed using the LCG formula
    seed = (A * seed + C) % M;

    // Apply a floating-point transformation to scale to [0, 1)
    double rand_val = static_cast<double>(seed) / static_cast<double>(M);

    // Return a value in the range [0, 1)
    return rand_val;
}

__global__ void MatrixMultiply(double *A, double *B, double *C, long n, long k) {
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure within matrix bounds
    if (row < n && col < n) {
        double value = 0.0;
        // Perform the matrix multiplication (dot product of row of A and column of B)
        for (int i = 0; i < k; i++) {
            value += A[row * k + i] * B[i * n + col];
        }

        double index_factor = (row * n + col + value) / (double)(n * n);  // Normalize between 0 and 1
        double index_increase = index_factor * n * 1.5;  // Higher increase for smaller index

        // Store the result in C
        C[row * n + col] = value + index_increase;
    }
}

__global__ void GenerateRandomMatrix(double* A, long n, long k, unsigned long long seed) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int idy = blockIdx.y * blockDim.y + threadIdx.y;
    long idx = blockIdx.y * blockDim.y + threadIdx.x;
    long idy = blockIdx.x * blockDim.x + threadIdx.y;

    // printf("blockDim.x: %d, blockDim.y: %d, blockIdx.x: %d , blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d, Row: %d, Col: %d \n", 
        // blockDim.x, blockDim.y,  blockIdx.x, blockIdx.y, threadIdx.x ,threadIdx.y, idx, idy);
    long divider = 1;

    if (n > 100 || k > 100) {
        divider = 10;
    }

    if (idx < n && idy < k) {
        // Generate more distinct random numbers by using different offsets for each thread
        unsigned long long seed_a = seed + idy * n + idx;  // Unique seed for each thread in matrix A

        double rand_num_a = lcgRandDevice(seed_a) / divider;

        if (idx < n && idy < k) {
            // printf("index: %d , Value: %f, Row: %d, Col: %d \n", idy * n + idx, lcgRandDevice(seed_a), idx, idy);
            A[idy * n + idx] = rand_num_a;  // A is of size n * k
        }
    }
}

std::string getComputerName() {
    char buffer[256];
#ifdef _WIN32
    // Windows-specific code
    DWORD size = sizeof(buffer);
    if (GetComputerNameA(buffer, &size)) {
        return std::string(buffer);
    } else {
        return "Unknown";
    }
#else
    // Linux-specific code
    if (gethostname(buffer, sizeof(buffer)) == 0) {
        return std::string(buffer);
    } else {
        return "Unknown";
    }
#endif
}

std::string getCurrentDateTime() {
    // Get current time
    std::time_t t = std::time(nullptr);  
    std::tm tm = *std::localtime(&t);  // Convert to local time
    
    // Format the date and time as YYYY_MM_DD HH_MM
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y_%m_%d_%H_%M");
    
    return oss.str();
}

void writeToResultFile(long N, long K, long seed, float bandwidth, double *mulMatrix, const std::string& result_path) {
    std::ofstream outFile(result_path);

    if (!outFile) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    outFile << "Dimension N: " << N << ", K: " << K << std::endl;

    outFile << "Matrix:\n";
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            outFile << std::fixed << std::setprecision(2) << mulMatrix[i * N + j] << " ";  // 2 decimal precision
        }
        outFile << "\n";
    }

    // Write the bandwidth to the file
    outFile << "Bandwidth: " << bandwidth << std::endl;

    // Close the file
    outFile.close();
    std::cout << "Validation Results saved to: " << result_path << std::endl;
}

bool isGPUAvailable(int deviceId) {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        std::cerr << "GPU " << deviceId << " not available: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error synchronizing GPU " << deviceId << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

int findAvailableGPU() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1; // Error, return invalid GPU id
    }

    // Iterate over all available GPUs and find one that is available
    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        if (isGPUAvailable(deviceId)) {
            // std::cout << "Using GPU: " << deviceId << std::endl;
            return deviceId;  // Return first available GPU
        }
    }

    std::cerr << "No available GPUs found!" << std::endl;
    return -1;
}

class H100Verifier {
public:
    H100Verifier(long m_dim_n, long m_dim_k) {
        this->m_dim_n = m_dim_n;
        this->m_dim_k = m_dim_k;
        m_MulMatrix = new double[m_dim_n * m_dim_n];
    }

    ~H100Verifier() {
        delete[] m_MulMatrix;
    }

    void testBandWidth() {
        // Performance test
        int deviceId = findAvailableGPU();
        if (deviceId == -1) return;

        cudaSetDevice(deviceId); // Set the chosen GPU
        

        CUdeviceptr d_perfTest;  // Declare d_perfTest here
        cuMemAlloc(&d_perfTest, MEMORY_TEST_SIZE);
        
        // Create events for timing
        CUevent start, stop;  // Declare timing events here
        float elapsedTime = 0;
        cuEventCreate(&start, CU_EVENT_DEFAULT);
        cuEventCreate(&stop, CU_EVENT_DEFAULT);

        // Time memory operations
        cuEventRecord(start, 0);
        cuMemsetD32(d_perfTest, VALIDATION_VALUE, MEMORY_TEST_SIZE / 4);
        cuEventRecord(stop, 0);
        cuEventSynchronize(stop);
        cuEventElapsedTime(&elapsedTime, start, stop);

        // Calculate and verify bandwidth
        m_bandWidth = (MEMORY_TEST_SIZE / (elapsedTime * 0.001)) * (1.0f / 1e9f);
    }

    void printMatrixFromDevice(double* d_A, long n, long k) {
        // Allocate memory on the host to hold the matrix
        double* h_A = new double[n * k];

        // Copy the matrix from device to host
        cudaError_t err = cudaMemcpy(h_A, d_A, n * k * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "printMatrixFromDevice Error: " << cudaGetErrorString(err) << std::endl;
            delete[] h_A;
            return;
        }

        // Print the matrix
        for (long i = 0; i < n; i++) {
            for (long j = 0; j < k; j++) {
                std::cout << std::fixed << std::setprecision(2) << h_A[i * k + j] << " ";  // Print with 2 decimal points
            }
            std::cout << std::endl;
        }

        // Free the host memory
        delete[] h_A;
    }

    void generateMatrix(double* matrix, long n, long k, unsigned long long seed) {
        dim3 threadsPerBlock(32, 32); // 1024 threads per block (32x32 block of threads)

        long temp = n;
        if (n > k) {
            n = k;
            k = temp;
        }

        std::cout << "n:" << n<< std::endl;
        std::cout << "k:" << k << std::endl;

        dim3 numBlocks((k + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
        // numBlocks.x = min(numBlocks.x, 65535); // Ensure numBlocks.x does not exceed the max allowed
        numBlocks.y = min(numBlocks.y, 65535); // Ensure numBlocks.y does not exceed the max allowed
        
        std::cout << "Num Block x:" <<  numBlocks.x << std::endl;
        std::cout << "Num Block y:" <<  numBlocks.y << std::endl;
        
        GenerateRandomMatrix<<<numBlocks, threadsPerBlock>>>(matrix, n, k, seed);
        cudaDeviceSynchronize();
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "generateMatrix CUDA Error in kernel launch: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    int verifyChallenge(unsigned long long seed = 1234, const std::string& result_path = "validate_result.txt") {
        double *d_A, *d_B, *d_C;
        cudaError_t err;
        getMaxMatrixDimensions();
        int deviceId = findAvailableGPU();
        if (deviceId == -1) return -1;
        // deviceId = 1;
        cudaSetDevice(deviceId); // Set the chosen GPU

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "Free memory: " << free_mem / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "Total memory: " << total_mem / (1024 * 1024 * 1024) << " GB" << std::endl;

        // Allocate memory on device
        std::cout << "Generating Matrix A" << std::endl;
        err = cudaMalloc(&d_A, m_dim_n * m_dim_k * sizeof(double));
        if (err != cudaSuccess) return handleCudaError(err);
        std::cout << "Generating Matrix B" << std::endl;
        err = cudaMalloc(&d_B, m_dim_k * m_dim_n * sizeof(double));
        if (err != cudaSuccess) return handleCudaError(err);
        std::cout << "Generating Matrix C" << std::endl;
        err = cudaMalloc(&d_C, m_dim_n * m_dim_n * sizeof(double));
        if (err != cudaSuccess) return handleCudaError(err);
    
        generateMatrix(d_A, m_dim_n, m_dim_k, seed);
        generateMatrix(d_B, m_dim_k, m_dim_n, seed + m_dim_k * m_dim_n);

        #if DEBUG
            // std::cout << "Matrix A:" << std::endl;
            // printMatrixFromDevice(d_A, m_dim_n, m_dim_k);
            // std::cout << "Matrix B:" << std::endl;
            // printMatrixFromDevice(d_B, m_dim_k, m_dim_n);
        #endif
    
        testBandWidth();

        // Check for any errors during kernel launch
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return handleCudaError(err);
        
        // Launch kernel for matrix multiplication C = A * B
        dim3 threadsPerBlock(32, 32);
        dim3 numBlocks((m_dim_n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m_dim_n + threadsPerBlock.y - 1) / threadsPerBlock.y);
        std::cout << "Matrix multiply:" << std::endl;
        MatrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m_dim_n, m_dim_k);
        // Check for errors during kernel launch

        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) return handleCudaError(err);
    
        // Copy result matrix C back to the host
        err = cudaMemcpy(m_MulMatrix, d_C, m_dim_n * m_dim_n * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return handleCudaError(err);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
        writeToResultFile(m_dim_n, m_dim_k, seed, m_bandWidth, m_MulMatrix, result_path);
    
        return 0;
    }

    void getMaxMatrixDimensions() {
        int deviceId = 0; // Default to first GPU
        cudaDeviceProp props;
    
        // Get device properties
        cudaError_t err = cudaGetDeviceProperties(&props, deviceId);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    
        // Get available memory
        size_t freeMemory, totalMemory;

        err = cudaMemGetInfo(&freeMemory, &totalMemory);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get memory info: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    
        std::cout << "Free GPU Memory: " << freeMemory / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Total GPU Memory: " << totalMemory / (1024.0 * 1024.0) << " MB" << std::endl;
        
        const long maxMemory = (long)totalMemory - 1048576000 * 2;  // Reduce total memory by 1 GB to leave some space for overhead
        std::cout << "Available GPU Memory: " << maxMemory / (1024.0 * 1024.0) << " MB" << std::endl;

        // Assuming double precision (8 bytes per element)
        const long elementSize = sizeof(double);
        
        // Calculate maximum matrix dimensions n * m that fit in free memory
        long maxElements = maxMemory / elementSize;  // Max number of elements we can allocate
        
        // Let's assume square matrix for simplicity (n = m)
        long max_m_dim_k = maxElements / (2 * m_dim_n) - m_dim_n;
        // m_dim_k = max_m_dim_k;

        std::cout << "Max matrix dimension (n = m): " << m_dim_n << " x " << max_m_dim_k << std::endl;
    
        // If you want to handle non-square matrices, you can change the logic accordingly.
        // For example, you could use a different approach to split the available memory into n and m.
    }

    double* getMulMatrix() const {
        return m_MulMatrix;
    }

    float getBandWidth() {
        return m_bandWidth;
    }

private:
    long m_dim_n;
    long m_dim_k;
    double* m_MulMatrix;
    unsigned long m_seed;
    float m_bandWidth = 0.0f;

    int handleCudaError(cudaError_t err) const {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
};


class ArgumentParser {
public:
    static void parseArguments(int argc, char* argv[], long& dim_n, long& dim_k, unsigned long long& seed, std::string& result_path) {
        static struct option long_options[] = {
            {"dim_n", required_argument, 0, 'n'},
            {"dim_k", required_argument, 0, 'k'},
            {"seed", required_argument, 0, 's'},
            {"result_path", required_argument, 0, 'r'}, // New argument for result path
            {0, 0, 0, 0}
        };

        int option_index = 0;
        int opt;
        while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
            switch (opt) {
                case 'n':
                    dim_n = std::atoi(optarg);
                    break;
                case 'k':
                    dim_k = std::atoi(optarg);
                    break;
                case 's':
                    seed = std::strtoull(optarg, nullptr, 10);
                    break;
                case 'r':
                    result_path = std::string(optarg);  // Store result path
                    break;
                default:
                    std::cerr << "Usage: --dim_n <value> --dim_k <value> --seed <value> --result_path <path>" << std::endl;
                    exit(EXIT_FAILURE);
            }
        }
    }
};
    
int main(int argc, char* argv[]) {
    long N = 1000, K = 5176864;
    unsigned long long seed = 234;
    std::string result_path = "validate_result.txt";

    ArgumentParser::parseArguments(argc, argv, N, K, seed, result_path);

    std::cout << "Seed: " << seed << std::endl;
    
    H100Verifier verifier(N, K);
    auto start = std::chrono::high_resolution_clock::now();
    int result = verifier.verifyChallenge(seed, result_path);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time on GPU: " << duration.count() << " seconds." << std::endl;

    return result;
}