#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/sha.h>
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>
#include <botan/botan.h>
#include <botan/hex.h>
#include <botan/cipher_mode.h>
#include <botan/hmac.h>
#include <botan/base64.h>
#include <botan/rng.h>
#include <botan/auto_rng.h>
#include <botan/hash.h>
#include <chrono>

using json = nlohmann::json;
namespace fs = std::filesystem;

// NVML Constants
const unsigned int NVML_SUCCESS = 0;
const unsigned int NVML_ERROR_UNINITIALIZED = 1;
const unsigned int NVML_ERROR_INVALID_ARGUMENT = 2;
const unsigned int NVML_ERROR_NOT_SUPPORTED = 3;
const unsigned int NVML_ERROR_NO_PERMISSION = 4;
const unsigned int NVML_ERROR_ALREADY_INITIALIZED = 5;
const unsigned int NVML_ERROR_NOT_FOUND = 6;
const unsigned int NVML_ERROR_INSUFFICIENT_SIZE = 7;
const unsigned int NVML_ERROR_INSUFFICIENT_POWER = 8;
const unsigned int NVML_ERROR_DRIVER_NOT_LOADED = 9;
const unsigned int NVML_ERROR_TIMEOUT = 10;
const unsigned int NVML_ERROR_UNKNOWN = 999;
const unsigned int NVML_ERROR_FUNCTION_NOT_FOUND = 13;

// Buffer sizes
const unsigned int NVML_DEVICE_NAME_BUFFER_SIZE = 64;
const unsigned int NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE = 80;
const unsigned int NVML_DEVICE_SERIAL_BUFFER_SIZE = 30;

// Forward declarations
struct nvmlDevice_st;
typedef struct nvmlDevice_st* nvmlDevice_t;
typedef unsigned int nvmlReturn_t;

struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
};

// Global variables
static void* nvmlLib = nullptr;
static std::mutex libLoadLock;
static int nvmlLibRefcount = 0;

// Function pointer types
typedef nvmlReturn_t (*nvmlInit_v2_t)(void);
typedef nvmlReturn_t (*nvmlShutdown_t)(void);
typedef nvmlReturn_t (*nvmlDeviceGetCount_v2_t)(unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetName_t)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t, nvmlMemory_t*);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t*);

// Function pointer variables
static nvmlInit_v2_t nvmlInit_v2_ptr = nullptr;
static nvmlShutdown_t nvmlShutdown_ptr = nullptr;
static nvmlDeviceGetCount_v2_t nvmlDeviceGetCount_v2_ptr = nullptr;
static nvmlDeviceGetHandleByIndex_v2_t nvmlDeviceGetHandleByIndex_v2_ptr = nullptr;
static nvmlDeviceGetName_t nvmlDeviceGetName_ptr = nullptr;
static nvmlDeviceGetMemoryInfo_t nvmlDeviceGetMemoryInfo_ptr = nullptr;
static nvmlDeviceGetUtilizationRates_t nvmlDeviceGetUtilizationRates_ptr = nullptr;

// Function implementations
nvmlReturn_t nvmlInit_v2() {
    if (!nvmlLib) return NVML_ERROR_UNINITIALIZED;
    if (!nvmlInit_v2_ptr) {
        nvmlInit_v2_ptr = (nvmlInit_v2_t)dlsym(nvmlLib, "nvmlInit_v2");
        if (!nvmlInit_v2_ptr) return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return nvmlInit_v2_ptr();
}

nvmlReturn_t nvmlShutdown() {
    if (!nvmlLib) return NVML_ERROR_UNINITIALIZED;
    if (!nvmlShutdown_ptr) {
        nvmlShutdown_ptr = (nvmlShutdown_t)dlsym(nvmlLib, "nvmlShutdown");
        if (!nvmlShutdown_ptr) return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return nvmlShutdown_ptr();
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount) {
    if (!nvmlLib) return NVML_ERROR_UNINITIALIZED;
    if (!nvmlDeviceGetCount_v2_ptr) {
        nvmlDeviceGetCount_v2_ptr = (nvmlDeviceGetCount_v2_t)dlsym(nvmlLib, "nvmlDeviceGetCount_v2");
        if (!nvmlDeviceGetCount_v2_ptr) return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return nvmlDeviceGetCount_v2_ptr(deviceCount);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) {
    if (!nvmlLib) return NVML_ERROR_UNINITIALIZED;
    if (!nvmlDeviceGetHandleByIndex_v2_ptr) {
        nvmlDeviceGetHandleByIndex_v2_ptr = (nvmlDeviceGetHandleByIndex_v2_t)dlsym(nvmlLib, "nvmlDeviceGetHandleByIndex_v2");
        if (!nvmlDeviceGetHandleByIndex_v2_ptr) return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return nvmlDeviceGetHandleByIndex_v2_ptr(index, device);
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
    if (!nvmlLib) return NVML_ERROR_UNINITIALIZED;
    if (!nvmlDeviceGetName_ptr) {
        nvmlDeviceGetName_ptr = (nvmlDeviceGetName_t)dlsym(nvmlLib, "nvmlDeviceGetName");
        if (!nvmlDeviceGetName_ptr) return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return nvmlDeviceGetName_ptr(device, name, length);
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    if (!nvmlLib) return NVML_ERROR_UNINITIALIZED;
    if (!nvmlDeviceGetMemoryInfo_ptr) {
        nvmlDeviceGetMemoryInfo_ptr = (nvmlDeviceGetMemoryInfo_t)dlsym(nvmlLib, "nvmlDeviceGetMemoryInfo");
        if (!nvmlDeviceGetMemoryInfo_ptr) return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return nvmlDeviceGetMemoryInfo_ptr(device, memory);
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization) {
    if (!nvmlLib) return NVML_ERROR_UNINITIALIZED;
    if (!nvmlDeviceGetUtilizationRates_ptr) {
        nvmlDeviceGetUtilizationRates_ptr = (nvmlDeviceGetUtilizationRates_t)dlsym(nvmlLib, "nvmlDeviceGetUtilizationRates");
        if (!nvmlDeviceGetUtilizationRates_ptr) return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return nvmlDeviceGetUtilizationRates_ptr(device, utilization);
}

// Utility functions
std::string runCommand(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    
    return result;
}

std::vector<uint8_t> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    
    return buffer;
}

// NVML initialization
void initNVML(const std::vector<uint8_t>& nvmlLib_content) {
    std::lock_guard<std::mutex> lock(libLoadLock);
    if (nvmlLib) return;

    char tempPath[] = "/tmp/libnvmlXXXXXX";
    int fd = mkstemp(tempPath);
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary file");
    }

    write(fd, nvmlLib_content.data(), nvmlLib_content.size());
    close(fd);

    nvmlLib = dlopen(tempPath, RTLD_LAZY);
    if (!nvmlLib) {
        unlink(tempPath);
        throw std::runtime_error(std::string("Failed to load NVML library: ") + dlerror());
    }

    unlink(tempPath);
    nvmlLibRefcount++;
}

nlohmann::json getGPUInfo() {
    nlohmann::json gpuData;
    gpuData["count"] = 0;
    gpuData["details"] = nlohmann::json::array();

    try {
        if (nvmlInit_v2() != NVML_SUCCESS) {
            throw std::runtime_error("Failed to initialize NVML");
        }

        unsigned int deviceCount = 0;
        if (nvmlDeviceGetCount_v2(&deviceCount) != NVML_SUCCESS) {
            throw std::runtime_error("Failed to get device count");
        }

        gpuData["count"] = deviceCount;

        for (unsigned int i = 0; i < deviceCount; i++) {
            nvmlDevice_t device;
            if (nvmlDeviceGetHandleByIndex_v2(i, &device) != NVML_SUCCESS) {
                continue;
            }

            nlohmann::json gpuDetails;
            
            char name[NVML_DEVICE_NAME_BUFFER_SIZE];
            if (nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE) == NVML_SUCCESS) {
                gpuDetails["name"] = name;
            }

            nvmlMemory_t memory;
            if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
                gpuDetails["memory"] = {
                    {"total", memory.total},
                    {"free", memory.free},
                    {"used", memory.used}
                };
            }

            nvmlUtilization_t utilization;
            if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
                gpuDetails["utilization"] = {
                    {"gpu", utilization.gpu},
                    {"memory", utilization.memory}
                };
            }

            gpuData["details"].push_back(gpuDetails);
        }

        nvmlShutdown();

    } catch (const std::exception& e) {
        gpuData["error"] = e.what();
    }

    return gpuData;
}

nlohmann::json getSystemInfo() {
    nlohmann::json data;
    
    try {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        int processorCount = 0;
        std::string modelName;

        while (std::getline(cpuinfo, line)) {
            if (line.find("processor") == 0) {
                processorCount++;
            }
            else if (line.find("model name") == 0) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    modelName = line.substr(pos + 2);
                }
            }
        }

        data["cpu"] = {
            {"count", processorCount},
            {"model", modelName}
        };

        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            data["memory"] = {
                {"total", si.totalram * si.mem_unit},
                {"free", si.freeram * si.mem_unit},
                {"used", (si.totalram - si.freeram) * si.mem_unit}
            };
        }

        struct statvfs fs;
        if (statvfs("/", &fs) == 0) {
            data["disk"] = {
                {"total", fs.f_blocks * fs.f_frsize},
                {"free", fs.f_bfree * fs.f_frsize},
                {"available", fs.f_bavail * fs.f_frsize}
            };
        }

        data["gpu"] = getGPUInfo();

    } catch (const std::exception& e) {
        data["error"] = e.what();
    }

    return data;
}

std::string fernet_encrypt(const std::string& key, const std::string& plaintext) {
    Botan::AutoSeeded_RNG rng;

    // Generate a random IV
    Botan::secure_vector<uint8_t> iv(16);
    rng.randomize(iv.data(), iv.size());

    // Encrypt the plaintext using AES-128 in CBC mode
    Botan::SymmetricKey aes_key(reinterpret_cast<const uint8_t*>(key.data()), key.size());
    std::unique_ptr<Botan::Cipher_Mode> cipher = Botan::Cipher_Mode::create("AES-128/CBC", Botan::ENCRYPTION);
    cipher->set_key(aes_key);
    cipher->start(iv.data(), iv.size());
    Botan::secure_vector<uint8_t> plaintext_vec(plaintext.begin(), plaintext.end());
    cipher->finish(plaintext_vec);

    // Create an HMAC signature
    auto hash = Botan::HashFunction::create("SHA-256");
    if(!hash) throw std::runtime_error("SHA-256 hash function not found");
    Botan::HMAC hmac(hash.release());
    hmac.set_key(aes_key);
    hmac.update(iv);
    hmac.update(plaintext_vec);
    Botan::secure_vector<uint8_t> signature = hmac.final();

    // Concatenate the IV, ciphertext, and signature
    std::vector<uint8_t> token;
    token.insert(token.end(), iv.begin(), iv.end());
    token.insert(token.end(), plaintext_vec.begin(), plaintext_vec.end());
    token.insert(token.end(), signature.begin(), signature.end());

    // Base64 encode the token
    return Botan::base64_encode(token);
}

std::string fernet_decrypt(const std::string& key, const std::string& token) {
    // Decode the base64 token
    auto decoded = Botan::base64_decode(token);

    // Extract components: IV, ciphertext, and HMAC
    Botan::secure_vector<uint8_t> iv(decoded.begin(), decoded.begin() + 16);
    Botan::secure_vector<uint8_t> ciphertext(decoded.begin() + 16, decoded.end() - 32);
    Botan::secure_vector<uint8_t> hmac_received(decoded.end() - 32, decoded.end());

    // Recreate HMAC to verify
    Botan::SymmetricKey aes_key(reinterpret_cast<const uint8_t*>(key.data()), key.size());
    auto hash = Botan::HashFunction::create("SHA-256");
    Botan::HMAC hmac(hash.release());
    hmac.set_key(aes_key);
    hmac.update(iv);
    hmac.update(ciphertext);
    Botan::secure_vector<uint8_t> hmac_computed = hmac.final();

    // Compare computed HMAC with the received HMAC
    if (hmac_received != hmac_computed) {
        throw std::runtime_error("HMAC verification failed");
    }

    // Decrypt the ciphertext
    std::unique_ptr<Botan::Cipher_Mode> decryptor = Botan::Cipher_Mode::create("AES-128/CBC", Botan::DECRYPTION);
    decryptor->set_key(aes_key);
    decryptor->start(iv);
    decryptor->finish(ciphertext);

    // Convert decrypted message back to string
    return std::string(reinterpret_cast<char*>(ciphertext.data()), ciphertext.size());
}

int main() {
    try {
        std::string nvmlPath = runCommand("find /usr -name 'libnvidia-ml.so.1'");
        if (!nvmlPath.empty()) {
            nvmlPath = nvmlPath.substr(0, nvmlPath.find_last_not_of("\n") + 1);
            
            auto nvmlContent = readBinaryFile(nvmlPath);
            initNVML(nvmlContent);
            
            nlohmann::json systemInfo = getSystemInfo();
            std::cout << systemInfo.dump(4) << std::endl;

            std::string key = "1234567890abcdef"; // Replace with your actual key
            std::string plaintext = "Hello, Botan!";

            std::string encrypted_token = fernet_encrypt(key, plaintext);
            std::cout << "Encrypted token: " << encrypted_token << std::endl;

            std::string decrypted_token = fernet_decrypt(key, encrypted_token);
            std::cout << "Decrypted token: " << decrypted_token << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}