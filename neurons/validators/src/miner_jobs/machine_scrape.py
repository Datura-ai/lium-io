from ctypes import *
import sys
import os
import json
import re
import shutil
import subprocess
import threading
import psutil
from functools import wraps
import hashlib
from base64 import b64encode
from cryptography.fernet import Fernet
import tempfile


nvmlLib = None
libLoadLock = threading.Lock()
_nvmlLib_refcount = 0

_nvmlReturn_t = c_uint
NVML_SUCCESS = 0
NVML_ERROR_UNINITIALIZED = 1
NVML_ERROR_INVALID_ARGUMENT = 2
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4
NVML_ERROR_ALREADY_INITIALIZED = 5
NVML_ERROR_NOT_FOUND = 6
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_INSUFFICIENT_POWER = 8
NVML_ERROR_DRIVER_NOT_LOADED = 9
NVML_ERROR_TIMEOUT = 10
NVML_ERROR_IRQ_ISSUE = 11
NVML_ERROR_LIBRARY_NOT_FOUND = 12
NVML_ERROR_FUNCTION_NOT_FOUND = 13
NVML_ERROR_CORRUPTED_INFOROM = 14
NVML_ERROR_GPU_IS_LOST = 15
NVML_ERROR_RESET_REQUIRED = 16
NVML_ERROR_OPERATING_SYSTEM = 17
NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18
NVML_ERROR_IN_USE = 19
NVML_ERROR_MEMORY = 20
NVML_ERROR_NO_DATA = 21
NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22
NVML_ERROR_INSUFFICIENT_RESOURCES = 23
NVML_ERROR_FREQ_NOT_SUPPORTED = 24
NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 25
NVML_ERROR_DEPRECATED = 26
NVML_ERROR_NOT_READY = 27
NVML_ERROR_GPU_NOT_FOUND = 28
NVML_ERROR_INVALID_STATE = 29
NVML_ERROR_UNKNOWN = 999

# buffer size
NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE = 16
NVML_DEVICE_UUID_BUFFER_SIZE = 80
NVML_DEVICE_UUID_V2_BUFFER_SIZE = 96
NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE = 80
NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE = 80
NVML_DEVICE_NAME_BUFFER_SIZE = 64
NVML_DEVICE_NAME_V2_BUFFER_SIZE = 96
NVML_DEVICE_SERIAL_BUFFER_SIZE = 30
NVML_DEVICE_PART_NUMBER_BUFFER_SIZE = 80
NVML_DEVICE_GPU_PART_NUMBER_BUFFER_SIZE = 80
NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE = 32
NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE = 32
NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE = 16
NVML_GRID_LICENSE_BUFFER_SIZE = 128
NVML_VGPU_NAME_BUFFER_SIZE = 64
NVML_GRID_LICENSE_FEATURE_MAX_COUNT = 3
NVML_VGPU_METADATA_OPAQUE_DATA_SIZE = sizeof(c_uint) + 256
NVML_VGPU_PGPU_METADATA_OPAQUE_DATA_SIZE = 256
NVML_DEVICE_GPU_FRU_PART_NUMBER_BUFFER_SIZE = 0x14

_nvmlClockType_t = c_uint
NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_CLOCK_VIDEO = 3
NVML_CLOCK_COUNT = 4

NVML_VALUE_NOT_AVAILABLE_ulonglong = c_ulonglong(-1)


class struct_c_nvmlDevice_t(Structure):
    pass  # opaque handle


c_nvmlDevice_t = POINTER(struct_c_nvmlDevice_t)

COMMANDS = {
    "CHECK_SYSBOX_COMPATIBILITY": [
        "docker", "run", "--rm",
        "--runtime=sysbox-runc",
        "--gpus", "all",
        "daturaai/compute-subnet-executor:latest", "nvidia-smi"
    ],
}


class _PrintableStructure(Structure):
    """
    Abstract class that produces nicer __str__ output than ctypes.Structure.
    e.g. instead of:
      >>> print str(obj)
      <class_name object at 0x7fdf82fef9e0>
    this class will print
      class_name(field_name: formatted_value, field_name: formatted_value)

    _fmt_ dictionary of <str _field_ name> -> <str format>
    e.g. class that has _field_ 'hex_value', c_uint could be formatted with
      _fmt_ = {"hex_value" : "%08X"}
    to produce nicer output.
    Default formatting string for all fields can be set with key "<default>" like:
      _fmt_ = {"<default>" : "%d MHz"} # e.g all values are numbers in MHz.
    If not set it's assumed to be just "%s"

    Exact format of returned str from this class is subject to change in the future.
    """
    _fmt_ = {}

    def __str__(self):
        result = []
        for x in self._fields_:
            key = x[0]
            value = getattr(self, key)
            fmt = "%s"
            if key in self._fmt_:
                fmt = self._fmt_[key]
            elif "<default>" in self._fmt_:
                fmt = self._fmt_["<default>"]
            result.append(("%s: " + fmt) % (key, value))
        return self.__class__.__name__ + "(" + ", ".join(result) + ")"

    def __getattribute__(self, name):
        res = super(_PrintableStructure, self).__getattribute__(name)
        # need to convert bytes to unicode for python3 don't need to for python2
        # Python 2 strings are of both str and bytes
        # Python 3 strings are not of type bytes
        # ctypes should convert everything to the correct values otherwise
        if isinstance(res, bytes):
            if isinstance(res, str):
                return res
            return res.decode()
        return res

    def __setattr__(self, name, value):
        if isinstance(value, str):
            # encoding a python2 string returns the same value, since python2 strings are bytes already
            # bytes passed in python3 will be ignored.
            value = value.encode()
        super(_PrintableStructure, self).__setattr__(name, value)


class c_nvmlMemory_t(_PrintableStructure):
    _fields_ = [
        ('c_nvmlMemory_t_total', c_ulonglong),
        ('c_nvmlMemory_t_free', c_ulonglong),
        ('c_nvmlMemory_t_used', c_ulonglong),
    ]
    _fmt_ = {'<default>': "%d B"}


class c_nvmlMemory_v2_t(_PrintableStructure):
    _fields_ = [
        ('c_nvmlMemory_v2_t_version', c_uint),
        ('c_nvmlMemory_v2_t_total', c_ulonglong),
        ('c_nvmlMemory_v2_t_reserved', c_ulonglong),
        ('c_nvmlMemory_v2_t_free', c_ulonglong),
        ('c_nvmlMemory_v2_t_used', c_ulonglong),
    ]
    _fmt_ = {'<default>': "%d B"}


nvmlMemory_v2 = 0x02000028


class c_nvmlUtilization_t(_PrintableStructure):
    _fields_ = [
        ('c_nvmlUtilization_t_gpu', c_uint),
        ('c_nvmlUtilization_t_memory', c_uint),
    ]
    _fmt_ = {'<default>': "%d %%"}


## Error Checking ##
class NVMLError(Exception):
    _valClassMapping = dict()
    # List of currently known error codes
    _errcode_to_string = {
        NVML_ERROR_UNINITIALIZED:       "Uninitialized",
        NVML_ERROR_INVALID_ARGUMENT:    "Invalid Argument",
        NVML_ERROR_NOT_SUPPORTED:       "Not Supported",
        NVML_ERROR_NO_PERMISSION:       "Insufficient Permissions",
        NVML_ERROR_ALREADY_INITIALIZED: "Already Initialized",
        NVML_ERROR_NOT_FOUND:           "Not Found",
        NVML_ERROR_INSUFFICIENT_SIZE:   "Insufficient Size",
        NVML_ERROR_INSUFFICIENT_POWER:  "Insufficient External Power",
        NVML_ERROR_DRIVER_NOT_LOADED:   "Driver Not Loaded",
        NVML_ERROR_TIMEOUT:             "Timeout",
        NVML_ERROR_IRQ_ISSUE:           "Interrupt Request Issue",
        NVML_ERROR_LIBRARY_NOT_FOUND:   "NVML Shared Library Not Found",
        NVML_ERROR_FUNCTION_NOT_FOUND:  "Function Not Found",
        NVML_ERROR_CORRUPTED_INFOROM:   "Corrupted infoROM",
        NVML_ERROR_GPU_IS_LOST:         "GPU is lost",
        NVML_ERROR_RESET_REQUIRED:      "GPU requires restart",
        NVML_ERROR_OPERATING_SYSTEM:    "The operating system has blocked the request.",
        NVML_ERROR_LIB_RM_VERSION_MISMATCH: "RM has detected an NVML/RM version mismatch.",
        NVML_ERROR_MEMORY:              "Insufficient Memory",
        NVML_ERROR_UNKNOWN:             "Unknown Error",
    }

    def __new__(typ, value):
        '''
        Maps value to a proper subclass of NVMLError.
        See _extractNVMLErrorsAsClasses function for more details
        '''
        if typ == NVMLError:
            typ = NVMLError._valClassMapping.get(value, typ)
        obj = Exception.__new__(typ)
        obj.value = value
        return obj

    def __str__(self):
        try:
            if self.value not in NVMLError._errcode_to_string:
                NVMLError._errcode_to_string[self.value] = str(nvmlErrorString(self.value))
            return NVMLError._errcode_to_string[self.value]
        except NVMLError:
            return "NVML Error with code %d" % self.value

    def __eq__(self, other):
        return self.value == other.value


class c_nvmlProcessInfo_v2_t(_PrintableStructure):
    _fields_ = [
        ('c_nvmlProcessInfo_v2_t_pid', c_uint),
        ('c_nvmlProcessInfo_v2_t_usedGpuMemory', c_ulonglong),
        ('c_nvmlProcessInfo_v2_t_gpuInstanceId', c_uint),
        ('c_nvmlProcessInfo_v2_t_computeInstanceId', c_uint),
    ]
    _fmt_ = {'_fmt_usedGpuMemory': "%d B"}


c_nvmlProcessInfo_v3_t = c_nvmlProcessInfo_v2_t

c_nvmlProcessInfo_t = c_nvmlProcessInfo_v3_t


def convertStrBytes(func):
    '''
    In python 3, strings are unicode instead of bytes, and need to be converted for ctypes
    Args from caller: (1, 'string', <__main__.c_nvmlDevice_t at 0xFFFFFFFF>)
    Args passed to function: (1, b'string', <__main__.c_nvmlDevice_t at 0xFFFFFFFF)>
    ----
    Returned from function: b'returned string'
    Returned to caller: 'returned string'
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        # encoding a str returns bytes in python 2 and 3
        args = [arg.encode() if isinstance(arg, str) else arg for arg in args]
        res = func(*args, **kwargs)
        # In python 2, str and bytes are the same
        # In python 3, str is unicode and should be decoded.
        # Ctypes handles most conversions, this only effects c_char and char arrays.
        if isinstance(res, bytes):
            if isinstance(res, str):
                return res
            return res.decode()
        return res

    if sys.version_info >= (3,):
        return wrapper
    return func


@convertStrBytes
def nvmlErrorString(result):
    fn = _nvmlGetFunctionPointer("nvmlErrorString")
    fn.restype = c_char_p  # otherwise return is an int
    ret = fn(result)
    return ret


def _nvmlCheckReturn(ret):
    if (ret != NVML_SUCCESS):
        raise NVMLError(ret)
    return ret


_nvmlGetFunctionPointer_cache = dict()  # function pointers are cached to prevent unnecessary libLoadLock locking


def _nvmlGetFunctionPointer(name):
    global nvmlLib

    if name in _nvmlGetFunctionPointer_cache:
        return _nvmlGetFunctionPointer_cache[name]

    libLoadLock.acquire()
    try:
        # ensure library was loaded
        if (nvmlLib == None):
            raise NVMLError(NVML_ERROR_UNINITIALIZED)
        try:
            _nvmlGetFunctionPointer_cache[name] = getattr(nvmlLib, name)
            return _nvmlGetFunctionPointer_cache[name]
        except AttributeError:
            raise NVMLError(NVML_ERROR_FUNCTION_NOT_FOUND)
    finally:
        # lock is always freed
        libLoadLock.release()


def nvmlInitWithFlags(flags, nvmlLib_content: bytes):
    _LoadNvmlLibrary(nvmlLib_content)

    #
    # Initialize the library
    #
    fn = _nvmlGetFunctionPointer("nvmlInitWithFlags")
    ret = fn(flags)
    _nvmlCheckReturn(ret)

    # Atomically update refcount
    global _nvmlLib_refcount
    libLoadLock.acquire()
    _nvmlLib_refcount += 1
    libLoadLock.release()
    return None


def nvmlInit(nvmlLib_content: bytes):
    nvmlInitWithFlags(0, nvmlLib_content)
    return None


def _LoadNvmlLibrary(nvmlLib_content: bytes):
    '''
    Load the library if it isn't loaded already
    '''
    global nvmlLib

    if (nvmlLib == None):
        # lock to ensure only one caller loads the library
        libLoadLock.acquire()

        try:
            # ensure the library still isn't loaded
            if (nvmlLib == None):
                try:
                    if (sys.platform[:3] == "win"):
                        # cdecl calling convention
                        try:
                            # Check for nvml.dll in System32 first for DCH drivers
                            nvmlLib = CDLL(os.path.join(os.getenv("WINDIR", "C:/Windows"), "System32/nvml.dll"))
                        except OSError as ose:
                            # If nvml.dll is not found in System32, it should be in ProgramFiles
                            # load nvml.dll from %ProgramFiles%/NVIDIA Corporation/NVSMI/nvml.dll
                            nvmlLib = CDLL(os.path.join(os.getenv("ProgramFiles", "C:/Program Files"), "NVIDIA Corporation/NVSMI/nvml.dll"))
                    else:
                        # assume linux
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_file.write(nvmlLib_content)
                            temp_file_path = temp_file.name

                        try:
                            nvmlLib = CDLL(temp_file_path)
                        finally:
                            os.remove(temp_file_path)
                except OSError as ose:
                    _nvmlCheckReturn(NVML_ERROR_LIBRARY_NOT_FOUND)
                if (nvmlLib == None):
                    _nvmlCheckReturn(NVML_ERROR_LIBRARY_NOT_FOUND)
        finally:
            # lock is always freed
            libLoadLock.release()


def nvmlDeviceGetCount():
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetCount_v2")
    ret = fn(byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value


@convertStrBytes
def nvmlSystemGetDriverVersion():
    c_version = create_string_buffer(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("nvmlSystemGetDriverVersion")
    ret = fn(c_version, c_uint(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value


@convertStrBytes
def nvmlDeviceGetUUID(handle):
    c_uuid = create_string_buffer(NVML_DEVICE_UUID_V2_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetUUID")
    ret = fn(handle, c_uuid, c_uint(NVML_DEVICE_UUID_V2_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_uuid.value


def nvmlSystemGetCudaDriverVersion():
    c_cuda_version = c_int()
    fn = _nvmlGetFunctionPointer("nvmlSystemGetCudaDriverVersion")
    ret = fn(byref(c_cuda_version))
    _nvmlCheckReturn(ret)
    return c_cuda_version.value


def nvmlShutdown():
    #
    # Leave the library loaded, but shutdown the interface
    #
    fn = _nvmlGetFunctionPointer("nvmlShutdown")
    ret = fn()
    _nvmlCheckReturn(ret)

    # Atomically update refcount
    global _nvmlLib_refcount
    libLoadLock.acquire()
    if (0 < _nvmlLib_refcount):
        _nvmlLib_refcount -= 1
    libLoadLock.release()
    return None


def nvmlDeviceGetHandleByIndex(index):
    c_index = c_uint(index)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetHandleByIndex_v2")
    ret = fn(c_index, byref(device))
    _nvmlCheckReturn(ret)
    return device


def nvmlDeviceGetCudaComputeCapability(handle):
    c_major = c_int()
    c_minor = c_int()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetCudaComputeCapability")
    ret = fn(handle, byref(c_major), byref(c_minor))
    _nvmlCheckReturn(ret)
    return (c_major.value, c_minor.value)


@convertStrBytes
def nvmlDeviceGetName(handle):
    c_name = create_string_buffer(NVML_DEVICE_NAME_V2_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetName")
    ret = fn(handle, c_name, c_uint(NVML_DEVICE_NAME_V2_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_name.value


def nvmlDeviceGetMemoryInfo(handle, version=None):
    if not version:
        c_memory = c_nvmlMemory_t()
        fn = _nvmlGetFunctionPointer("nvmlDeviceGetMemoryInfo")
    else:
        c_memory = c_nvmlMemory_v2_t()
        c_memory.c_nvmlMemory_v2_t_version = version
        fn = _nvmlGetFunctionPointer("nvmlDeviceGetMemoryInfo_v2")
    ret = fn(handle, byref(c_memory))
    _nvmlCheckReturn(ret)
    return c_memory


def nvmlDeviceGetPowerManagementLimit(handle):
    c_limit = c_uint()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetPowerManagementLimit")
    ret = fn(handle, byref(c_limit))
    _nvmlCheckReturn(ret)
    return c_limit.value


def nvmlDeviceGetClockInfo(handle, type_clock):
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetClockInfo")
    ret = fn(handle, _nvmlClockType_t(type_clock), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value


def nvmlDeviceGetCurrPcieLinkWidth(handle):
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetCurrPcieLinkWidth")
    width = c_uint()
    ret = fn(handle, byref(width))
    _nvmlCheckReturn(ret)
    return width.value


def nvmlDeviceGetPcieSpeed(device):
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetPcieSpeed")
    ret = fn(device, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value


def nvmlDeviceGetDefaultApplicationsClock(handle, type_clock):
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetDefaultApplicationsClock")
    ret = fn(handle, _nvmlClockType_t(type_clock), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value


def nvmlDeviceGetSupportedMemoryClocks(handle):
    # first call to get the size
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetSupportedMemoryClocks")
    ret = fn(handle, byref(c_count), None)

    if (ret == NVML_SUCCESS):
        # special case, no clocks
        return []
    elif (ret == NVML_ERROR_INSUFFICIENT_SIZE):
        # typical case
        clocks_array = c_uint * c_count.value
        c_clocks = clocks_array()

        # make the call again
        ret = fn(handle, byref(c_count), c_clocks)
        _nvmlCheckReturn(ret)

        procs = []
        for i in range(c_count.value):
            procs.append(c_clocks[i])

        return procs
    else:
        # error case
        raise NVMLError(ret)


def nvmlDeviceGetUtilizationRates(handle):
    c_util = c_nvmlUtilization_t()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetUtilizationRates")
    ret = fn(handle, byref(c_util))
    _nvmlCheckReturn(ret)
    return c_util


class nvmlFriendlyObject(object):
    def __init__(self, dictionary):
        for x in dictionary:
            setattr(self, x, dictionary[x])

    def __str__(self):
        return self.__dict__.__str__()


def nvmlStructToFriendlyObject(struct):
    d = {}
    for x in struct._fields_:
        key = x[0]
        value = getattr(struct, key)
        # only need to convert from bytes if bytes, no need to check python version.
        d[key] = value.decode() if isinstance(value, bytes) else value
    obj = nvmlFriendlyObject(d)
    return obj


def nvmlDeviceGetComputeRunningProcesses_v2(handle):
    # first call to get the size
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetComputeRunningProcesses_v2")
    ret = fn(handle, byref(c_count), None)
    if (ret == NVML_SUCCESS):
        # special case, no running processes
        return []
    elif (ret == NVML_ERROR_INSUFFICIENT_SIZE):
        # typical case
        # oversize the array incase more processes are created
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_v2_t * c_count.value
        c_procs = proc_array()
        # make the call again
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)
        procs = []
        for i in range(c_count.value):
            # use an alternative struct for this object
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if obj.c_nvmlProcessInfo_v2_t_usedGpuMemory == NVML_VALUE_NOT_AVAILABLE_ulonglong.value:
                # special case for WDDM on Windows, see comment above
                obj.c_nvmlProcessInfo_v2_t_usedGpuMemory = None
            procs.append(obj)
        return procs
    else:
        # error case
        raise NVMLError(ret)


def run_cmd(cmd):
    proc = subprocess.run(cmd, shell=True, capture_output=True, check=False, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_cmd error {cmd=!r} {proc.returncode=} {proc.stdout=!r} {proc.stderr=!r}"
        )
    return proc.stdout


def get_network_speed():
    """Get upload and download speed of the machine."""
    data = {"upload_speed": None, "download_speed": None}
    try:
        speedtest_cmd = run_cmd("speedtest-cli --json")
        speedtest_data = json.loads(speedtest_cmd)
        data["upload_speed"] = speedtest_data["upload"] / 1_000_000  # Convert to Mbps
        data["download_speed"] = speedtest_data["download"] / 1_000_000  # Convert to Mbps
    except Exception as exc:
        data["network_speed_error"] = repr(exc)
    return data

def speedcheck_output():
    data = {"upload_speed": None, "download_speed": None}
    try:
        speedtest_cmd = run_cmd("/root/app/.venv/bin/speedcheck run --type ookla")
        json_start = speedtest_cmd.find('{')
        json_str = speedtest_cmd[json_start:]
        speedtest_data = json.loads(json_str)
        data["download_speed"] = float(speedtest_data["Download Speed"].split()[0]) #extract the number
        data["upload_speed"] = float(speedtest_data["Upload Speed"].split()[0]) #extract the number
    except Exception as exc:
        data["network_speed_error"] = repr(exc)
    return data

def netmeasure_output():
    data = {"upload_speed": None, "download_speed": None}
    try:
        speedtest_cmd = run_cmd(f"/root/app/.venv/bin/netmeasure speedtest_dotnet")
        download_match = re.search(r'Download Rate: ([\d.]+) bit/s', speedtest_cmd)
        upload_match = re.search(r'Upload Rate: ([\d.]+) bit/s', speedtest_cmd)

        if download_match and upload_match:
            download_speed = float(download_match.group(1))
            upload_speed = float(upload_match.group(1))
            
            # Convert to Mbps
            data["download_speed"] = download_speed / 1_000_000 # Convert to Mbps 
            data["upload_speed"] = upload_speed / 1_000_000 # Convert to Mbps
    except Exception as exc:
        data["network_speed_error"] = repr(exc)
    return data

def benchmark_network_speed():
    """Benchmark network speed using different methods"""
    data = get_network_speed()
    if data.get("download_speed") or data.get("upload_speed"):
        return data
    
    data = speedcheck_output()
    if data.get("download_speed") or data.get("upload_speed"):
        return data
    
    data = netmeasure_output()
    if data.get("download_speed") or data.get("upload_speed"):
        return data
    
    return {"upload_speed": None, "download_speed": None}

def get_docker_info(content: bytes):
    data = {
        "docker_version": "",
        "docker_container_id": "",
        "docker_containers": []
    }

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(content)
        docker_path = temp_file.name

    try:
        run_cmd(f'chmod +x {docker_path}')

        result = run_cmd(f'{docker_path} version --format "{{{{.Client.Version}}}}"')
        data["docker_version"] = result.strip()

        result = run_cmd(f'{docker_path} ps --no-trunc --format "{{{{.ID}}}}"')
        container_ids = result.strip().split('\n')

        containers = []

        for container_id in container_ids:
            # Get the image ID of the container
            result = run_cmd(f'{docker_path} inspect --format "{{{{.Image}}}}" {container_id}')
            image_id = result.strip()

            # Get the image details
            result = run_cmd(f'{docker_path}  inspect --format "{{{{json .RepoDigests}}}}" {image_id}')
            repo_digests = json.loads(result.strip())

            # Get the container name
            result = run_cmd(f'{docker_path} inspect --format "{{{{.Name}}}}" {container_id}')
            container_name = result.strip().lstrip('/')

            digest = None
            if repo_digests:
                digest = repo_digests[0].split('@')[1]
                if repo_digests[0].split('@')[0] == 'daturaai/compute-subnet-executor':
                    data["docker_container_id"] = container_id

            if digest:
                containers.append({'each_container_id': container_id, 'each_digest': digest, "each_name": container_name})
            else:
                containers.append({'each_container_id': container_id, 'each_digest': '', "each_name": container_name})

        data["docker_containers"] = containers

    finally:
        os.remove(docker_path)

    return data


def get_md5_checksum_from_path(file_path):
    md5_hash = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def get_md5_checksum_from_file_content(file_content: bytes):
    md5_hash = hashlib.md5()
    md5_hash.update(file_content)
    return md5_hash.hexdigest()


def get_sha256_checksum_from_file_content(file_content: bytes):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_content)
    return sha256_hash.hexdigest()


def get_libnvidia_ml_path():
    try:
        original_path = run_cmd("find /usr -name 'libnvidia-ml.so.1'").strip()
        return original_path.split('\n')[-1]
    except:
        return ''


def get_file_content(path: str):
    with open(path, 'rb') as f:
        content = f.read()

    return content


def get_gpu_processes(pids: set, containers: list[dict]):
    if not pids:
        return []

    processes = []
    for pid in pids:
        try:
            cmd = f'cat /proc/{pid}/cgroup'
            info = run_cmd(cmd).strip()

            # Find the container name by checking if the container ID is in the info
            container_name = None
            # if info == "0::/":
            #     container_name = "executor"
            # else:
            #     for container in containers:
            #         if container['id'] in info:
            #             container_name = container['name']
            #             break
            for container in containers:
                if container['each_container_id'] in info:
                    container_name = container['each_name']
                    break

            processes.append({
                "processes_pid": pid,
                "processes_info": info,
                "processes_container_name": container_name
            })
        except:
            pass

    return processes


def check_sysbox_gpu_compatibility() -> tuple[bool, str]:
    """
    Checks if the system supports running Docker containers with the sysbox-runc runtime
    and NVIDIA GPU access (--gpus all).

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating compatibility and a message.
    """
    test_command = COMMANDS["CHECK_SYSBOX_COMPATIBILITY"]

    try:
        result = subprocess.run(
            test_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return True, "Sysbox runtime supports GPU access."
        else:
            return False, "Sysbox runtime does not support GPU access."

    except subprocess.TimeoutExpired:
        return False, "Test command timed out."

    except FileNotFoundError:
        return False, "Docker is not installed or not found in PATH."

    except Exception as e:
        return False, f"An unexpected error occurred: {e}"


def get_machine_specs():
    """Get Specs of miner machine."""
    data = {}

    if os.environ.get('LD_PRELOAD'):
        return data

    data["data_gpu"] = {"gpu_count": 0, "gpu_details": []}
    gpu_process_ids = set()

    libnvidia_path = get_libnvidia_ml_path()
    if not libnvidia_path:
        return data

    nvmlLib_content = get_file_content(libnvidia_path)
    docker_content = get_file_content("/usr/bin/docker")
    nvidia_smi_content = get_file_content('/usr/bin/nvidia-smi')

    try:
        nvmlInit(nvmlLib_content)

        device_count = nvmlDeviceGetCount()

        data["data_gpu"] = {
            "gpu_count": device_count,
            "gpu_driver": nvmlSystemGetDriverVersion(),
            "gpu_cuda_driver": nvmlSystemGetCudaDriverVersion(),
            "gpu_details": []
        }

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            # graphic_clock = nvmlDeviceGetDefaultApplicationsClock(handle, NVML_CLOCK_GRAPHICS)
            # memory_clock = nvmlDeviceGetDefaultApplicationsClock(handle, NVML_CLOCK_MEM)
            # memory_clocks = nvmlDeviceGetSupportedMemoryClocks(handle)
            # print(graphic_clock)
            # print(memory_clock)
            # print(memory_clocks)

            cuda_compute_capability = nvmlDeviceGetCudaComputeCapability(handle)
            major = cuda_compute_capability[0]
            minor = cuda_compute_capability[1]

            # Get GPU utilization rates
            utilization = nvmlDeviceGetUtilizationRates(handle)

            data["data_gpu"]["gpu_details"].append(
                {
                    "gpu.name": nvmlDeviceGetName(handle),
                    "gpu.uuid": nvmlDeviceGetUUID(handle),
                    "gpu.capacity": nvmlDeviceGetMemoryInfo(handle).c_nvmlMemory_t_total / (1024 ** 2),  # in MB
                    "gpu.cuda": f"{major}.{minor}",
                    "gpu.power_limit": nvmlDeviceGetPowerManagementLimit(handle) / 1000,
                    "gpu.graphics_speed": nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS),
                    "gpu.memory_speed": nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM),
                    "gpu.pcie": nvmlDeviceGetCurrPcieLinkWidth(handle),
                    "gpu.speed_pcie": nvmlDeviceGetPcieSpeed(handle),
                    "gpu.utilization": utilization.c_nvmlUtilization_t_gpu,
                    "gpu.memory_utilization": utilization.c_nvmlUtilization_t_memory,
                }
            )

            processes = nvmlDeviceGetComputeRunningProcesses_v2(handle)

            # Collect process IDs
            for proc in processes:
                gpu_process_ids.add(proc.c_nvmlProcessInfo_v2_t_pid)

        nvmlShutdown()
    except Exception as exc:
        # print(f'Error getting os specs: {exc}', flush=True)
        data["gpu_scrape_error"] = repr(exc)

        # Scrape the NVIDIA Container Runtime config
        nvidia_cfg_cmd = 'cat /etc/nvidia-container-runtime/config.toml'
        try:
            data["data_nvidia_cfg"] = run_cmd(nvidia_cfg_cmd)
        except Exception as exc:
            data["nvidia_cfg_scrape_error"] = repr(exc)

        # Scrape the Docker Daemon config
        docker_cfg_cmd = 'cat /etc/docker/daemon.json'
        try:
            data["data_docker_cfg"] = run_cmd(docker_cfg_cmd)
        except Exception as exc:
            data["data_docker_cfg_scrape_error"] = repr(exc)

    data["data_docker"] = get_docker_info(docker_content)

    data['data_processes'] = get_gpu_processes(gpu_process_ids, data["data_docker"]["docker_containers"])

    data["data_cpu"] = {"cpu_count": 0, "cpu_model": "", "cpu_clocks": []}
    
    is_supported, log_text = check_sysbox_gpu_compatibility()
    data["data_sysbox_runtime"] = is_supported
    if not is_supported:
        data["data_sysbox_runtime_scrape_error"] = log_text

    try:
        lscpu_output = run_cmd("lscpu")
        data["data_cpu"]["cpu_model"] = re.search(r"Model name:\s*(.*)$", lscpu_output, re.M).group(1)
        data["data_cpu"]["cpu_count"] = int(re.search(r"CPU\(s\):\s*(.*)", lscpu_output).group(1))
        data["data_cpu"]["cpu_utilization"] = psutil.cpu_percent(interval=1)
    except Exception as exc:
        # print(f'Error getting cpu specs: {exc}', flush=True)
        data["cpu_scrape_error"] = repr(exc)

    data["data_ram"] = {}
    try:
        # with open("/proc/meminfo") as f:
        #     meminfo = f.read()

        # for name, key in [
        #     ("MemAvailable", "available"),
        #     ("MemFree", "free"),
        #     ("MemTotal", "total"),
        # ]:
        #     data["ram"][key] = int(re.search(rf"^{name}:\s*(\d+)\s+kB$", meminfo, re.M).group(1))
        # data["ram"]["used"] = data["ram"]["total"] - data["ram"]["available"]
        # data['ram']['utilization'] = (data["ram"]["used"] / data["ram"]["total"]) * 100

        mem = psutil.virtual_memory()
        data["data_ram"] = {
            "ram_total": mem.total / 1024,  # in kB
            "ram_free": mem.free / 1024,
            "ram_used": mem.used / 1024,
            "ram_available": mem.available / 1024,
            "ram_utilization": mem.percent
        }
    except Exception as exc:
        # print(f"Error reading /proc/meminfo; Exc: {exc}", file=sys.stderr)
        data["ram_scrape_error"] = repr(exc)

    data["data_hard_disk"] = {}
    try:
        disk_usage = shutil.disk_usage(".")
        data["data_hard_disk"] = {
            "hard_disk_total": disk_usage.total // 1024,  # in kB
            "hard_disk_used": disk_usage.used // 1024,
            "hard_disk_free": disk_usage.free // 1024,
            "hard_disk_utilization": (disk_usage.used / disk_usage.total) * 100
        }
    except Exception as exc:
        # print(f"Error getting disk_usage from shutil: {exc}", file=sys.stderr)
        data["hard_disk_scrape_error"] = repr(exc)

    data["data_os"] = ""
    try:
        data["data_os"] = run_cmd('lsb_release -d | grep -Po "Description:\\s*\\K.*"').strip()
    except Exception as exc:
        # print(f'Error getting os specs: {exc}', flush=True)
        data["os_scrape_error"] = repr(exc)

    
    data["data_network"] = benchmark_network_speed()

    data["data_md5_checksums"] = {
        "md5_checksums_nvidia_smi": f"{get_md5_checksum_from_file_content(nvidia_smi_content)}:{get_sha256_checksum_from_file_content(nvidia_smi_content)}",
        "md5_checksums_libnvidia_ml": f"{get_md5_checksum_from_file_content(nvmlLib_content)}:{get_sha256_checksum_from_file_content(nvmlLib_content)}",
        "md5_checksums_docker": f"{get_md5_checksum_from_file_content(docker_content)}:{get_sha256_checksum_from_file_content(docker_content)}",
    }

    return data


def _encrypt(key: str, payload: str) -> str:
    key_bytes = b64encode(hashlib.sha256(key.encode('utf-8')).digest(), altchars=b"-_")
    return Fernet(key_bytes).encrypt(payload.encode("utf-8")).decode("utf-8")


machine_specs = get_machine_specs()
encryption_key = "".join(machine_specs["data_gpu"]["gpu_details"][0].keys())
encoded_str = _encrypt(encryption_key, json.dumps(machine_specs))
print(encoded_str)
