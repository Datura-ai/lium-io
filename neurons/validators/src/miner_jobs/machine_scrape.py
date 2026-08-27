from ctypes import *
import sys
import os
import glob
import http.client
import json
import re
import shutil
import socket
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
    "CHECK_STORAGE_LIMIT_ABILITY": [
        "docker", "run", "--rm",
        "--storage-opt", "size=1g",
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


def nvmlDeviceGetPowerManagementDefaultLimit(handle):
    c_limit = c_uint()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetPowerManagementDefaultLimit")
    ret = fn(handle, byref(c_limit))
    _nvmlCheckReturn(ret)
    return c_limit.value


def nvmlDeviceGetPowerManagementLimitConstraints(handle):
    c_min_limit = c_uint()
    c_max_limit = c_uint()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetPowerManagementLimitConstraints")
    ret = fn(handle, byref(c_min_limit), byref(c_max_limit))
    _nvmlCheckReturn(ret)
    return c_min_limit.value, c_max_limit.value


def safeNvmlValue(get_value):
    try:
        return get_value()
    except Exception:
        return None


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
    # Strip LD_LIBRARY_PATH so PyInstaller's bundled libs don't leak into subprocesses
    # and conflict with system-linked shared libs (e.g. libssl vs libcrypto version mismatch).
    env = {**os.environ}
    env.pop("LD_LIBRARY_PATH", None)
    proc = subprocess.run(cmd, shell=True, capture_output=True, check=False, text=True, env=env)
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

def cloudflare_speed():
    """Measure network speed using Cloudflare's speed endpoint via curl."""
    data = {"upload_speed": None, "download_speed": None}
    try:
        # Download: 50 MB
        out = run_cmd(
            "curl -o /dev/null -s -w '%{speed_download}' "
            "--max-time 15 "
            "'https://speed.cloudflare.com/__down?bytes=50000000'"
        )
        data["download_speed"] = round(float(out) * 8 / 1_000_000, 2)  # bytes/s → Mbps

        # Upload: 25 MB via stdin pipe (no temp file)
        out = run_cmd(
            "dd if=/dev/zero bs=1M count=25 2>/dev/null | "
            "curl -o /dev/null -s -w '%{speed_upload}' "
            "--max-time 15 -X POST --data-binary @- "
            "'https://speed.cloudflare.com/__up'"
        )
        data["upload_speed"] = round(float(out) * 8 / 1_000_000, 2)  # bytes/s → Mbps
    except Exception as exc:
        data["network_speed_error"] = repr(exc)
    return data


def benchmark_network_speed():
    """Run network speed methods in fallback order, stopping once both metrics are satisfied.

    Methods are tried in order: speedtest_cli → cloudflare → netmeasure → speedcheck.
    Each method is only called if at least one metric (download or upload) is still missing.
    All executed per-method raw results are stored under 'measurements' for logging.
    """
    order = [
        ("speedtest_cli", get_network_speed),
        ("cloudflare", cloudflare_speed),
        ("netmeasure", netmeasure_output),
        ("speedcheck", speedcheck_output),
    ]

    measurements = {}
    download: float | None = None
    upload: float | None = None
    download_source: str | None = None
    upload_source: str | None = None

    for name, method in order:
        if download is not None and upload is not None:
            break

        result = method()
        measurements[name] = result

        if download is None and result.get("download_speed"):
            download = result["download_speed"]
            download_source = name

        if upload is None and result.get("upload_speed"):
            upload = result["upload_speed"]
            upload_source = name

    return {
        "download_speed": download,
        "upload_speed": upload,
        "download_source": download_source,
        "upload_source": upload_source,
        "measurements": measurements,
    }

DOCKER_SOCKET_PATH = "/var/run/docker.sock"
# /system/df walks the graph driver, so it is slow on nodes with many images; cap it rather than
# let a scrape round hang on it.
DOCKER_API_TIMEOUT_SECONDS = 20
VLOOPBACK_DRIVER_PREFIX = "vloopback"


class UnixSocketHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def docker_api_get(path):
    # read one docker daemon endpoint over its unix socket; the scrape runs inside the privileged
    # executor container, where the host socket is bind-mounted. The CLI reports these sizes as
    # human strings ("45.08GB"); the API returns exact integers.
    conn = UnixSocketHTTPConnection(DOCKER_SOCKET_PATH, DOCKER_API_TIMEOUT_SECONDS)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"docker api {path} returned HTTP {response.status}")
        return json.loads(body)
    finally:
        conn.close()


def get_vloopback_volume_bytes(docker_root_dir):
    # disk held by loopback-backed volumes, which /system/df misses entirely: it only accounts for
    # the `local` driver. The plugin keeps one backing file per volume, named after the volume.
    volumes = (docker_api_get("/volumes") or {}).get("Volumes") or []
    names = [
        volume.get("Name")
        for volume in volumes
        if (volume.get("Driver") or "").startswith(VLOOPBACK_DRIVER_PREFIX)
        and volume.get("Name")
        and "/" not in volume.get("Name")
    ]
    if not names:
        return 0

    # DATA_DIR is set to <DockerRootDir>/loopback at install time, but the plugin writes it inside
    # its own rootfs; reachable here because the executor container shares the host PID namespace.
    data_dirs = glob.glob(
        f"/proc/1/root{docker_root_dir}/plugins/*/rootfs{docker_root_dir}/loopback"
    )
    if not data_dirs:
        # every volume would be counted as 0 and the breakdown would silently under-report by
        # terabytes; fail like the rest of the docker half so the miss lands in an error key
        raise RuntimeError(f"no vloopback plugin data dir under {docker_root_dir} for {len(names)} volumes")

    total = 0
    for name in names:
        for data_dir in data_dirs:
            try:
                # st_blocks, not st_size: a preallocated volume takes its whole declared size on
                # disk while holding nothing, a sparse one takes only what it wrote.
                total += os.stat(os.path.join(data_dir, name)).st_blocks * 512
                break
            except OSError:
                continue
    return total


def get_docker_disk_usage():
    # what actually filled the disk, split by kind, in kB to match the other hard_disk fields
    df = docker_api_get("/system/df")
    containers = sum(
        max(int(container.get("SizeRw") or 0), 0) for container in df.get("Containers") or []
    )
    volumes = sum(
        max(int((volume.get("UsageData") or {}).get("Size") or 0), 0)
        for volume in df.get("Volumes") or []
    )
    docker_root_dir = (docker_api_get("/info") or {}).get("DockerRootDir") or "/var/lib/docker"
    volumes += get_vloopback_volume_bytes(docker_root_dir)

    return {
        "hard_disk_images": int(df.get("LayersSize") or 0) // 1024,
        "hard_disk_containers": containers // 1024,
        "hard_disk_volumes": volumes // 1024,
    }


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
        paths = [p for p in original_path.split('\n') if p]
        for p in paths:
            if 'x86_64' in p or 'lib64' in p:
                return p
        return paths[0] if paths else ''
    except Exception:
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



def check_storage_limit_ability() -> tuple[bool, str]:
    """
    Checks if the system supports limiting the storage size of a container.
    """
    test_command = COMMANDS["CHECK_STORAGE_LIMIT_ABILITY"]

    try:
        result = subprocess.run(
            test_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return True, "Storage limit is supported."
        else:
            return False, "Storage limit is not supported."

    except subprocess.TimeoutExpired:
        return False, "Test command timed out."

    except Exception as e:
        return False, f"An unexpected error occurred: {e}"


NVIDIA_PARAMS_PATH = "/proc/driver/nvidia/params"
BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
PROC_SELF_STATUS_PATH = "/proc/self/status"
NVIDIACTL_PATH = "/dev/nvidiactl"
PROC_DIR = "/proc"
FILLER_CONTAINER_NAME_PREFIX = "filler_"
ENTRY_COMMAND_LIMIT = 200
ENTRY_REPORT_LIMIT = 20
EXEC_EVENT_WINDOW_SECONDS = 3600
EXEC_CREATE_STATUS_PREFIX = "exec_create: "
INFINIBAND_SYSFS_PATH = "/sys/class/infiniband"
# Enough of the GID table to carry both the link-local entries and the IPv4-mapped one. Its index
# is driver-specific - mlx5 puts it at 2-3, Intel irdma at 1 - so consumers must match on the
# IPV4_MAPPED_GID_PREFIX below, never on a position.
GID_TABLE_ENTRIES_READ = 4
IPV4_MAPPED_GID_PREFIX = "0000:0000:0000:0000:0000:ffff:"


class NcuProfilingObservation:
    # Plain class rather than a dataclass/NamedTuple: obfuscator.py only carries the imports on its
    # allowlist into the packaged scrape, so this file must not grow new ones.
    def __init__(self, access: str, scrape_error: str) -> None:
        self.access = access
        self.scrape_error = scrape_error


def check_ncu_profiling_access() -> NcuProfilingObservation:
    # Host driver flag RmProfilingAdminOnly: 0 -> GPU performance counters open to every workload
    # on the host ("unrestricted"), 1 -> admin-only ("restricted"). Anything unreadable stays
    # "unknown" - the backend fails closed on it; never guess a value.
    try:
        with open(NVIDIA_PARAMS_PATH) as params_file:
            params_content = params_file.read()
    except Exception as e:
        return NcuProfilingObservation("unknown", f"Cannot read {NVIDIA_PARAMS_PATH}: {e}")

    match = re.search(r"^RmProfilingAdminOnly:\s*(\d+)\s*$", params_content, re.M)
    if match is None:
        return NcuProfilingObservation(
            "unknown", "RmProfilingAdminOnly not present in nvidia driver params"
        )

    flag_value = match.group(1)
    if flag_value == "0":
        return NcuProfilingObservation("unrestricted", "")
    if flag_value == "1":
        return NcuProfilingObservation("restricted", "")
    return NcuProfilingObservation(
        "unknown", f"Unexpected RmProfilingAdminOnly value: {flag_value}"
    )


class GpuPowerCapProbe:
    # Plain class rather than a dataclass/NamedTuple: obfuscator.py only carries the imports on its
    # allowlist into the packaged scrape, so this file must not grow new ones.
    def __init__(self, cap_eff: str, nvidiactl_owner_uid: int | None, scrape_error: str) -> None:
        self.cap_eff = cap_eff
        self.nvidiactl_owner_uid = nvidiactl_owner_uid
        self.scrape_error = scrape_error


def probe_gpu_power_cap_ability() -> GpuPowerCapProbe:
    # DAH-2704: `nvidia-smi -pl` (the PEARL filler's power cap) exits 4 unless the executor container
    # holds CAP_SYS_ADMIN *and* its root owns /dev/nvidiactl - a sysbox userns maps the device to
    # nobody while keeping every capability, so neither reading alone tells the two apart. Report both
    # raw values and let the backend decide; an unreadable one stays empty/None, never a guess.
    # /proc/self/status is this scrape's own process, which is meaningful only because the scrape and
    # the validator's `nvidia-smi -pl` run in the same SSH context - keep them together.
    scrape_errors: list[str] = []

    cap_eff: str = ""
    try:
        with open(PROC_SELF_STATUS_PATH) as status_file:
            status_content = status_file.read()
        match = re.search(r"^CapEff:\s*([0-9a-fA-F]+)\s*$", status_content, re.M)
        if match is None:
            scrape_errors.append(f"CapEff not present in {PROC_SELF_STATUS_PATH}")
        else:
            cap_eff = match.group(1)
    except Exception as e:
        scrape_errors.append(f"Cannot read {PROC_SELF_STATUS_PATH}: {e}")

    nvidiactl_owner_uid: int | None = None
    try:
        nvidiactl_owner_uid = os.stat(NVIDIACTL_PATH).st_uid
    except Exception as e:
        scrape_errors.append(f"Cannot stat {NVIDIACTL_PATH}: {e}")

    return GpuPowerCapProbe(cap_eff, nvidiactl_owner_uid, "; ".join(scrape_errors))


class FillerEntryProbe:
    # Plain class rather than a dataclass/NamedTuple: obfuscator.py only carries the imports on its
    # allowlist into the packaged scrape, so this file must not grow new ones.
    def __init__(self, entries: list, scrape_error: str) -> None:
        self.entries = entries
        self.scrape_error = scrape_error


def docker_api_get_events(path: str) -> list:
    # The events endpoint answers with one JSON object per line, so it needs its own reader.
    # A `since`/`until` pair that both lie in the past makes the daemon send its buffered
    # events and close, instead of streaming new ones forever.
    conn = UnixSocketHTTPConnection(DOCKER_SOCKET_PATH, DOCKER_API_TIMEOUT_SECONDS)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"docker api {path} returned HTTP {response.status}")
        return [json.loads(line) for line in body.splitlines() if line.strip()]
    finally:
        conn.close()


def read_pid_namespace(pid: int) -> str:
    # The kernel prints one identifier per namespace ("pid:[4026532817]"), so two processes are in
    # the same PID namespace when this string matches.
    return os.readlink(f"{PROC_DIR}/{pid}/ns/pid")


def read_process_parent_and_start_ticks(pid: int) -> tuple:
    # /proc/<pid>/stat holds the command name in brackets, and it may contain spaces and brackets
    # of its own, so the fields are counted from the LAST bracket: parent pid is the 2nd of them
    # and the start time the 20th. The start time counts clock ticks since the host booted.
    with open(f"{PROC_DIR}/{pid}/stat") as stat_file:
        stat_line = stat_file.read()
    fields = stat_line[stat_line.rindex(")") + 1:].split()
    return int(fields[1]), int(fields[19])


def read_process_command(pid: int) -> str:
    try:
        with open(f"{PROC_DIR}/{pid}/cmdline", "rb") as cmdline_file:
            command = cmdline_file.read().decode("utf-8", "replace")
    except Exception:
        return ""
    return command.replace("\0", " ").strip()[:ENTRY_COMMAND_LIMIT]


def read_host_uptime_seconds() -> float:
    with open(f"{PROC_DIR}/uptime") as uptime_file:
        return float(uptime_file.read().split()[0])


def find_open_sessions(init_pid: int, container_name: str, container_start_ticks: int) -> list:
    """Sessions still open in the container that were started from outside it.

    Everything a container starts itself has a parent in the same PID namespace, or is an orphan
    the container's own init adopted. `docker exec` and `nsenter` are the two ways in from the
    host, and both leave the same mark: the new process joins the namespace while its parent
    stays outside it. Only the session itself is reported - what it then runs has a parent inside.
    This is the only way to see an `nsenter`, which never reaches the docker daemon.
    """
    container_namespace = read_pid_namespace(init_pid)
    if container_namespace == read_pid_namespace(os.getpid()):
        # The container shares the host PID namespace, where every process looks like a member
        # and the parentage tells nothing apart. Report none rather than the whole host.
        return []
    clock_ticks_per_second = os.sysconf("SC_CLK_TCK")

    members = {}
    for process_dir in os.listdir(PROC_DIR):
        if not process_dir.isdigit():
            continue
        member_pid = int(process_dir)
        try:
            if read_pid_namespace(member_pid) != container_namespace:
                continue
            members[member_pid] = read_process_parent_and_start_ticks(member_pid)
        except Exception:
            continue  # /proc is a moving target: the process exited during the scan

    found = []
    for member_pid, (parent_pid, start_ticks) in members.items():
        if member_pid == init_pid or parent_pid in members:
            continue
        found.append({
            "entry_container": container_name,
            "entry_kind": "open_session",
            "entry_pid": member_pid,
            "entry_seconds_after_start": round(
                (start_ticks - container_start_ticks) / clock_ticks_per_second, 1
            ),
            "entry_command": read_process_command(member_pid),
        })
    return found


def find_docker_exec_events(container_start_times: dict) -> list:
    """Every `docker exec` the daemon recorded for these containers, finished ones included.

    A guard script execs for milliseconds, so the live scan above almost never meets one. The
    daemon keeps its last events in memory and hands them out with the command line, which turns
    those short visits into a fact we can read once per cycle. `container_start_times` maps a
    filler container name to when it started, in seconds since the host booted.
    """
    boot_time_unix = psutil.boot_time()
    now_unix = boot_time_unix + read_host_uptime_seconds()
    events = docker_api_get_events(
        f"/events?since={int(now_unix - EXEC_EVENT_WINDOW_SECONDS)}&until={int(now_unix)}"
    )

    found = []
    for event in events:
        status = str(event.get("status") or "")
        if not status.startswith(EXEC_CREATE_STATUS_PREFIX):
            continue
        container_name = ((event.get("Actor") or {}).get("Attributes") or {}).get("name") or ""
        if container_name not in container_start_times:
            continue
        event_unix = event.get("time")
        if not isinstance(event_unix, (int, float)) or isinstance(event_unix, bool):
            continue
        container_start_unix = boot_time_unix + container_start_times[container_name]
        found.append({
            "entry_container": container_name,
            "entry_kind": "docker_exec",
            "entry_pid": None,  # the session is over; the daemon keeps the command, not the pid
            "entry_seconds_after_start": round(event_unix - container_start_unix, 1),
            "entry_command": status[len(EXEC_CREATE_STATUS_PREFIX):][:ENTRY_COMMAND_LIMIT],
        })
    return found


def entry_offset_seconds(entry: dict) -> float:
    # A plain function, not a lambda: obfuscator.py renames a lambda's argument only where it is
    # read, which leaves the packaged scrape referring to a name nothing binds.
    return entry["entry_seconds_after_start"]


def probe_filler_container_entries() -> FillerEntryProbe:
    # DAH-2787: an idle node earns the unrented incentive while it runs Lium's own job, and nobody
    # may work inside that job's container. Report the visits raw, each with the seconds between
    # the container's start and the visit - the validator holds the grace that keeps our own setup
    # execs, which all happen while the container is being created, out of the verdict.
    scrape_errors = []
    found = []
    try:
        containers = docker_api_get("/containers/json") or []
    except Exception as e:
        return FillerEntryProbe([], f"Cannot list containers: {e}")

    clock_ticks_per_second = os.sysconf("SC_CLK_TCK")
    container_start_times = {}
    for container in containers:
        container_name = ""
        for name in container.get("Names") or []:
            if name.lstrip("/").startswith(FILLER_CONTAINER_NAME_PREFIX):
                container_name = name.lstrip("/")
                break
        if not container_name:
            continue
        try:
            details = docker_api_get(f"/containers/{container.get('Id')}/json")
            init_pid = int(((details or {}).get("State") or {}).get("Pid") or 0)
            if init_pid <= 0:  # 0 means the container is not running
                continue
            container_start_ticks = read_process_parent_and_start_ticks(init_pid)[1]
            container_start_times[container_name] = container_start_ticks / clock_ticks_per_second
            found.extend(find_open_sessions(init_pid, container_name, container_start_ticks))
        except Exception as e:
            scrape_errors.append(f"Cannot read {container_name}: {e}")

    if container_start_times:
        try:
            found.extend(find_docker_exec_events(container_start_times))
        except Exception as e:
            scrape_errors.append(f"Cannot read docker events: {e}")

    # A guard script that execs every few minutes fills a whole window with the same visit; the
    # newest ones carry the same proof in a payload the backend can hold.
    found.sort(key=entry_offset_seconds, reverse=True)
    return FillerEntryProbe(found[:ENTRY_REPORT_LIMIT], "; ".join(scrape_errors))


def get_host_boot_id() -> str:
    # Reboot marker: the profiling flag only changes on a driver reload/reboot, so a changed
    # boot_id tells the backend a stale observation may have flipped (DAH-2182).
    try:
        with open(BOOT_ID_PATH) as boot_id_file:
            return boot_id_file.read().strip()
    except Exception:
        return ""


def read_sysfs_value(path: str) -> str:
    try:
        with open(path) as sysfs_file:
            return sysfs_file.read().strip()
    except Exception:
        return ""


class InfinibandPort:
    # Plain class rather than a dataclass/NamedTuple: obfuscator.py only carries the imports on its
    # allowlist into the packaged scrape, so this file must not grow new ones.
    def __init__(
        self,
        device: str,
        port: str,
        node_guid: str,
        sys_image_guid: str,
        link_layer: str,
        state: str,
        phys_state: str,
        rate: str,
        lid: str,
        sm_lid: str,
        pkey: str,
        gids: list[str],
    ) -> None:
        self.device = device
        self.port = port
        self.node_guid = node_guid
        self.sys_image_guid = sys_image_guid
        self.link_layer = link_layer
        self.state = state
        self.phys_state = phys_state
        self.rate = rate
        self.lid = lid
        self.sm_lid = sm_lid
        self.pkey = pkey
        self.gids = gids

    def as_payload(self) -> dict[str, str | list[str]]:
        return {
            "ib_device": self.device,
            "ib_port": self.port,
            "ib_node_guid": self.node_guid,
            "ib_sys_image_guid": self.sys_image_guid,
            "ib_link_layer": self.link_layer,
            "ib_state": self.state,
            "ib_phys_state": self.phys_state,
            "ib_rate": self.rate,
            "ib_lid": self.lid,
            "ib_sm_lid": self.sm_lid,
            "ib_pkey": self.pkey,
            "ib_gids": self.gids,
        }


def read_infiniband_port(
    device_path: str, node_guid: str, sys_image_guid: str, port_path: str
) -> InfinibandPort:
    # Whole GID table, not just gids/0: every prod port carries the default fe80:: prefix, so the
    # prefix alone identifies nothing. The IPv4-mapped entry is the one that tells two Ethernet
    # ports they share a segment - find it by IPV4_MAPPED_GID_PREFIX, its index moves by driver.
    gids = [read_sysfs_value(f"{port_path}/gids/{index}") for index in range(GID_TABLE_ENTRIES_READ)]
    # POSITIONAL ON PURPOSE: obfuscator.py renames __init__ parameters but leaves keyword names at
    # the call site, so a keyword call raises TypeError in the packaged scrape and nowhere else.
    # That is what made the first prod rollout return an empty list on every host (DAH-2571).
    return InfinibandPort(
        os.path.basename(device_path),
        os.path.basename(port_path),
        node_guid,
        sys_image_guid,
        read_sysfs_value(f"{port_path}/link_layer"),
        read_sysfs_value(f"{port_path}/state"),
        read_sysfs_value(f"{port_path}/phys_state"),
        read_sysfs_value(f"{port_path}/rate"),
        read_sysfs_value(f"{port_path}/lid"),
        read_sysfs_value(f"{port_path}/sm_lid"),
        read_sysfs_value(f"{port_path}/pkeys/0"),
        [gid for gid in gids if gid],
    )


class InfinibandObservation:
    # Plain class rather than a dataclass/NamedTuple: obfuscator.py only carries the imports on its
    # allowlist into the packaged scrape, so this file must not grow new ones.
    def __init__(self, ports: list[InfinibandPort], scrape_error: str) -> None:
        self.ports = ports
        self.scrape_error = scrape_error


def get_infiniband_ports() -> InfinibandObservation:
    """Every RDMA port the host exposes, as reported by the kernel (DAH-2571).

    Facts only - which ports are usable and which fabric each one sits on. Deciding whether two
    machines are on the same fabric is the backend's job: it needs both hosts, this sees one.

    An empty port list has three different causes and they must stay distinguishable: the host has
    no RDMA hardware, the scrape cannot see the sysfs tree from where it runs, or the walk itself
    failed. Reporting [] for all three is what made the first prod rollout unreadable - every
    executor came back empty, including hosts known to carry 24 mlx5 devices, with no way to tell
    which case it was.
    """
    if not os.path.isdir(INFINIBAND_SYSFS_PATH):
        return InfinibandObservation([], f"{INFINIBAND_SYSFS_PATH} does not exist")

    ports: list[InfinibandPort] = []
    try:
        # os.listdir, not glob: glob swallows EACCES/EIO and returns [], which would be reported as
        # "lists no devices" - the same kind of silent nothing this function exists to stop.
        device_names = sorted(os.listdir(INFINIBAND_SYSFS_PATH))
        for device_name in device_names:
            device_path = f"{INFINIBAND_SYSFS_PATH}/{device_name}"
            node_guid = read_sysfs_value(f"{device_path}/node_guid")
            sys_image_guid = read_sysfs_value(f"{device_path}/sys_image_guid")
            ports_path = f"{device_path}/ports"
            if not os.path.isdir(ports_path):
                continue
            for port_name in sorted(os.listdir(ports_path)):
                ports.append(
                    read_infiniband_port(device_path, node_guid, sys_image_guid, f"{ports_path}/{port_name}")
                )
    except Exception as e:
        # An interconnect reading is never worth failing the whole machine scrape over, but the
        # reason has to travel with the empty result.
        return InfinibandObservation(ports, f"Error walking {INFINIBAND_SYSFS_PATH}: {e}")

    if not device_names:
        return InfinibandObservation([], f"{INFINIBAND_SYSFS_PATH} lists no devices")
    if not ports:
        return InfinibandObservation([], f"{len(device_names)} device(s) present, none exposing a port")
    return InfinibandObservation(ports, "")


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
            memory_info = nvmlDeviceGetMemoryInfo(handle)

            data["data_gpu"]["gpu_details"].append(
                {
                    "gpu.name": nvmlDeviceGetName(handle),
                    "gpu.uuid": nvmlDeviceGetUUID(handle),
                    "gpu.capacity": memory_info.c_nvmlMemory_t_total / (1024 ** 2),  # in MB
                    # Not interchangeable with gpu.memory_utilization below: that one is NVML's
                    # memory-BUS duty cycle, which drops to 0 on a loaded-but-idle GPU.
                    # Reads a few hundred MB above `nvidia-smi memory.used` because NVML v1 counts the
                    # driver-reserved block (measured: 386 MB on an A4000, 728 MB on a B200).
                    "gpu.memory_used_mb": memory_info.c_nvmlMemory_t_used / (1024 ** 2),  # in MB
                    "gpu.cuda": f"{major}.{minor}",
                    "gpu.power_limit": nvmlDeviceGetPowerManagementLimit(handle) / 1000,
                    "gpu.power_default_limit": safeNvmlValue(
                        lambda: nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000
                    ),
                    "gpu.power_max_limit": safeNvmlValue(
                        lambda: nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000
                    ),
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
        
    is_supported, log_text = check_storage_limit_ability()
    data["data_storage_limit_supported"] = is_supported
    if not is_supported:
        data["data_storage_limit_scrape_error"] = log_text

    ncu_profiling = check_ncu_profiling_access()
    data["data_ncu_profiling_access"] = ncu_profiling.access
    if ncu_profiling.scrape_error:
        data["data_ncu_profiling_scrape_error"] = ncu_profiling.scrape_error
    data["data_boot_id"] = get_host_boot_id()

    power_cap_probe = probe_gpu_power_cap_ability()
    data["data_container_cap_eff"] = power_cap_probe.cap_eff
    data["data_nvidiactl_owner_uid"] = power_cap_probe.nvidiactl_owner_uid
    if power_cap_probe.scrape_error:
        data["data_power_cap_probe_error"] = power_cap_probe.scrape_error

    filler_entry_probe = probe_filler_container_entries()
    data["data_filler_entries"] = filler_entry_probe.entries
    if filler_entry_probe.scrape_error:
        data["data_filler_entry_scrape_error"] = filler_entry_probe.scrape_error

    infiniband = get_infiniband_ports()
    data["data_infiniband_ports"] = [port.as_payload() for port in infiniband.ports]
    if infiniband.scrape_error:
        data["data_infiniband_scrape_error"] = infiniband.scrape_error

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
        disk_usage = shutil.disk_usage("/")
        data["data_hard_disk"] = {
            "hard_disk_total": disk_usage.total // 1024,  # in kB
            "hard_disk_used": disk_usage.used // 1024,
            "hard_disk_free": disk_usage.free // 1024,
            "hard_disk_utilization": (disk_usage.used / disk_usage.total) * 100
        }
    except Exception as exc:
        # print(f"Error getting disk_usage from shutil: {exc}", file=sys.stderr)
        data["hard_disk_scrape_error"] = repr(exc)

    try:
        data["data_hard_disk"].update(get_docker_disk_usage())
    except Exception as exc:
        # kept apart from hard_disk_scrape_error: the docker socket is the fragile half, and a
        # node that loses only the breakdown must keep reporting total/used/free.
        data["hard_disk_docker_scrape_error"] = repr(exc)

    data["data_os"] = ""
    try:
        data["data_os"] = run_cmd('lsb_release -d | grep -Po "Description:\\s*\\K.*"').strip()
    except Exception as exc:
        # print(f'Error getting os specs: {exc}', flush=True)
        data["os_scrape_error"] = repr(exc)

    # uname reports the host kernel even from inside the container, unlike lsb_release above
    data["data_kernel"] = ""
    try:
        data["data_kernel"] = run_cmd("uname -r").strip()
    except Exception as exc:
        data["kernel_scrape_error"] = repr(exc)

    
    data["data_network"] = benchmark_network_speed()

    data["data_md5_checksums"] = {
        "md5_checksums_nvidia_smi": f"{get_md5_checksum_from_file_content(nvidia_smi_content)}:{get_sha256_checksum_from_file_content(nvidia_smi_content)}",
        "md5_checksums_libnvidia_ml": f"{get_md5_checksum_from_file_content(nvmlLib_content)}:{get_sha256_checksum_from_file_content(nvmlLib_content)}",
        "md5_checksums_docker": f"{get_md5_checksum_from_file_content(docker_content)}:{get_sha256_checksum_from_file_content(docker_content)}",
    }

    if not data.get("data_gpu", {}).get("gpu_details", []):
        print(json.dumps({"error": "no_gpu_details", "data": data}))
        sys.exit(1)

    return data


def _encrypt(key: str, payload: str) -> str:
    key_bytes = b64encode(hashlib.sha256(key.encode('utf-8')).digest(), altchars=b"-_")
    return Fernet(key_bytes).encrypt(payload.encode("utf-8")).decode("utf-8")


machine_specs = get_machine_specs()
encryption_key = "".join(machine_specs["data_gpu"]["gpu_details"][0].keys())
encoded_str = _encrypt(encryption_key, json.dumps(machine_specs))
print(encoded_str)
