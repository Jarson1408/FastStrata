#pragma once
// Included only by STRATA_ENABLE_HIP builds (the hip_compat directory precedes the ROCm includes on the
// include path).  It maps the CUDA driver API surface strata-poly uses for its virtual-memory-managed VRAM
// (cuMemCreate, cuMemMap, ...) onto HIP's memory management API, which the HIP runtime provides.
#include <hip/hip_runtime.h>

typedef hipError_t CUresult;
typedef int CUdevice;
// HIP's own CUdeviceptr/handle equivalents are pointer-typed; CUDA's are integers, and strata-poly's Mapped
// struct stores them as unsigned long long.  Keep CUDA's shape and cast in the wrappers below.
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef hipMemAllocationProp CUmemAllocationProp;
typedef hipMemAccessDesc CUmemAccessDesc;

#define CUDA_SUCCESS hipSuccess
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported

#define cuDeviceGet hipDeviceGet
#define cuDeviceGetAttribute hipDeviceGetAttribute
#define cuMemGetAllocationGranularity hipMemGetAllocationGranularity

inline CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    return hipMemRelease((hipMemGenericAllocationHandle_t) handle);
}

inline CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop,
                            unsigned long long flags) {
    return hipMemCreate((hipMemGenericAllocationHandle_t*) handle, size, prop, flags);
}

inline CUresult cuGetErrorString(CUresult err, const char** s) {
    *s = hipGetErrorString(err);
    return hipSuccess;
}
inline CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
                                    unsigned long long flags) {
    return hipMemAddressReserve((void**) ptr, size, alignment, (void*) addr, flags);
}
inline CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) { return hipMemAddressFree((void*) ptr, size); }
inline CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle,
                         unsigned long long flags) {
    return hipMemMap((void*) ptr, size, offset, (hipMemGenericAllocationHandle_t) handle, flags);
}
inline CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) { return hipMemUnmap((void*) ptr, size); }
inline CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
    return hipMemSetAccess((void*) ptr, size, desc, count);
}
