/* C API 实现:只做参数校验与对象生命周期(含 workspace 的分配/释放),算法全部委托 plan/executor */
#include "myfft.h"
#include "plan.h"
#include "executor.h"
#ifdef MYFFT_HAVE_CUDA
#include "gpu/gpu_fft.h"
#endif
#include <new>

struct myfftPlan {
    myfft::PlanData stData;
};

extern "C" {

MYFFT_API myfftResult myfftCreate(myfftHandle* phPlan) {
    if (!phPlan) return MYFFT_INVALID_VALUE;
    *phPlan = new (std::nothrow) myfftPlan();
    return *phPlan ? MYFFT_SUCCESS : MYFFT_ALLOC_FAILED;
}

MYFFT_API myfftResult myfftMakePlan1d(myfftHandle hPlan, int nX, int nBatch) {
    if (!hPlan) return MYFFT_INVALID_VALUE;
    myfftResult enRc = myfft::Plan1d(nX, nBatch, hPlan->stData);

#ifdef MYFFT_HAVE_CUDA
    /* 设备缓冲随 plan 分配、随 destroy 释放;重复 MakePlan 先回收旧缓冲 */
    myfft::GpuFreeBuffer(hPlan->stData.pWorkDevIn);
    myfft::GpuFreeBuffer(hPlan->stData.pWorkDevOut);
    hPlan->stData.pWorkDevIn  = nullptr;
    hPlan->stData.pWorkDevOut = nullptr;
    if (enRc == MYFFT_SUCCESS) {
        const size_t nBytes = (size_t)nX * nBatch * sizeof(myfftComplex);
        if (myfft::GpuAllocBuffer(&hPlan->stData.pWorkDevIn,  nBytes) != MYFFT_SUCCESS ||
            myfft::GpuAllocBuffer(&hPlan->stData.pWorkDevOut, nBytes) != MYFFT_SUCCESS) {
            myfft::GpuFreeBuffer(hPlan->stData.pWorkDevIn);
            myfft::GpuFreeBuffer(hPlan->stData.pWorkDevOut);
            hPlan->stData.pWorkDevIn  = nullptr;
            hPlan->stData.pWorkDevOut = nullptr;
            return MYFFT_ALLOC_FAILED;
        }
    }
#endif
    return enRc;
}

MYFFT_API myfftResult myfftExecC2C(myfftHandle hPlan, const myfftComplex* pstIn,
                                   myfftComplex* pstOut, myfftDirection enDir) {
    if (!hPlan || !pstIn || !pstOut) return MYFFT_INVALID_VALUE;
    if (enDir != MYFFT_FORWARD && enDir != MYFFT_INVERSE) return MYFFT_INVALID_VALUE;
    if (hPlan->stData.vecGraph.empty()) return MYFFT_INVALID_VALUE;  /* 尚未 MakePlan */
    return myfft::Execute(hPlan->stData, pstIn, pstOut, enDir);
}

MYFFT_API myfftResult myfftDestroy(myfftHandle hPlan) {
#ifdef MYFFT_HAVE_CUDA
    if (hPlan) {
        myfft::GpuFreeBuffer(hPlan->stData.pWorkDevIn);
        myfft::GpuFreeBuffer(hPlan->stData.pWorkDevOut);
    }
#endif
    delete hPlan;
    return MYFFT_SUCCESS;
}

MYFFT_API const char* myfftGetErrorString(myfftResult enR) {
    switch (enR) {
    case MYFFT_SUCCESS:         return "success";
    case MYFFT_INVALID_VALUE:   return "invalid argument";
    case MYFFT_NOT_SUPPORTED:   return "size not supported by this build";
    case MYFFT_ALLOC_FAILED:    return "allocation failed";
    case MYFFT_INTERNAL_ERROR:  return "internal error";
    }
    return "unknown error";
}

MYFFT_API int myfftVersion(void) { return (0 << 16) | (1 << 8) | 0; }

} /* extern "C" */
