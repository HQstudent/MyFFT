/* GPU 后端 M1 实现:朴素正确性版。
 * 执行形态:每次 exec 做两次搬运(host↔device)+ 一次 kernel 发射——
 * 刻意保底的最简形态,M2 再考虑常驻缓冲与搬运摊销。
 * 本文件同时是 kernel 注册表(registry.cuh 设想)的第一版真身:
 * LOG2N 1..10 × {FORWARD, INVERSE} 查函数指针表发射。 */
#include "gpu_fft.h"
#include "kernels/radix2.cuh"
#include <cuda_runtime.h>
#include <cstring>

namespace myfft {

myfftResult GpuAllocBuffer(void** ppBuf, size_t nBytes) {
    if (!ppBuf || nBytes == 0) return MYFFT_INVALID_VALUE;
    if (cudaMalloc(ppBuf, nBytes) != cudaSuccess) return MYFFT_ALLOC_FAILED;
    return MYFFT_SUCCESS;
}

void GpuFreeBuffer(void* pBuf) {
    if (pBuf) cudaFree(pBuf);
}

/* ---- kernel 注册表(第一版)----------------------------------------------
 * 统一签名的 host 发射函数,按 (方向, log2(尺寸)) 查表取出。
 * M2 扩展方向:表项携带 grid/block/共享内存等 launch 属性,并按架构分表。 */
using PfnFft1d = void (*)(const float2* pdIn, float2* pdOut, int nBatch, int nStride);

template <int DIR, int LOG2N>
static void RunPow2Block(const float2* pdIn, float2* pdOut, int nBatch, int nStride) {
    Fft1dPow2Block<DIR, LOG2N><<<dim3(nBatch), dim3(1 << LOG2N)>>>(pdIn, pdOut, nStride);
}

static PfnFft1d SelectPow2Kernel(int nDir, int nLog2) {
    static const PfnFft1d aPfnForward[11] = {
        nullptr,
        RunPow2<-1, 1>, RunPow2<-1, 2>, RunPow2<-1, 3>, RunPow2<-1, 4>, RunPow2<-1, 5>,
        RunPow2<-1, 6>, RunPow2<-1, 7>, RunPow2<-1, 8>, RunPow2<-1, 9>, RunPow2<-1, 10>
    };
    static const PfnFft1d aPfnInverse[11] = {
        nullptr,
        RunPow2<1, 1>, RunPow2<1, 2>, RunPow2<1, 3>, RunPow2<1, 4>, RunPow2<1, 5>,
        RunPow2<1, 6>, RunPow2<1, 7>, RunPow2<1, 8>, RunPow2<1, 9>, RunPow2<1, 10>
    };
    if (nLog2 < 1 || nLog2 > 10) return nullptr;
    return (nDir < 0) ? aPfnForward[nLog2] : aPfnInverse[nLog2];
}

myfftResult GpuExecFft1d(const myfftComplex* pstIn, myfftComplex* pstOut,
                         int nSize, int nBatch, int nDir,
                         void* pDevIn, void* pDevOut) {
    if (!pstIn || !pstOut || !pDevIn || !pDevOut) return MYFFT_INVALID_VALUE;
    if (nSize <= 0 || nBatch <= 0) return MYFFT_INVALID_VALUE;

    if (nSize == 1) {   /* 单点变换是恒等(1/N 系数为 1);kernel 不覆盖 LOG2N=0 */
        if (pstIn != pstOut) std::memcpy(pstOut, pstIn, (size_t)nBatch * sizeof(myfftComplex));
        return MYFFT_SUCCESS;
    }

    int nLog2 = 0;
    while ((1 << nLog2) < nSize) ++nLog2;
    PfnFft1d pfnKernel = SelectPow2Kernel(nDir, nLog2);
    if (!pfnKernel) return MYFFT_NOT_SUPPORTED;   /* >1024:单 kernel 上限,M2 多 kernel 分解解除 */

    const size_t nBytes = (size_t)nSize * nBatch * sizeof(myfftComplex);
    float2* pdIn  = static_cast<float2*>(pDevIn);
    float2* pdOut = static_cast<float2*>(pDevOut);

    if (cudaMemcpy(pdIn, pstIn, nBytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return MYFFT_INTERNAL_ERROR;

    pfnKernel(pdIn, pdOut, nBatch, nSize);

    cudaError_t enErr = cudaGetLastError();          /* 捕获发射错误 */
    if (enErr != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        return MYFFT_INTERNAL_ERROR;

    if (cudaMemcpy(pstOut, pdOut, nBytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return MYFFT_INTERNAL_ERROR;
    return MYFFT_SUCCESS;
}

} // namespace myfft
