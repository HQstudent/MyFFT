#pragma once
/* ============================================================================
 * M2 目标代码:GPU radix-2 block kernel(每 block 一条变换,n <= 1024)
 *
 * 状态:已写好待编译验证——需要装 CUDA Toolkit 后开启
 *       MYFFT_ENABLE_CUDA=ON,第一步就是对拍 CPU 黄金参考。
 * 设计要点(M2 的优化方向都在这):
 *   - shared memory 中完成全部 butterfly,全局访存只有一读一写(理想带宽路径)
 *   - 位反转在 shared memory 写入时完成,省独立置换 pass;M2 换 Stockham 彻底消除
 *   - LOG2N/N 为编译期常量:循环完全展开,twiddle 常量折叠
 *   - M2 升级点:多变换/block 提高占用率、__shfl_xor_sync 做 warp 内交换、
 *     寄存器驻留小尺寸、float4 向量化 IO、blockDim 模板扫描选最优
 * ==========================================================================*/
#include <cuda_runtime.h>

namespace myfft {
namespace kernels {

constexpr int kMaxBlockPoints = 1024;   /* 单 block 上限(1 block = 1 transform) */

template <int DIR, int LOG2N>
__global__ void Fft1dPow2Block(const float2* __restrict__ pstIn,
                               float2* __restrict__ pstOut,
                               int nBatchStride) {
    constexpr int N = 1 << LOG2N;
    __shared__ float2 aSmem[N];

    const int nT = threadIdx.x;
    /* 位反转读入:DIT 需要位反转序;__brev 是单条 PTX 指令 */
    aSmem[__brev(nT) >> (32 - LOG2N)] = pstIn[blockIdx.x * nBatchStride + nT];

    #pragma unroll
    for (int nLen = 2; nLen <= N; nLen <<= 1) {
        __syncthreads();
        const int nHalf = nLen >> 1;   /* nLen 为循环变量,只能 const;完全展开后仍是编译期值 */
        const int nBase = (nT / nLen) * nLen;
        const int nJ    = nT % nLen;
        if (nJ < nHalf) {
            const float fAng = (float)DIR * 6.28318530717958647692f * nJ / nLen;
            /* M2 起换 sincosf/查表;__cosf/__sinf 为快速近似,先保正确性用标准版 */
            const float2 stW = make_float2(cosf(fAng), sinf(fAng));
            const float2 stA = aSmem[nBase + nJ];
            const float2 stC = aSmem[nBase + nJ + nHalf];
            const float fTr = stC.x * stW.x - stC.y * stW.y;
            const float fTi = stC.x * stW.y + stC.y * stW.x;
            aSmem[nBase + nJ]         = make_float2(stA.x + fTr, stA.y + fTi);
            aSmem[nBase + nJ + nHalf] = make_float2(stA.x - fTr, stA.y - fTi);
        }
    }
    __syncthreads();

    if (DIR == 1) {  /* INVERSE: 1/N 缩放融合在写回处 */
        const float fS = 1.0f / (float)N;
        pstOut[blockIdx.x * nBatchStride + nT] = make_float2(aSmem[nT].x * fS, aSmem[nT].y * fS);
    } else {
        pstOut[blockIdx.x * nBatchStride + nT] = aSmem[nT];
    }
}

} // namespace kernels
} // namespace myfft
