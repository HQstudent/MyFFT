#pragma once
/* 逐点复数乘 kernel:M3 Bluestein 路径的核心部件(y_k = A_k * conj(B_k) * scale)
 * 纯访存受限(约 0.3 FLOP/byte),tensor core/CUTLASS 在此无用武之地;
 * 优化方向只有:合并访存(线性索引天然满足)、float4 向量化、与相邻操作融合。
 */
#include <cuda_runtime.h>

namespace myfft {
namespace kernels {

__global__ void PwMulConj(const float2* __restrict__ pstA,
                          const float2* __restrict__ pstB,
                          float2* __restrict__ pstY,
                          float fScale, int nCount) {
    const int nI = blockIdx.x * blockDim.x + threadIdx.x;
    if (nI >= nCount) return;
    const float2 stVa = pstA[nI], stVb = pstB[nI];
    /* (ar+i*ai)(br-i*bi) = (ar*br+ai*bi) + i*(ai*br-ar*bi) */
    pstY[nI] = make_float2((stVa.x * stVb.x + stVa.y * stVb.y) * fScale,
                           (stVa.y * stVb.x - stVa.x * stVb.y) * fScale);
}

} // namespace kernels
} // namespace myfft
