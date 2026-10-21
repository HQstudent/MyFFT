#pragma once
/* kernel 注册表(M2 启用)
 *
 * 形态:host 侧持有 (key -> launch 参数模板) 的静态表,heuristics 产出的
 * 路径节点携带 kernel id,executor 查表发射。新增 kernel 只改两处:
 *   1. 本文件的实例化 + 注册
 *   2. heuristics 的选择规则
 * 这是避免"加一个 kernel 改五处代码"的结构性保障。
 *
 * M2 时的样子(示意):
 *   struct KernelEntry { void (*pfnLaunch)(dim3, dim3, size_t, cudaStream_t,
 *                                          const void*, void*, int); };
 *   extern KernelEntry g_vecKernelTable[];   // 下标即 kernel id
 *   enum KernelId { K_FFT1D_POW2_BLOCK_256, K_FFT1D_POW2_BLOCK_512, ... };
 */
#include "radix2.cuh"
#include "pw.cuh"
