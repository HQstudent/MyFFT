#pragma once
/* GPU 后端接口(host 侧声明,不含 CUDA 头——api.cpp / executor.cpp 可直接引用)。
 * 实现在 gpu_fft.cu,仅 MYFFT_ENABLE_CUDA=ON 时参与编译;
 * 编译进库时 CMake 同时传入 MYFFT_HAVE_CUDA 宏,两处必须同步。
 */
#include "myfft.h"
#include <cstddef>

namespace myfft {

/* plan 时分配/释放设备缓冲(workspace 内部化的 GPU 部分) */
myfftResult GpuAllocBuffer(void** ppBuf, size_t nBytes);
void        GpuFreeBuffer(void* pBuf);

/* M1 朴素执行:输入(host)拷入 pDevIn → 发射 kernel → 结果从 pDevOut 拷回。
 * pDevIn/pDevOut 是 plan 时分配好的设备缓冲,执行期不再申请显存。
 * nSize 超出单 kernel 能力(2..1024)时返回 MYFFT_NOT_SUPPORTED,
 * 由 executor 决定是否回退 CPU。 */
myfftResult GpuExecFft1d(const myfftComplex* pstIn, myfftComplex* pstOut,
                         int nSize, int nBatch, int nDir,
                         void* pDevIn, void* pDevOut);

} // namespace myfft
