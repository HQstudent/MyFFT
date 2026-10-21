#pragma once
/* CPU 黄金参考实现:迭代 radix-2 Cooley-Tukey,内部双精度计算。
 * 这个文件永远保留——它是 GPU 实现的对拍基准(tests/ref_dft.h 的朴素 DFT
 * 只用于小尺寸,日常回归以本实现为主参照)。
 */
#include "myfft.h"

namespace myfft {

void CpuFftRadix2C2C(const myfftComplex* pstIn, myfftComplex* pstOut,
                     int nSize, int nBatch, int nDirection);

} // namespace myfft
