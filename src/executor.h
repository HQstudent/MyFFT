#pragma once
/* 执行器:按 plan.vecGraph 顺序执行操作。
 * M1:双后端——MYFFT_HAVE_CUDA 编译期决定有无 GPU 能力,默认走 GPU;
 * 环境变量 MYFFT_BACKEND=cpu 运行期强制回退 CPU 黄金参考(对拍用)。
 * M2:GPU 路径的 CopyIn/Scale 变成真实 kernel 节点,并接入 stream 语义。
 */
#include "plan.h"

namespace myfft {

myfftResult Execute(PlanData& stPlan, const myfftComplex* pstIn,
                    myfftComplex* pstOut, int nDir);

} // namespace myfft
