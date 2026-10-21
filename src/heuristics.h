#pragma once
/* 路径选择规则(M0 版)
 * M2:按(尺寸档, batch, 精度, 架构)查表选 kernel;
 * M3:非 2 幂 → 混合基/Bluestein 路径;
 * M5:大尺寸 → 分解+转置的多节点路径。
 */
#include "myfft.h"

namespace myfft {

enum class PathKind {
    Unsupported,        /* 当前版本不支持 */
    Pow2SingleKernel,   /* 2 的幂,单 kernel 完成(M0: CPU; M2: GPU shared mem) */
};

PathKind SelectPath1d(int nX, int nBatch);

} // namespace myfft
