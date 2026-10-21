#include "plan.h"
#include "heuristics.h"

namespace myfft {

/* graph builder:按选定路径生成操作序列。
 * M0 只有一条路径;新增路径时在这里加分支,executor 只认节点不认算法,
 * 这是"加算法不动执行器"的关键。 */
static void BuildPow2SingleKernel(PlanData& stPlan) {
    stPlan.vecGraph.push_back({OpKind::CopyIn});
    stPlan.vecGraph.push_back({OpKind::Fft1d});
    stPlan.vecGraph.push_back({OpKind::Scale});   /* M0 由后端融合,节点仅作记录 */
}

myfftResult Plan1d(int nX, int nBatch, PlanData& stOut) {
    if (nX <= 0 || nBatch <= 0) return MYFFT_INVALID_VALUE;

    switch (SelectPath1d(nX, nBatch)) {
    case PathKind::Pow2SingleKernel:
        stOut.nX = nX;
        stOut.nBatch = nBatch;
        stOut.vecGraph.clear();
        BuildPow2SingleKernel(stOut);
        /* workspace 随 plan 一次性分配(CPU in-place 中转),执行期零分配 */
        stOut.vecWorkHost.assign((size_t)nX * nBatch, myfftComplex{0.f, 0.f});
        return MYFFT_SUCCESS;
    default:
        stOut.vecGraph.clear();
        stOut.vecWorkHost.clear();   /* 失败不留半成品 */
        return MYFFT_NOT_SUPPORTED;
    }
}

} // namespace myfft
