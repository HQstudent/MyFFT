#include "executor.h"
#include "cpu/cpu_fft.h"
#ifdef MYFFT_HAVE_CUDA
#include "gpu/gpu_fft.h"
#include <cstdlib>
#endif

namespace myfft {

#ifdef MYFFT_HAVE_CUDA
static bool UseGpuBackend(const PlanData& stPlan) {
    const char* szBackend = std::getenv("MYFFT_BACKEND");
    if (szBackend && szBackend[0] == 'c') return false;   /* MYFFT_BACKEND=cpu */
    /* M1:单 kernel 只覆盖 2..1024 点,更大尺寸回退 CPU;M2 多 kernel 分解解除 */
    return stPlan.nX >= 2 && stPlan.nX <= 1024
        && stPlan.pWorkDevIn && stPlan.pWorkDevOut;
}
#endif

myfftResult Execute(PlanData& stPlan, const myfftComplex* pstIn,
                    myfftComplex* pstOut, int nDir) {
    const size_t nTotal = (size_t)stPlan.nX * stPlan.nBatch;

#ifdef MYFFT_HAVE_CUDA
    if (UseGpuBackend(stPlan)) {
        return GpuExecFft1d(pstIn, pstOut, stPlan.nX, stPlan.nBatch, nDir,
                            stPlan.pWorkDevIn, stPlan.pWorkDevOut);
    }
#endif

    /* CPU 路径(黄金参考,永远保留) */
    const myfftComplex* pstSrc = pstIn;
    if (pstIn == pstOut) {
        stPlan.vecWorkHost.assign(pstIn, pstIn + nTotal);   /* plan 时已分配,此处仅重填 */
        pstSrc = stPlan.vecWorkHost.data();
    }

    for (const GraphNode& stNode : stPlan.vecGraph) {
        switch (stNode.enKind) {
        case OpKind::CopyIn:   /* M0 恒等:CPU kernel 直接读 pstSrc */
            break;
        case OpKind::Fft1d:
            CpuFftRadix2C2C(pstSrc, pstOut, stPlan.nX, stPlan.nBatch, nDir);
            break;
        case OpKind::Scale:    /* 已由 cpu kernel 融合(INVERSE 的 1/N) */
            break;
        }
    }
    return MYFFT_SUCCESS;
}

} // namespace myfft
