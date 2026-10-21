/* 1D 尺寸扫描性能测试(M0: CPU 基线;M2 起加 CUDA 后端分支与带宽利用率列)
 * 报告口径:5*N*log2(N)*batch FLOP / 时间。
 * GPU 时代真正要盯的指标是 DRAM 带宽利用率(ncu: dram__throughput),
 * CPU 阶段先建立"同一张表跟踪性能演化"的习惯。 */
#include "myfft.h"
#include <vector>
#include <chrono>
#include <cstdio>
#include <cmath>

int main() {
    std::printf("%8s %8s %12s %12s\n", "N", "batch", "best_ms", "GFLOP/s");

    for (int nSize = 1 << 8; nSize <= (1 << 16); nSize <<= 1) {
        const int nBatch = (1 << 20) / nSize;   /* 总点数恒定 ~1M,便于横向比 */
        std::vector<myfftComplex> vecIn((size_t)nSize * nBatch);
        std::vector<myfftComplex> vecOut((size_t)nSize * nBatch);
        for (size_t nI = 0; nI < vecIn.size(); ++nI) { vecIn[nI].x = (float)(nI % 97); vecIn[nI].y = 0.f; }

        myfftHandle hPlan;
        myfftCreate(&hPlan);
        myfftMakePlan1d(hPlan, nSize, nBatch);

        double dBestMs = 1e30;
        for (int nRep = 0; nRep < 5; ++nRep) {
            auto clsT0 = std::chrono::steady_clock::now();
            myfftExecC2C(hPlan, vecIn.data(), vecOut.data(), MYFFT_FORWARD);
            auto clsT1 = std::chrono::steady_clock::now();
            double dMs = std::chrono::duration<double, std::milli>(clsT1 - clsT0).count();
            if (dMs < dBestMs) dBestMs = dMs;
        }
        const double dFlop = 5.0 * nSize * std::log2((double)nSize) * nBatch;
        std::printf("%8d %8d %12.3f %12.1f\n", nSize, nBatch, dBestMs, dFlop / (dBestMs * 1e-3) / 1e9);
        myfftDestroy(hPlan);
    }
    return 0;
}
