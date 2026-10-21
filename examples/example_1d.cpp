/* 用户视角的最小示例:8 点脉冲的 FFT(频谱应为常量 1) */
#include "myfft.h"
#include <cstdio>

int main() {
    myfftComplex stX[8] = {};
    stX[0] = { 1.0f, 0.0f };                   /* 时域脉冲 */

    myfftHandle hPlan;
    myfftCreate(&hPlan);
    myfftMakePlan1d(hPlan, 8, 1);
    myfftComplex stY[8];
    myfftExecC2C(hPlan, stX, stY, MYFFT_FORWARD);
    myfftDestroy(hPlan);

    std::printf("FFT of delta(8):\n");
    for (int nI = 0; nI < 8; ++nI)
        std::printf("  X[%d] = %+.6f %+.6fi\n", nI, stY[nI].x, stY[nI].y);
    return 0;
}
