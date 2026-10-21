#pragma once
/* 朴素 O(N^2) DFT,双精度:只用于小尺寸(<= 1024)的黄金对拍。
 * 与 myfft 约定一致:FORWARD 无缩放,INVERSE 含 1/N。
 */
#include <vector>
#include <complex>
#include <cmath>

namespace myfft_test {

inline void NaiveDft(const std::vector<std::complex<double>>& vecX,
                     std::vector<std::complex<double>>& vecY, int nDir) {
    const size_t nCount = vecX.size();
    const double dPi = 3.14159265358979323846;
    vecY.assign(nCount, std::complex<double>(0.0, 0.0));
    for (size_t nK = 0; nK < nCount; ++nK) {
        std::complex<double> clsAcc(0.0, 0.0);
        for (size_t nJ = 0; nJ < nCount; ++nJ) {
            const double dAng = (double)nDir * 2.0 * dPi * (double)nK * (double)nJ / (double)nCount;
            clsAcc += vecX[nJ] * std::complex<double>(std::cos(dAng), std::sin(dAng));
        }
        vecY[nK] = (nDir == 1) ? clsAcc / (double)nCount : clsAcc;
    }
}

/* 相对 L2 误差(以参考值为分母),对拍用的唯一误差度量 */
inline double RelL2(const std::vector<std::complex<double>>& vecRef,
                    const myfftComplex* pstGot) {
    double dNum = 0.0, dDen = 0.0;
    for (size_t nI = 0; nI < vecRef.size(); ++nI) {
        const std::complex<double> clsG(pstGot[nI].x, pstGot[nI].y);
        const std::complex<double> clsD = vecRef[nI] - clsG;
        dNum += clsD.real() * clsD.real() + clsD.imag() * clsD.imag();
        dDen += vecRef[nI].real() * vecRef[nI].real() + vecRef[nI].imag() * vecRef[nI].imag();
    }
    return std::sqrt(dNum / (dDen + 1e-300));
}

} // namespace myfft_test
