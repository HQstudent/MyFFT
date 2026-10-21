#include "cpu_fft.h"
#include <cmath>

namespace myfft {

static inline unsigned ReverseBits(unsigned nX, int nBits) {
    unsigned nR = 0;
    for (int nI = 0; nI < nBits; ++nI) { nR = (nR << 1) | (nX & 1u); nX >>= 1; }
    return nR;
}

void CpuFftRadix2C2C(const myfftComplex* pstIn, myfftComplex* pstOut,
                     int nSize, int nBatch, int nDirection) {
    int nLevels = 0;
    while ((1 << nLevels) < nSize) ++nLevels;

    for (int nB = 0; nB < nBatch; ++nB) {
        const myfftComplex* pstSrc = pstIn + (size_t)nB * nSize;
        myfftComplex*       pstDst = pstOut + (size_t)nB * nSize;

        for (int nI = 0; nI < nSize; ++nI) pstDst[ReverseBits((unsigned)nI, nLevels)] = pstSrc[nI];

        for (int nLen = 2; nLen <= nSize; nLen <<= 1) {
            int nHalf = nLen >> 1;
            for (int nBase = 0; nBase < nSize; nBase += nLen) {
                for (int nJ = 0; nJ < nHalf; ++nJ) {
                    double dAng  = nDirection * 2.0 * 3.14159265358979323846 * nJ / nLen;
                    double dWr = std::cos(dAng), dWi = std::sin(dAng);
                    myfftComplex stA = pstDst[nBase + nJ];
                    myfftComplex stC = pstDst[nBase + nJ + nHalf];
                    double dTr = stC.x * dWr - stC.y * dWi;
                    double dTi = stC.x * dWi + stC.y * dWr;
                    pstDst[nBase + nJ].x         = (float)(stA.x + dTr);
                    pstDst[nBase + nJ].y         = (float)(stA.y + dTi);
                    pstDst[nBase + nJ + nHalf].x = (float)(stA.x - dTr);
                    pstDst[nBase + nJ + nHalf].y = (float)(stA.y - dTi);
                }
            }
        }

        if (nDirection == MYFFT_INVERSE) {
            float fS = 1.0f / (float)nSize;
            for (int nI = 0; nI < nSize; ++nI) { pstDst[nI].x *= fS; pstDst[nI].y *= fS; }
        }
    }
}

} // namespace myfft
