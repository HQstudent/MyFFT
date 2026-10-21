/* 正确性测试:零外部依赖的自写 harness。
 * 回归基线:1) 小尺寸对拍朴素 DFT  2) round-trip 恒等  3) 参数校验  4) in-place
 * 从 M2 起,同一套用例跑在 GPU 后端上,GPU vs CPU 对拍也走 RelL2。 */
#include "myfft.h"
#include "ref_dft.h"
#include <vector>
#include <complex>
#include <cstdio>
#include <cstdlib>

static int g_nPass = 0, g_nFail = 0;
#define CHECK(cond, msg) \
    do { if (cond) { ++g_nPass; } else { ++g_nFail; std::printf("  FAIL: %s (line %d)\n", msg, __LINE__); } } while (0)

static std::vector<myfftComplex> RandomInput(int nSize, unsigned nSeed) {
    std::srand(nSeed);
    std::vector<myfftComplex> vecData((size_t)nSize);
    for (int nI = 0; nI < nSize; ++nI) {
        vecData[nI].x = (float)(std::rand() / (double)RAND_MAX * 2.0 - 1.0);
        vecData[nI].y = (float)(std::rand() / (double)RAND_MAX * 2.0 - 1.0);
    }
    return vecData;
}

/* 小尺寸:与朴素 DFT 对拍 */
static void TestVsNaive() {
    for (int nSize = 1; nSize <= 1024; nSize <<= 1) {
        for (int nDir = -1; nDir <= 1; nDir += 2) {
            auto vecInput = RandomInput(nSize, 42u + (unsigned)nSize + (unsigned)nDir);
            std::vector<std::complex<double>> vecRefIn(nSize);
            for (int nI = 0; nI < nSize; ++nI) vecRefIn[nI] = { vecInput[nI].x, vecInput[nI].y };
            std::vector<std::complex<double>> vecRef;
            myfft_test::NaiveDft(vecRefIn, vecRef, nDir);

            std::vector<myfftComplex> vecOut(nSize);
            myfftHandle hPlan;
            CHECK(myfftCreate(&hPlan) == MYFFT_SUCCESS, "create");
            CHECK(myfftMakePlan1d(hPlan, nSize, 1) == MYFFT_SUCCESS, "make plan");
            CHECK(myfftExecC2C(hPlan, vecInput.data(), vecOut.data(), (myfftDirection)nDir) == MYFFT_SUCCESS, "exec");
            double dErr = myfft_test::RelL2(vecRef, vecOut.data());
            CHECK(dErr < 1e-5, "RelL2 < 1e-5");
            myfftDestroy(hPlan);
        }
    }
}

/* 大尺寸:FFT^-1(FFT(x)) == x */
static void TestRoundtrip() {
    for (int nSize = 2048; nSize <= (1 << 16); nSize <<= 1) {
        auto vecInput = RandomInput(nSize, 7u);
        std::vector<myfftComplex> vecMid(nSize), vecBack(nSize);
        myfftHandle hPlan;
        myfftCreate(&hPlan);
        myfftMakePlan1d(hPlan, nSize, 1);
        myfftExecC2C(hPlan, vecInput.data(), vecMid.data(), MYFFT_FORWARD);
        myfftExecC2C(hPlan, vecMid.data(), vecBack.data(), MYFFT_INVERSE);
        std::vector<std::complex<double>> vecRef(nSize);
        for (int nI = 0; nI < nSize; ++nI) vecRef[nI] = { vecInput[nI].x, vecInput[nI].y };
        CHECK(myfft_test::RelL2(vecRef, vecBack.data()) < 1e-5, "roundtrip identity");
        myfftDestroy(hPlan);
    }
}

/* batch:等价于逐条变换 */
static void TestBatch() {
    const int nSize = 256, nBatch = 4;
    myfftHandle hPlan;
    myfftCreate(&hPlan);
    CHECK(myfftMakePlan1d(hPlan, nSize, nBatch) == MYFFT_SUCCESS, "batch plan");
    std::vector<myfftComplex> vecAll;
    for (int nB = 0; nB < nBatch; ++nB) {
        auto vecOne = RandomInput(nSize, 100u + (unsigned)nB);
        vecAll.insert(vecAll.end(), vecOne.begin(), vecOne.end());
    }
    std::vector<myfftComplex> vecOut((size_t)nSize * nBatch);
    CHECK(myfftExecC2C(hPlan, vecAll.data(), vecOut.data(), MYFFT_FORWARD) == MYFFT_SUCCESS, "batch exec");
    bool bOk = true;
    for (int nB = 0; nB < nBatch && bOk; ++nB) {
        std::vector<std::complex<double>> vecRefIn(nSize);
        for (int nI = 0; nI < nSize; ++nI) vecRefIn[nI] = { vecAll[(size_t)nB * nSize + nI].x, vecAll[(size_t)nB * nSize + nI].y };
        std::vector<std::complex<double>> vecRef;
        myfft_test::NaiveDft(vecRefIn, vecRef, -1);
        bOk = myfft_test::RelL2(vecRef, vecOut.data() + (size_t)nB * nSize) < 1e-5;
    }
    CHECK(bOk, "batch == per-transform");
    myfftDestroy(hPlan);
}

/* in-place:out == in */
static void TestInplace() {
    const int nSize = 512;
    auto vecInput = RandomInput(nSize, 9u);
    std::vector<std::complex<double>> vecRefIn(nSize);
    for (int nI = 0; nI < nSize; ++nI) vecRefIn[nI] = { vecInput[nI].x, vecInput[nI].y };
    std::vector<std::complex<double>> vecRef;
    myfft_test::NaiveDft(vecRefIn, vecRef, -1);

    myfftHandle hPlan;
    myfftCreate(&hPlan);
    myfftMakePlan1d(hPlan, nSize, 1);
    CHECK(myfftExecC2C(hPlan, vecInput.data(), vecInput.data(), MYFFT_FORWARD) == MYFFT_SUCCESS, "inplace exec");
    CHECK(myfft_test::RelL2(vecRef, vecInput.data()) < 1e-5, "inplace result");
    myfftDestroy(hPlan);
}

/* 非法参数与不支持的尺寸 */
static void TestInvalid() {
    myfftHandle hPlan;
    CHECK(myfftCreate(nullptr) == MYFFT_INVALID_VALUE, "null handle ptr");
    myfftCreate(&hPlan);
    CHECK(myfftMakePlan1d(hPlan, 0, 1) == MYFFT_INVALID_VALUE, "nx=0");
    CHECK(myfftMakePlan1d(hPlan, -8, 1) == MYFFT_INVALID_VALUE, "nx<0");
    CHECK(myfftMakePlan1d(hPlan, 64, 0) == MYFFT_INVALID_VALUE, "batch=0");
    CHECK(myfftMakePlan1d(hPlan, 100, 1) == MYFFT_NOT_SUPPORTED, "nx=100 (M0: pow2 only)");
    myfftDestroy(hPlan);
    myfftDestroy(nullptr);   /* 不应崩溃 */
}

int main() {
    const char* szBackend = std::getenv("MYFFT_BACKEND");
    std::printf("[myfft] version 0x%06x  backend request: %s\n",
                myfftVersion(), szBackend ? szBackend : "(default)");
    std::printf("== TestVsNaive ==\n");    TestVsNaive();
    std::printf("== TestRoundtrip ==\n");  TestRoundtrip();
    std::printf("== TestBatch ==\n");      TestBatch();
    std::printf("== TestInplace ==\n");    TestInplace();
    std::printf("== TestInvalid ==\n");    TestInvalid();
    std::printf("\n%d passed, %d failed\n", g_nPass, g_nFail);
    return g_nFail == 0 ? 0 : 1;
}
