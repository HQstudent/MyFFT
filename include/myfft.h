#ifndef MYFFT_H
#define MYFFT_H
/*
 * myfft — 自研 GPU FFT 库
 *
 * 公共 C API(唯一对外头文件),模仿 cuFFT 的 plan-based 模型:
 *   myfftCreate -> myfftMakePlan1d -> myfftExecC2C(可重复) -> myfftDestroy
 *
 * ABI 策略:纯 C 接口 + 不透明 handle;主版本号即 ABI 版本
 * (libmyfft.so.1 / myfft1.dll)。
 */

#ifdef MYFFT_STATIC            /* 静态拼接开发/测试模式:不产生导入导出符号 */
#  define MYFFT_API
#elif defined(_WIN32)
#  ifdef MYFFT_BUILD
#    define MYFFT_API __declspec(dllexport)
#  else
#    define MYFFT_API __declspec(dllimport)
#  endif
#else
#  define MYFFT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { float x, y; } myfftComplex;

typedef struct myfftPlan* myfftHandle;

/* 变换方向 = 指数符号约定,与 cuFFT 一致 */
typedef enum {
    MYFFT_FORWARD = -1,   /* exp(-i*2*pi*k*n/N) */
    MYFFT_INVERSE =  1    /* exp(+i*2*pi*k*n/N),含 1/N 缩放 */
} myfftDirection;

typedef enum {
    MYFFT_SUCCESS = 0,
    MYFFT_INVALID_VALUE,   /* 参数非法:空指针、nx<=0、batch<=0 */
    MYFFT_NOT_SUPPORTED,   /* 当前版本不支持的尺寸(M0:仅 2 的幂) */
    MYFFT_ALLOC_FAILED,
    MYFFT_INTERNAL_ERROR
} myfftResult;

MYFFT_API myfftResult myfftCreate(myfftHandle* plan);
MYFFT_API myfftResult myfftMakePlan1d(myfftHandle plan, int nx, int batch);
MYFFT_API myfftResult myfftExecC2C(myfftHandle plan,
                                   const myfftComplex* in,
                                   myfftComplex* out,
                                   myfftDirection dir);
MYFFT_API myfftResult myfftDestroy(myfftHandle plan);

MYFFT_API const char* myfftGetErrorString(myfftResult r);
MYFFT_API int myfftVersion(void);   /* (major<<16)|(minor<<8)|patch */

#ifdef __cplusplus
}
#endif

#endif /* MYFFT_H */
