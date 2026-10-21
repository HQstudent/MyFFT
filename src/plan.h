#pragma once
/* 内部头文件:plan 数据结构与规划接口(非公开 API) */
#include <vector>
#include "myfft.h"

namespace myfft {

/* ---- graph IR:一个 plan 编译成一串操作节点 ------------------------------
 * M0 只有 CPU 一个 Fft1d 节点;M2 起 CUDA kernel、transpose、Bluestein 的
 * 逐点乘等都以节点形式进入序列,由 executor 统一发射。
 * ---------------------------------------------------------------------- */
enum class OpKind {
    CopyIn,    /* 输入落位(处理 in-place/布局);M0 为恒等 */
    Fft1d,     /* 1D 变换本体 */
    Scale,     /* 缩放;M0 由 CPU kernel 融合执行 */
};

struct GraphNode {
    OpKind enKind;
    /* M2+:kernel id、grid/block、shared memory 尺寸等执行属性放这里 */
};

struct PlanData {
    int  nX = 0;
    int  nBatch = 1;
    std::vector<GraphNode> vecGraph;   /* 按序执行的操作序列 */

    /* M1 工作区:plan 时分配、destroy 时释放(内部化,不进公共 API)。
     * vecWorkHost:CPU 后端 in-place 执行的中转缓冲;
     * pWorkDev*:GPU 设备缓冲(输入/输出各一),以 void* 存放是为了让本
     * 头文件不引入 CUDA 头——分配/释放在 gpu 模块内实现,
     * 由 api.cpp 的生命周期函数(MakePlan/Destroy)调用。 */
    std::vector<myfftComplex> vecWorkHost;
    void* pWorkDevIn  = nullptr;
    void* pWorkDevOut = nullptr;
};

/* 校验参数 + 选路径 + 生成 graph。内部调用 heuristics 与 graph builder */
myfftResult Plan1d(int nX, int nBatch, PlanData& stOut);

} // namespace myfft
