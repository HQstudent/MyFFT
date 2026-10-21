# myfft 开发计划(ROADMAP)

验收口径统一为两条:
1. **正确性**:对拍朴素 DFT / CPU 黄金参考,相对 L2 误差 FP32 < 1e-5;
2. **性能**:与同机 cuFFT 同尺寸对比,以及 Nsight Compute 的 DRAM 带宽利用率
   (健康值 60%~90%,这是 FFT 库唯一重要的硬件指标)。

---

## M0 工程骨架 + CPU 黄金参考 ✅(已完成)

- [x] 单一 C API(myfft.h)与 plan-based 生命周期跑通
- [x] CPU radix-2 参考实现(双精度内部)
- [x] 测试设施:朴素 DFT 对拍 / round-trip / batch / in-place / 参数校验
- [x] bench 尺寸扫描表(CPU 基线),examples
- [x] CUDA kernel 文件就位(radix2.cuh / pw.cuh),待 M2 编译
- 验收:ctest 全绿(本机 VS2022)

## M1 CUDA 正确性骨架(约 1~2 周;前置:安装 CUDA Toolkit)

> 2026-10-21:M1 代码已全部就位——CMake 接入(多架构 fatbin)、kernel 发射注册表
> 第一版、executor 双后端(MYFFT_BACKEND 可切)、workspace 内部化(plan 时分配)、
> 双后端测试(ctest correctness / correctness_cpu)。本机无 CUDA Toolkit,
> 待内网 4090 服务器编译验证后勾选。M1 限制:GPU 单 kernel 覆盖 2..1024 点,
> 更大尺寸自动回退 CPU(M2 以多 kernel 分解解除)。

- [ ] 安装 CUDA Toolkit(含 VS2022 集成);`MYFFT_ENABLE_CUDA=ON` 跑通编译
- [ ] 朴素 GPU 版:先跑 `radix2.cuh` 的 block kernel(每 block 一条变换)
- [ ] executor 双后端:CPU/GPU 由 CMake 选项切换,API 不变
- [ ] GPU vs CPU 对拍进测试(同一套 test_correctness 跑两种后端)
- [ ] workspace 管理 API 内部化(plan 时分配,destroy 时释放)
- 验收:GPU 后端全尺寸测试绿;`dumpbin /exports myfft.dll` 只有 myfft_* 导出

## M2 2 的幂性能冲刺(约 1~2 月)

- [ ] kernel 注册表启用:LOG2N 1~10 全实例化,heuristics 查表选
- [ ] Stockham 自排序算法替换显式位反转
- [ ] warp shuffle butterfly、寄存器驻留小尺寸、float4 向量化 IO
- [ ] 多变换/block 提高小尺寸批量的占用率;blockDim 扫描调优
- [ ] bench 加"vs cuFFT"列;ncu 剖面报告入 docs/
- 验收:常用尺寸(256~65536,batch 场景)达到 cuFFT 80%+,带宽利用率 > 60%

## M3 混合基 + Bluestein(约 1~2 月)

- [ ] radix-3/5/7(11 可选)模板 kernel,heuristics 做尺寸分解
- [ ] Bluestein 路径:chirp 预计算(plan 时)+ PwMulConj + 复用 M2 FFT
- [ ] 大素数、2^k*3^m*5^n 等全尺寸对拍
- 验收:任意 N ≤ 10^6 正确;混合基尺寸性能 ≥ cuFFT 70%

## M4 R2C/C2R、batch 布局、2D/3D(约 2 月)

- [ ] R2C/C2R(奇偶打包,省一半带宽)与 myfftExecR2C/C2R API
- [ ] MakePlanMany(带 stride/embed)
- [ ] 2D/3D:逐维 1D + transpose kernel
- 验收:2D 4096² 达 cuFFT 70%;R2C 与等价 C2C 一致性测试

## M5 大尺寸分解路径 + FP64(约 2 月)

- [ ] 多 kernel 分解 + 全局转置(corner turn)处理 2^20+
- [ ] FP64 全路径
- [ ] (可选差异化)CUTLASS:小尺寸 DFT 写成 batched GEMM 跑 tensor core
- 验收:2^24 单变换不 OOM 且性能 ≥ cuFFT 60%;FP64 全测试绿

## M6 生态(时间盒,按需)

- [ ] half/bf16、多 GPU、Python 绑定、plan cache 与 CUDA Graph 兼容

---

## 工作节奏建议

- 每个里程碑一个 git 分支,合入前 ctest 必须全绿(CPU+GPU 两后端)。
- M2 起建立每晚全尺寸回归(正确性 + 性能表入库,防止性能回退)。
- 每完成一档尺寸路径,在 docs/ 补一份 ncu 剖面结论,累计成调优档案。
