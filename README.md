# myfft — 自研 GPU FFT 库

模仿 cuFFT 架构(plan-based、单一动态库、纯 C API)的 FFT 库自研项目。
当前阶段 **M1**:CPU 黄金参考后端已跑通;GPU 后端代码已就位
(`MYFFT_ENABLE_CUDA=ON`),待内网 4090 服务器编译验证。

## 目录结构

```
myfft/
├── include/myfft.h        # 唯一公共头文件(纯 C API)
├── src/
│   ├── api.cpp            #   1. C API:handle 生命周期、参数校验、workspace 分配/释放
│   ├── plan.{h,cpp->graph.cpp}
│   │                      #   2. 规划:参数校验、路径选择、graph 生成
│   ├── heuristics.*       #      路径选择规则表(M3 加混合基/Bluestein)
│   ├── graph.cpp          #   3. IR:plan 编译成操作节点序列
│   ├── executor.*         #   5. 执行器:遍历 graph 派发;CPU/GPU 双后端
│   ├── cpu/cpu_fft.*      #   CPU 黄金参考(radix-2,双精度,永远保留)
│   ├── gpu/gpu_fft.{h,cu} #   6. GPU 后端:发射注册表第一版 + host↔device 搬运(M1)
│   └── kernels/           #   4. CUDA kernel(radix2.cuh 已随 M1 接入编译)
│       ├── radix2.cuh     #      块内 shared-mem radix-2(模板化)
│       ├── pw.cuh         #      逐点复数乘(Bluestein 部件)
│       └── registry.cuh   #      kernel 注册表设想(M2 扩展为携带 launch 属性)
├── tests/                 # 正确性:朴素 DFT 对拍 + round-trip(零依赖)
├── bench/                 # 性能:尺寸扫描表(CPU 基线)
├── examples/              # 用户视角最小示例
└── docs/ROADMAP.md        # 开发计划(里程碑与验收标准)
```

## 构建与运行(Windows + VS2022;Linux 同理)

```bash
cmake -B build                                # 或 -G "Visual Studio 17 2022"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure   # 正确性测试
./build/bin/Release/bench_1d.exe                       # 性能扫描
./build/bin/Release/example_1d.exe
```

M1 起开启 CUDA 后端(需 CUDA Toolkit;Linux/4090 同理):

```bash
cmake -B build -DMYFFT_ENABLE_CUDA=ON
cmake --build build -j                                    # Linux
ctest --test-dir build --output-on-failure                # correctness(GPU) + correctness_cpu
```

- **后端选择**:默认 GPU;环境变量 `MYFFT_BACKEND=cpu` 强制走 CPU 黄金参考
  (同一二进制双后端对拍)。
- **多架构 fatbin**:默认编译 75/86/89/90(GTX1650/3090/4090/H100),
  可用 `-DCMAKE_CUDA_ARCHITECTURES=89` 只编本机。
- **M1 限制**:GPU 单 kernel 覆盖 2..1024 点,更大尺寸自动回退 CPU
  (M2 以多 kernel 分解解除)。

## 设计决策(为什么长这样)

- **单一动态库 + 版本化 soname**:产物只有一个 `myfft.dll` / `libmyfft.so.1`,
  ABI 变更 = 升主版本号。内部符号默认隐藏,只导出 `myfft_*` C 函数。
- **Plan-based 模型**:创建贵(规划)、执行便宜(查表发射)。所有算法演进
  只发生在 heuristics/graph/executor 内部,公共 API 永不改变形态。
- **graph IR**:plan 被编译成操作节点序列,executor 只认节点不认算法。
  M0 只有一个 Fft1d 节点,但链路(Create→Plan→Exec)已完整走通。
- **CPU 黄金参考永远保留**:GPU 实现的一切正确性结论都来自与它对拍。

状态与计划详见 [docs/ROADMAP.md](docs/ROADMAP.md)。
