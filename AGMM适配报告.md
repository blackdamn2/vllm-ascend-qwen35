# AGMM 在 6156 / A3 上的适配与验证

日期：2026-11-24。目标：Qwen3.5-122B-A10B-W8A8，先在 A3 适配用户给定的 agmm\_0.32.0rc1.patch，为后续 A5 实测提供代码与证据。

## 结果

A3 完整服务已跑通，四个 rank 主模型各改写 36 处 GDN AG＋GEMM，MTP 图各改写 1 处。保留 FULL\_DECODE\_ONLY、MTP3 与所有原启动参数。两轮融合版各 40 条请求均成功，图捕获和投机执行无报错。

本轮 TTFT 均值下降 5.84%，ITL 均值上升 1.36%。这支持将其保留为 prefill 优化候选，不能作为 decode / TPOT 优化通过。样本为 A1→B1→B2→A2；尚不是多轮交错实验或统计显著性证明。原 patch 的策略本身只在大 token eager 路径使用融合，decode/capture 保留原 collective；当前 V2 的大 prefill 包括 MTP 图也可进入融合。

运行服务最终恢复原版；适配版以独立 patch、源码和启动脚本保存。之前没有稳定收益的 SwiGLU 与 out\_proj 实验已单独保存、记录并回退，见本地产物 saved-decode-changes/修改记录与回退.md。

## 固定配置

TP4、EP、VLLM\_USE\_V2\_MODEL\_RUNNER=1；unset OMP\_PROC\_BIND；max-model-len=131072、max-num-batched-tokens=48384、max-num-seqs=32、gpu-memory-utilization=0.94；prefix cache、async scheduling、MTP3、FULL\_DECODE\_ONLY；capture sizes 为 1,2,3,4,5,6,7,8,16,20,24,32,48,64,96,128,192。

原有 additional-config 保持：enable\_fused\_mc2=0、enable\_prefill\_mc2=false、multistream\_overlap\_shared\_expert=true、enable\_flashcomm1=true、gdn\_internal\_prefill\_checkpoints=true、enable\_shared\_expert\_dp=true。原启动脚本为 start-original.sh，适配版为 start-agmm.sh。仅使用 AGMM 开关、代码覆盖路径与独立编译缓存加载新代码；未通过改模型、TP/EP、图、MTP、调度或长度参数绕过问题。

## 核心改动

1. 通信参数使用 HCCL communicator 字符串，BF16 权重明确按 N×K→K×N 转置；不再把 ProcessGroup 直接传给原生算子。
2. 不在 decoder 中根据 forward 阶段删除 AllGather。BF16 opaque op 固定返回完整 padded 投影和完整 BF16 gathered 输入，GDN 的 in\_proj\_ba 分支继续使用同一 gathered 输入，原裁剪节点保留。
3. Fake 实现只依赖输入 tensor metadata，不访问 ForwardContext、layer registry、HCCL 或设备状态。val、example\_value、tensor\_meta 与完整／裁剪输出保持一致。
4. 改写接入实际 npugraph\_ex 编译入口，兼容 Dynamo 的 OpOverloadPacket / example\_value 与 make\_fx 的 default overload / val。只匹配相同通信组、维度、dtype、layout 和 stream 的合法链。
5. MXFP8 保留 K×N 权重、E8M0 scale、group\_sizes=\[1,1,32]。接口缺失时执行原量化路径，绝不把 FP8 权重送给 BF16 GEMM；scale dtype 不明确则不融合。其他分支仍需要精确 BF16 gathered 输入时，MXFP8 链保持原图。
6. AGMM 源码 hash 进入后端编译键；启动脚本另以源码 hash 隔离 AOT 缓存根目录，防止旧图命中。缓存保护和 A5 的严格 scale guard 在完整性能测试后补充，已通过相应单测；同一计算实现还接受了真实权重逐层验证。

## 性能

负载：输入 10750，输出 750，总长 11500；共享前缀 4608，request rate=1，并发=8，每轮 40 条。每轮重置 prefix cache，测试期间无并行 NPU 微基准。以下 ITL 为 AIPerf inter\_token\_latency，不与 A5 的 step 时间直接换算。

|轮次|ITL ms|TTFT ms|完成 / 错误|平均接受长度（含目标 token）|
|-|-:|-:|-:|-:|
|baseline-A1|13.3642|834.44|40 / 0|3.8827|
|candidate-B1|13.6906|786.86|40 / 0|3.8892|
|candidate-B2|13.6484|784.85|40 / 0|3.8829|
|restored-A2|13.6070|834.71|40 / 0|3.8907|

原版均值 ITL=13.4856 ms，融合版=13.6695 ms。原版 TTFT=834.57 ms，融合版=785.85 ms。四轮的每次 draft token 数均为 3，MTP 未被关闭。

## 精度与功能证据

* BF16 primitive 四卡依次通过 eager、compile-eager、aot\_eager、npugraph\_ex force-eager、npugraph\_ex 五层验证。覆盖非 TP 整除 token、主投影／BA 分支、capture 后更新输入再 replay。
* K3072 / N5120 对照 CPU FP64 的 12 组测试：最小匹配比例 1.0，最大绝对误差 0.0156236622，最大 NRMSE 0.0016855187；gathered 输入逐位相同。详见 validation-summary.json 和 precision-real-shapes.log。
* 完整模型的 16 条基础输出与原版一致。8 条 10800-token 长输入问题全部正确，生成文本及 token 序列均与原版一致；恢复原版后再重复一轮，输出也正确。
* 对低概率 top-5 logprob 的差异未直接判定为 AGMM 问题：AGMM 对原版的最大共同候选 logprob 差为 3.06249，原版自身重复为 2.74894。换算成最大共同候选概率差分别为 0.000485885 和 0.00105972。此为 top-5 交集检查，不代表全词表误差，也不是整模型精度数据集验收。
* 为进一步隔离误差，使用独立 precision-overlay，在同一次真实长输入 forward 中逐层同时执行融合及原始 AG＋F.linear。四卡共 148 次（每 rank 主模型36＋MTP prefill 1）真实权重／真实激活比较全部通过：AllGather 输入逐位一致，最小匹配率 1.000000000，最大绝对误差 0.125，最大 NRMSE 4.80810522e-05。门限：BF16 atol=rtol=2^-6，匹配率≥99%，max\_abs≤1，NRMSE<0.004，且输出均有限。验证逻辑不包含在生产补丁，也未参与性能计时。

已测范围未发现超出门限的算子精度问题。A5 MXFP8 仅完成 Fake、接口参数与 fallback 契约测试，尚无 A5 硬件精度或性能结论；不能把 A3 BF16 通过写成 A5 MXFP8 已验证。

## 失败与经验

|现象|处理与结论|
|-|-|
|原 patch 的 linear\_op / decoder hunk 不适配当前分支|不强覆盖，改在现有编译图上匹配实际 AG＋GEMM|
|原 helper eager 报 ProcessGroup / str 类型错误|先修 HCCL 接口；不是有证据的 A5 图根因|
|首版测试缺当前 VllmConfig|修测试 fixture，原业务配置不动|
|完整 padded 输出带着裁剪后的 tensor\_meta|分别生成 val、example\_value、tensor\_meta，并增加一致性断言|
|服务缺 acl 模块，原版重启也失败|让编排进程继承 CANN set\_env.sh；属于环境问题|
|服务健康但没有改写日志|检查真实编译入口，不能把健康状态当作融合生效|
|实际图改写数为 0|补齐 packet / default 及 example\_value / val 表示差异，新增回归|
|低概率 logprob 有变化|补跑原版重复，另做同输入同权重逐层比较，不凭文本一致跳过精度验证|

详细过程在 attempts.md；原始错误、成功五层日志与完整服务日志保存在证据压缩包中。不能据此声称已确认用户在 A5 遇到的原始错误根因。

## 使用与回退

agmm-shape-safe.patch 针对当前 6156 部署分支生成，并已通过 git apply --check。全新目录重建 overlay 的源码 manifest 也与已验证版本一致。不要与用户原始 patch 叠加应用。先在目标 checkout 检查 patch；保持原服务参数，只加载新代码、启用 VLLM\_ASCEND\_ENABLE\_AGMM=1，并使用随源码版本隔离的缓存。A5 需先进行硬件精度验收再测试性能。

当前 6156 原版源码和 /work/start-qwen35.sh 的 hash 与改动前一致；API argv 与快照一致，服务 health=200。恢复时只处理本实验容器的 8156 服务进程，没有停止其他业务容器。

独立影子验证目录仅用于验证；生产补丁不含额外 GEMM、比较、文件开关或同步。

