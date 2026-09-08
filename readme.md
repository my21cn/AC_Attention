# AC_Attention —— 解析延拓注意力（Analytic-Continuation Attention）

用**解析延拓（锚点 jet 展开 / 矩层级）对 Attention 做降维变换**，把 K、V 压缩为层次聚类树上
每个簇的矩喷流（moment jets），使单查询复杂度从 $O(nd)$ 降至 $O((C(d,P)+d)\,d_{\rm out}\log n)$，
总复杂度 $O(n(C(d,P)+d)\,d_{\rm out}\log n)$ —— 在序列长度 $n$ 上**接近 $O(\log n)$**（单查询摊还）。

> 作者：**魏勇勤** ｜ 时间：**2026年8月30日**
> 完整数学推导（定理、证明、误差界）：见 <code>AC_Attention.html</code>
> 参考实现：<code>ac_attention.py</code>（numpy，正确性优先）

---

## 一、核心思想（三句话）

1. **复化与解析延拓**：把注意力嵌入整函数族 $A(z)=D(z)^{-1}e^{zQK^\top/\sqrt d}V$，
   物理注意力是 $A(1)$；整函数由锚点处的 jet（矩）唯一确定。
2. **降维变换 + K/V 压缩**：每个簇 $C$（锚点 $c_C$，$\delta=k-c_C$）内
   $$e^{\langle q,k\rangle/\sqrt d}=e^{\langle q,c_C\rangle/\sqrt d}\sum_{|\alpha|\le P}
   \frac{q^{\alpha}\,\delta^{\alpha}}{\alpha!\,d^{|\alpha|/2}}+R_{P+1},\qquad
   |R_{P+1}|\ \le\ e^{u}\,\frac{u^{P+1}}{(P+1)!},\quad u=\frac{\|q\|\,r_C}{\sqrt d}.$$
   交互由 $C(d,P)=\binom{d+P}{P}$ 个矩完全决定：
   **K 压缩**为标量矩 $m_\alpha(C)=\sum_{j\in C}\delta_j^\alpha$，
   **V 压缩**为向量矩 $M_\alpha^V(C)=\sum_{j\in C}\delta_j^\alpha v_j$。
3. **层次分解**：2-means 方向 + 中位分裂建树（深度 $L=\lceil\log_2(n/s)\rceil$），
   每个查询沿最近质心下行，逐层取**兄弟块**做远场喷流近似、叶子精确计算 —— 每查询恰好 $L$ 个远场块。

误差按 $1/(P+1)!$ 几何衰减；控制参数 $u=\|q\|\,r_C/\sqrt d$。单位范数 $Q,K$ 时
$u^*\le 2/\sqrt d<1$ 自动成立（$d>4$），$P=3$ 即达 $\sim10^{-6}$ 相对误差。

---

## 二、安装

<pre>pip install numpy</pre>

仅依赖 numpy（参考实现用 2.x 开发，1.24+ 亦可）。把 <code>ac_attention.py</code> 放到项目目录即可。

## 三、快速开始（调用方法）

<pre>
import numpy as np
from ac_attention import ac_attention, exact_attention

rng = np.random.default_rng(42)
n, d = 2048, 16
Q = rng.normal(size=(n, d)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
K = rng.normal(size=(n, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
V = rng.normal(size=(n, d))

# 调用：与标准注意力同接口（Q,K 记得做归一化，见"参数建议"）
out = ac_attention(Q, K, V, P=3, leaf_size=8, eps=1e-3)

# 对照：独立蛮力实现（仅用于验证，O(n^2)）
ref = exact_attention(Q, K, V)
err = np.linalg.norm(out - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-9)
print("mean rel err =", err.mean(), " max rel err =", err.max())
# 典型输出（d=16, P=3, 单位范数）：mean rel err ≈ 3.4e-6
</pre>

自检（内置演示）：

<pre>python ac_attention.py</pre>

## 四、API 参考

| 函数 | 签名 | 说明 |
|---|---|---|
| <code>ac_attention</code> | <code>ac_attention(Q, K, V, P=3, leaf_size=8, eps=1e-3, seed=0, kmeans_iters=4, timing=False)</code> | AC_Attention 前向。<code>Q:(n,d), K:(n,d), V:(n,d_out)</code>；<code>timing=True</code> 时返回 <code>(out, t_build, t_query)</code> |
| <code>exact_attention</code> | <code>exact_attention(Q, K, V)</code> | 独立 $O(n^2)$ 蛮力参考（验证/对照用） |
| <code>build_tree</code> / <code>compute_jets</code> | — | 层次聚类树 / 矩喷流压缩（可离线预计算、增量更新 KV 缓存） |

**参数建议**

| 参数 | 默认 | 建议 |
|---|---|---|
| <code>P</code> | 3 | 截断阶上限。$d\le32$ 取 3；$d>32$ 取 2（$C(d,P)=\binom{d+P}{P}$ 随 $d^P$ 增长） |
| <code>eps</code> | 1e-3 | 每个远场块的相对误差预算；每个簇按 $u_C^{p+1}/(p+1)!\le\varepsilon$ 自适应降阶 |
| <code>leaf_size</code> | 8 | 叶子精确计算的 token 数（常数，影响常数因子） |
| <code>seed</code> / <code>kmeans_iters</code> | 0 / 4 | 聚类随机性 / 2-means 迭代数 |

## 五、复杂度（$d, d_{\rm out}, P$ 固定）

| 指标 | AC_Attention | 标准 Attention |
|---|---|---|
| 单查询 | $O((C(d,P)+d)\,d_{\rm out}\log n)$ | $O(nd)$ |
| 总时间 | $O(n(C(d,P)+d)\,d_{\rm out}\log n)$ | $O(n^2d)$ |
| K/V 压缩存储 | $O(n\,C(d,P)\,d_{\rm out}\log n)$ | 无压缩 |

$C(d,P)$：$d{=}16$ 时 $17\,(P{=}1)/153\,(P{=}2)/969\,(P{=}3)$；$d{=}64$ 时 $65\,(P{=}1)/2145\,(P{=}2)$。

## 六、实测精度（numpy 参考实现，2026-08-30）

逐行相对误差 $\|\hat o_i-o_i\|/(\|o_i\|+10^{-9})$，与 <code>exact_attention</code> 差分测试：

| 场景 | 误差（mean / max） |
|---|---|
| $n{=}256..2048$，$d{=}16$，单位范数，$P{=}3$ | $\approx3\times10^{-6}\;/\;7\times10^{-6}$ |
| $n{=}1024$，$d{=}64$，单位范数，$P{=}2$ | $2.4\times10^{-6}\;/\;3.6\times10^{-6}$ |
| 10 个紧致簇，$P{=}4$ | $9.5\times10^{-6}\;/\;5.5\times10^{-5}$ |
| scale=2（$Q,K$ 同乘 2） | $8.8\times10^{-4}\;/\;1.8\times10^{-3}$ |
| scale=4（$B\approx16$，退化域） | $1.0\times10^{-1}\;/\;2.7\times10^{-1}$ |

锚点 jet 恒等式单簇验证：$P=0..4$ 相对误差 $2.8\times10^{-2}\to9.0\times10^{-4}\to2.4\times10^{-5}\to3.1\times10^{-7}\to1.2\times10^{-8}$（几何衰减 ✓）。

实测计时（$d{=}16,P{=}3$）：查询阶段 $n$ 从 512 增至 16384 时 0.10s→1.19s（≈$n^{0.9..1.1}$，与 $O(n\log n)$ 一致），精确实现 0.003s→2.75s（≈$n^2$）；**查询阶段在 $n\approx2\times10^4$ 处反超**（16384 时已快 2.3×），外推 $n{=}6.5\times10^4$ 约 7–8×。参考实现常数来自 numpy 逐块开销，GPU/融合实现预计再降 2–3 个数量级。

## 七、适用域与局限（务必阅读）

- **有利域**：归一化的 $Q,K$（LayerNorm/RoPE 之后）：$u^*\le2/\sqrt d\ll1$，$P=2\sim3$ 即 $\sim10^{-6}$。
- **退化域**：对数幅度 $B\gg1$ 时需 $P\approx eB$（误差界 $u^{P+1}/(P+1)!$ 的推论）；实测 scale=4 时 $P=3$ 误差升到 $10^{-1}$。
- 误差界依赖两条显式假设（有界 logit、层次半径收缩）；不满足时算法照常运行但不保证 $\varepsilon$。
- "接近 $O(\log n)$"指单查询在 $n$ 上的摊还阶数；常数含 $C(d,P)\,d_{\rm out}$。

## 八、文件清单

| 文件 | 内容 |
|---|---|
| <code>AC_Attention.html</code> | 完整数学推导：复化族、锚点 jet、主定理、证明、实验（作者：魏勇勤，2026年8月30日） |
| <code>对比.html</code> | 与标准/多头/线性/稀疏注意力的三轴对比：复杂度、KV 压缩率、精度（同基准实测，作者：魏勇勤，2026年8月30日） |
| <code>readme.md</code> | 本文件：调用方法与使用说明 |
| <code>ac_attention.py</code> | 参考实现（numpy）：<code>ac_attention</code> / <code>exact_attention</code> / 树与喷流工具 |

## 九、引用

> 魏勇勤. AC_Attention：基于解析延拓降维变换的注意力近似算法. 2026年8月30日.

技术谱系：FMM 多极展开（Greengard–Rokhlin 1987）、H-矩阵（Hackbusch 1999）、线性注意力（Katharopoulos 2020）、Performer（Choromanski 2021）、聚类注意力（Vyas 2020）；详见 HTML 参考文献。
