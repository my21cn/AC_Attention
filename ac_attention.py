# -*- coding: utf-8 -*-
"""
AC_Attention —— Analytic-Continuation Attention（解析延拓注意力）
==============================================================

用解析延拓（锚点 jet 展开 / 矩层级）对 Attention 做降维变换：
把 K、V 压缩为层次聚类树上每个簇的矩张量（moment jets），
使单查询的计算复杂度从 O(n·d) 降至 O((C(d,P)+d)·d_out·log n)，
总复杂度 O(n·(C(d,P)+d)·d_out·log n)，在 n 上接近 O(log n)（单查询摊还）。

作者：魏勇勤    时间：2026年8月30日

数学推导详见 AC_Attention.html；调用方法详见 readme.md。

核心恒等式（|alpha| <= P 截断）：

    e^{<q,k>/sqrt(d)} = e^{<q,c>/sqrt(d)} * sum_alpha q^alpha * delta^alpha
                                                   / (alpha! * d^{|alpha|/2}) + R_{P+1}

其中 c 是簇锚点、delta = k - c。余项 |R_{P+1}| <= e^u * u^{P+1}/(P+1)!，
u = ||q|| * r_C / sqrt(d)。当 u < 1（单位范数 Q,K 的典型情形，u <= 2/sqrt(d)）时
误差按 1/(P+1)! 几何衰减；本实现按每个簇的 u 自适应选择截断阶 <= P（自适应解析延拓）。
"""

from __future__ import annotations

import math
import time

import numpy as np


# ---------------------------------------------------------------- 矩单项索引

def monomials(d: int, P: int):
    """所有满足 |alpha| <= P 的 d 维多单项 alpha（按次数升序），
    系数 1/(alpha! * d^{|alpha|/2})，各 alpha 的次数，以及每个次数的下标切片。

    递归按剩余次数预算生成，只产出 C(d+P, P) 个有效项（不枚举 (P+1)^d 全空间）。
    """
    alphas = []
    prefix = []

    def gen(dims_left: int, budget: int):
        if dims_left == 0:
            alphas.append(tuple(prefix))
            return
        for v in range(budget + 1):
            prefix.append(v)
            gen(dims_left - 1, budget - v)
            prefix.pop()

    gen(d, P)
    order = sorted(range(len(alphas)), key=lambda j: sum(alphas[j]))
    alphas = [alphas[j] for j in order]
    coef = np.empty(len(alphas))
    for j, a in enumerate(alphas):
        deg = sum(a)
        fac = 1.0
        for ai in a:
            fac *= math.factorial(ai)
        coef[j] = 1.0 / (fac * (d ** (deg / 2.0)))
    degs = np.array([sum(a) for a in alphas])
    slices = [np.nonzero(degs == p)[0] for p in range(P + 1)]
    return alphas, coef, degs, slices


def monomial_features(X: np.ndarray, alphas: list, chunk: int = 64) -> np.ndarray:
    """X: (n, d) -> (n, len(alphas))，逐行计算 x^alpha = prod_r x_r^{alpha_r}。

    分块向量化：np.power(X[:,None,:], E[None]) 一次算一块单项式，再沿 d 求积。
    """
    n = X.shape[0]
    if len(alphas) == 0 or n == 0:
        return np.empty((n, len(alphas)))
    E = np.asarray(alphas, dtype=np.int32)              # (C, d) 指数张量
    F = np.empty((n, len(alphas)))
    for s in range(0, len(alphas), chunk):
        Ee = E[s:s + chunk]
        F[:, s:s + chunk] = np.prod(
            np.power(X[:, None, :], Ee[None, :, :]), axis=2)
    return F


def _needed_order(u: float, eps: float, P_max: int) -> int:
    """给 u = ||q||*r_C/sqrt(d) 与相对误差预算 eps，选最小截断阶：
    相对余项 u^{p+1}/(p+1)! <= eps（e^u 因子在相对误差中约去）。"""
    p = 0
    term = u                             # p=0 时的相对余项 u^1 / 1!
    while term > eps and p < P_max:
        p += 1
        term *= u / (p + 1)
    return p


# ---------------------------------------------------------------- 层次聚类树（2-means 方向 + 中位分裂）

def _split(indices, K, rng, leaf_size, iters):
    indices = np.asarray(indices)
    node = {"idx": indices}
    if len(indices) <= leaf_size:
        return node
    X = K[indices]
    # 2-means 初始化：随机点 + 距其最远点
    s0 = X[int(rng.integers(len(X)))]
    s1 = X[int(np.argmax(np.sum((X - s0) ** 2, axis=1)))]
    c0, c1 = s0.copy(), s1.copy()
    for _ in range(iters):
        d0 = np.sum((X - c0) ** 2, axis=1)
        d1 = np.sum((X - c1) ** 2, axis=1)
        lab = d0 <= d1
        if lab.all() or not lab.any():
            break
        c0 = X[lab].mean(axis=0)
        c1 = X[~lab].mean(axis=0)
    # 沿 2-means 分离方向做中位分裂：既保持几何分离、又保证树平衡（深度 <= log2(n/s)）
    direction = c1 - c0
    if float(np.dot(direction, direction)) < 1e-12:
        direction = rng.normal(size=K.shape[1])
    proj = X @ direction
    order = np.argsort(proj)
    half = len(indices) // 2
    node["left"] = _split(indices[order[:half]], K, rng, leaf_size, iters)
    node["right"] = _split(indices[order[half:]], K, rng, leaf_size, iters)
    return node


def build_tree(K: np.ndarray, leaf_size: int = 8, seed: int = 0, kmeans_iters: int = 4):
    """在 K 上建 2-means 方向 + 中位分裂的二叉树。返回 (tree, centroid, radius)。"""
    rng = np.random.default_rng(seed)
    tree = _split(list(range(K.shape[0])), K, rng, leaf_size, kmeans_iters)
    centroid, radius = {}, {}

    def fill(node):
        c = K[node["idx"]].mean(axis=0)
        centroid[id(node)] = c
        radius[id(node)] = float(np.max(np.linalg.norm(K[node["idx"]] - c, axis=1)))
        if "left" in node:
            fill(node["left"])
            fill(node["right"])

    fill(tree)
    return tree, centroid, radius


# ---------------------------------------------------------------- 簇矩喷流（K/V 压缩）

def compute_jets(tree, K, V, alphas, coef, slices, P, eps, qmax, inv_sqrt_d):
    """为每个簇计算矩喷流（截断阶 Pn 按 u = qmax*r_C/sqrt(d) 自适应，<= P）：

    M_alpha = sum_j delta_j^alpha * v_j（V 压缩），m_alpha = sum_j delta_j^alpha
    （K 压缩，用于 softmax 归一化项）。jets[node_id] = (质心 c, M, m, Pn)。
    """
    jets = {}

    def rec(node):
        idx = node["idx"]
        c = K[idx].mean(axis=0)
        r = float(np.max(np.linalg.norm(K[idx] - c, axis=1)))
        Pn = _needed_order(qmax * r * inv_sqrt_d, eps, P)
        keep = np.concatenate([slices[p] for p in range(Pn + 1)])
        delta = K[idx] - c
        F = monomial_features(delta, [alphas[j] for j in keep])   # (n_c, C(d,Pn))
        jets[id(node)] = (c, F.T @ V[idx], F.sum(axis=0), Pn)
        if "left" in node:
            rec(node["left"])
            rec(node["right"])

    rec(tree)
    return jets


# ---------------------------------------------------------------- 主算法（查询阶段按簇批量向量化）

def ac_attention(Q, K, V, P=3, leaf_size=8, eps=1e-3, seed=0, kmeans_iters=4, timing=False):
    """AC_Attention 前向（自适应解析延拓截断 + 层次 K/V 压缩）。

    参数
    ----
    Q, K, V : (n, d) / (n, d) / (n, d_out)
    P       : 截断阶上限（每个簇按 u = ||q||*r_C/sqrt(d) 自适应选择 <= P 的阶数）
    leaf_size : 叶子精确计算的最大 token 数（常数）
    eps     : 每个远场块的相对误差预算（余项 e^u u^{p+1}/(p+1)! <= eps）
    seed    : 聚类随机种子；kmeans_iters : 2-means 迭代数

    返回 out 形状同 Q；timing=True 时返回 (out, t_build, t_query)。
    复杂度（P, d, d_out 固定）：单查询 O((C(d,P)+d)*d_out*log n)，
    总 O(n*(C(d,P)+d)*d_out*log n)；C(d,P) = C(d+P, P)。
    """
    n, d = Q.shape
    d_out = V.shape[1]
    inv_sqrt_d = 1.0 / math.sqrt(d)
    qmax = float(np.max(np.linalg.norm(Q, axis=1)))
    eps_block = eps       # 每个远场块的相对误差预算（总误差上界为 L*eps，softmax 加权后实测远小）

    t0 = time.perf_counter()
    tree, centroid, radius = build_tree(K, leaf_size, seed, kmeans_iters)
    # 系数与 q 单项式特征（全局排序，按次数升序；阶数前缀索引即次数 <= Pn 的全部项）
    alphas, coef, _, slices = monomials(d, P)
    jets = compute_jets(tree, K, V, alphas, coef, slices, P, eps_block, qmax, inv_sqrt_d)
    qf_full = monomial_features(Q, alphas)                # O(n*C(d,P)*d)
    t1 = time.perf_counter()

    # ---- 查询路由：沿最近质心下行，逐层收集 (远场块节点, 查询集合) 对 ----
    blocks = []                    # (块节点对象, 查询下标数组)
    leaf_sq = []                   # (叶子节点对象, 查询下标数组)
    stack = [(tree, np.arange(n))]
    while stack:
        node, sq = stack.pop()
        if "left" not in node:
            leaf_sq.append((node, sq))
            continue
        cL = centroid[id(node["left"])]
        cR = centroid[id(node["right"])]
        goL = Q[sq] @ cL >= Q[sq] @ cR
        sL, sR = sq[goL], sq[~goL]
        blocks.append((node["right"], sL))
        blocks.append((node["left"], sR))
        stack.append((node["left"], sL))
        stack.append((node["right"], sR))

    # ---- 数值稳定位移：叶子精确 logits 与各远场块锚点 logits 的逐查询最大值 ----
    shift = np.full(n, -np.inf)
    leaf_cache = []
    for leaf, sq in leaf_sq:
        S = Q[sq] @ K[leaf["idx"]].T * inv_sqrt_d
        shift[sq] = np.maximum(shift[sq], S.max(axis=1))
        leaf_cache.append((sq, leaf["idx"], S))
    for blk, sq in blocks:
        anchor = (Q[sq] @ centroid[id(blk)]) * inv_sqrt_d
        shift[sq] = np.maximum(shift[sq], anchor)

    # ---- 远场块：锚点 jet（降维变换后的压缩交互） ----
    num = np.zeros((n, d_out))
    den = np.zeros(n)
    for blk, sq in blocks:
        _, M, m, Pn = jets[id(blk)]
        end = math.comb(d + Pn, Pn)                       # 次数 <= Pn 的单项式个数
        w = np.exp((Q[sq] @ centroid[id(blk)]) * inv_sqrt_d - shift[sq])
        coeffs = w[:, None] * coef[:end][None, :] * qf_full[sq][:, :end]
        num[sq] += coeffs @ M[:end]
        den[sq] += coeffs @ m[:end]

    # ---- 叶子：精确 softmax（常数小块） ----
    for sq, ltokens, S in leaf_cache:
        lw = np.exp(S - shift[sq][:, None])
        num[sq] += lw @ V[ltokens]
        den[sq] += lw.sum(axis=1)

    out = num / den[:, None]
    t2 = time.perf_counter()
    return (out, t1 - t0, t2 - t1) if timing else out


def exact_attention(Q, K, V):
    """独立蛮力参考实现（O(n^2)，仅用于验证与对照）。"""
    S = Q @ K.T / math.sqrt(K.shape[1])
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return (E @ V) / E.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- 自检 / 演示

def _rel_row_err(A, B):
    return np.linalg.norm(A - B, axis=1) / (np.linalg.norm(B, axis=1) + 1e-9)


def _demo():
    print("AC_Attention 自检：解析延拓降维 + K/V 压缩（作者：魏勇勤 2026-08-30）\n")
    rng = np.random.default_rng(42)
    for n, d, scale in [(256, 16, 1.0), (1024, 16, 1.0), (2048, 16, 1.0), (1024, 16, 4.0)]:
        Q = rng.normal(size=(n, d)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        K = rng.normal(size=(n, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
        V = rng.normal(size=(n, d))
        Qs, Ks = Q * scale, K * scale
        t0 = time.perf_counter()
        ac = ac_attention(Qs, Ks, V, P=3, leaf_size=8)
        t1 = time.perf_counter()
        ex = exact_attention(Qs, Ks, V)
        t2 = time.perf_counter()
        err = _rel_row_err(ac, ex)
        print(f"n={n:5d}  d={d:2d}  scale={scale:.1f}  "
              f"mean_rel_err={err.mean():.3e}  max_rel_err={err.max():.3e}  "
              f"t_AC={t1-t0:6.3f}s  t_exact={t2-t1:6.3f}s")
    print("\n提示：完整推导见 AC_Attention.html；调用方法见 readme.md。")


if __name__ == "__main__":
    _demo()
