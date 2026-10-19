#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两阶段检索 + 五道闸门 + 发散度（§3.2、§15、§16 的实现）。

**这个函数是纯的**：`retrieve(snapshot, query, d, params) -> RetrieveResult`。
不写盘、不改图、不用随机数 ⇒ 同快照 + 同参数 = **逐位相同**（§16.5 的验收指标）。
要落盘（`access_count`、Hebbian 强化）请另外调 `lifecycle.commit_access()`。

阶段一（种子）：`seed = ω₁·ẑ(相关) + ω₂·ẑ(重要性) + ω₃·ẑ(S_now) + ω₄·recency`，取 top-K。
   ⚠️ `ẑ(·)` 是**先归一再加权**（§16.7-⑥）：四样东西量纲不同，等权直接相加是错的。

阶段二（扩散）：每轮剪阈值、每跳按边类型衰减、只取 top-M、总展开有硬上限。

**第六道闸门「口径闸门」（§15.2-⑥）**：`activation` 是"想起来了什么"，
`rank_scores` 是"该按什么顺序读"。两者**故意分开** —— 阶段一的种子分（z 归一，~O(1)）和
阶段二的扩散分（出度归一后 ~O(0.01)）不在一个量纲上，直接按 `activation` 排序会让
"同名的另一件事"永远压住正确答案（早期夹具上反复出现）。
见 `calibrate_rank()`；它**只改排序**，不改激活集合、不改图、不改观测量。

边能不能走（三条规则，对应 §15.1③ / §15-⑤ / §17）：
1. `derived_from`：**永远能走** —— 它是唯一的跨场景通道（§15.1③）；
2. `co_occur`：**同场景能走**；一端是**语义层**节点（无场景锚）时也能走，
   但贡献要乘 `semantic_weight(D)` —— 这就是 §15-⑤"语义层豁免"由 `D` 来调（§16.1 最后一行）；
   **情景层 ⇄ 情景层 跨场景：掐断**（§9.3 那条"麦当劳那次不许和游泳那次揉在一起"）；
3. `play`（§17 玩梗边）：**默认关**；开了也只允许 `D ≥ play_min_d` 的检索走到。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

from .config import Params
from .divergence import DParams, params_for
from .lexicon import lexical_relevance, z_normalize
from .lifecycle import view_of
from .types import Edge, EdgeKind, GraphSnapshot, NodeKind, out_degree_normalized


@dataclass(frozen=True)
class Query:
    """一次检索的输入。`relevance` 可以注入（将来换向量召回时用），不填就用词面。"""

    terms: tuple[str, ...]
    scene: Optional[str] = None
    now: float = 0.0
    relevance: Optional[Mapping[int, float]] = None
    # 🆕 §19.1「随机检索器」：显式给定种子节点（跳过阶段一的词面匹配）。
    #    典型用法：`random_seed_nodes()` 抽几个随机节点塞进来 ——
    #    这不是"相关检索"，是**自发回忆/走神**（involuntary memory）。
    seed_nodes: tuple[int, ...] = ()
    # 🆕 随机源的种子：**必须记进结果**，否则 §16.5 的"跑两遍逐位相同"会失效。
    seed: int = 0


@dataclass(frozen=True)
class RetrieveResult:
    """一次检索的结果（**纯数据**，可以直接序列化进实验产物）。"""

    d: float
    dparams: DParams
    seeds: tuple[int, ...]
    activation: Mapping[int, float]
    depth: Mapping[int, int]
    seed_scene: Optional[str]
    expanded_raw: int          # ★剪枝**前**展开过的节点数（爆炸测试看这个，§16.7-⑤）
    pruned: int
    capped: bool               # 是否撞到 N_max 上限
    seed_scores: Mapping[int, float] = field(default_factory=dict)
    # 第六道闸门（§15.2-⑥）后的**可比排序分**：同场景、同阶段内保序，跨阶段可比。
    # ⚠️ 它和 `activation` 不是一回事：`activation` 回答"想起来了什么"，它回答"先读哪条"。
    rank_scores: Mapping[int, float] = field(default_factory=dict)
    # 🆕 §19.2 随机跳（teleport）的审计字段：跳了几次、用的哪个随机种子。
    teleports: int = 0
    seed: int = 0
    # 本次检索**实际生效**的边增益门槛（可能被"出度上限"压低，见 retrieve 里的说明）
    effective_min_gain: float = 0.0
    # 结果自带场景/类型映射（都是只读的快照投影）——这样 ratio 属性不需要再拿 snapshot，
    # 结果本身就能序列化、能被测试独立复核。
    scene_map: Mapping[int, Optional[str]] = field(default_factory=dict)
    semantic_ids: frozenset[int] = frozenset()

    @property
    def activated(self) -> int:
        return len(self.activation)

    @property
    def cross_scene_ratio(self) -> float:
        """激活集合里**不属于种子场景**的比例（§16.5 说的"发散的直接证据"）。"""

        if not self.activation:
            return 0.0
        other = sum(1 for nid in self.activation if self.scene_map.get(nid) != self.seed_scene)
        return other / len(self.activation)

    @property
    def semantic_ratio(self) -> float:
        if not self.activation:
            return 0.0
        n = sum(1 for nid in self.activation if nid in self.semantic_ids)
        return n / len(self.activation)

    @property
    def avg_hops(self) -> float:
        """激活节点的平均跳数（§16.5 的指标之一：随 `D` 单调增）。"""

        if not self.depth:
            return 0.0
        return sum(self.depth.values()) / len(self.depth)

    @property
    def diffused_cross_scene_ratio(self) -> float:
        """**只看扩散出来的**（深度 ≥ 1）节点里，跨出种子场景的比例。

        比"整个激活集合"的比值锐利得多：种子本来就是词面命中的，可能横跨几个场景，
        会把扩散的效果冲淡。§16.5 想看的"发散"其实是这一项。
        """

        diffused = [nid for nid, dep in self.depth.items() if dep >= 1 and nid in self.activation]
        if not diffused:
            return 0.0
        other = sum(1 for nid in diffused if self.scene_map.get(nid) != self.seed_scene)
        return other / len(diffused)

    def top(self, k: int = 5) -> list[tuple[int, float]]:
        """最该被读进去的 `k` 条 —— 按**口径闸门后的排序分**（不是裸激活值）。

        ⚠️ 早期 改：以前这里按 `activation` 排，于是"同名的另一件事"永远排在
        正确答案前面（见 `calibrate_rank()` 的说明）。要拿"想起来了什么"用 `activation`。
        `rank_scores` 为空（老产物/手工构造）时退化为原来的行为。
        """

        table = self.rank_scores or self.activation
        return sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def top_by_activation(self, k: int = 5) -> list[tuple[int, float]]:
        """按**裸激活值**排的 `k` 条（"扩散到的最强节点"，用于爆炸/发散类指标）。"""

        return sorted(self.activation.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def as_dict(self) -> dict:
        return {
            "D": self.d,
            "params": self.dparams.as_dict(),
            "seeds": list(self.seeds),
            "seed_scene": self.seed_scene,
            "activated": self.activated,
            "expanded_raw": self.expanded_raw,
            "pruned": self.pruned,
            "capped": self.capped,
            "cross_scene_ratio": round(self.cross_scene_ratio, 6),
            "diffused_cross_scene_ratio": round(self.diffused_cross_scene_ratio, 6),
            "semantic_ratio": round(self.semantic_ratio, 6),
            "avg_hops": round(self.avg_hops, 6),
            "activation": {str(k): round(v, 9) for k, v in sorted(self.activation.items())},
            "rank_scores": {str(k): round(v, 9) for k, v in sorted(self.rank_scores.items())},
            "depth": {str(k): v for k, v in sorted(self.depth.items())},
        }


def edge_allowed(
    edge: Edge,
    snapshot: GraphSnapshot,
    dp: DParams,
    params: Params,
) -> bool:
    """这一跳能不能走（见模块 docstring 的三条规则）。"""

    if edge.kind is EdgeKind.PLAY:
        return bool(params.enable_play_edges) and dp.d >= params.play_min_d
    if edge.kind is EdgeKind.DERIVED_FROM:
        return True
    # co_occur
    src_scene = snapshot.scene_of(edge.src)
    dst_scene = snapshot.scene_of(edge.dst)
    if src_scene == dst_scene:
        return True
    src_semantic = snapshot.kind_of(edge.src) is NodeKind.SEMANTIC
    dst_semantic = snapshot.kind_of(edge.dst) is NodeKind.SEMANTIC
    if src_semantic or dst_semantic:
        return True  # §15-⑤ 的语义层豁免；强度由 semantic_weight(D) 承担
    return bool(params.cross_scene_bridge)  # 情景层跨场景：默认掐断（§17.5-①）


def edge_gain(
    edge: Edge,
    snapshot: GraphSnapshot,
    dp: DParams,
    params: Params,
    w_hat: Mapping[tuple[int, int, str], float],
) -> float:
    """一跳能传多少：`ŵ × γ_type × (语义层权重)`。"""

    # ⚠️ key 带 kind：平行边（同一对节点上的 co_occur + derived_from）不能共用同一个 ŵ
    gain = w_hat.get((edge.src, edge.dst, edge.kind.value), 0.0) * dp.gamma.get(edge.kind, 0.0)
    if snapshot.kind_of(edge.dst) is NodeKind.SEMANTIC:
        gain *= dp.semantic_weight
    return gain


def seed_scores(
    snapshot: GraphSnapshot,
    query: Query,
    params: Params,
    *,
    relevance: Optional[Mapping[int, float]] = None,
) -> dict[int, float]:
    """阶段一的种子分（§15.0 的公式，已按 §16.7-⑥ 归一 + 显式权重）。"""

    rel = dict(relevance if relevance is not None else (query.relevance or {}))
    if not rel:
        rel = lexical_relevance(snapshot, query.terms, query.scene)
    candidates = {nid: score for nid, score in rel.items() if score > 0.0 and nid in snapshot.nodes}
    if not candidates:
        return {}
    # 候选集内的各特征（先归一，再加权）
    importance = {nid: max(0.0, snapshot.nodes[nid].obs.importance) for nid in candidates}
    views = {nid: view_of(snapshot.nodes[nid], query.now, params) for nid in candidates}
    salience = {nid: v.salience_now for nid, v in views.items()}
    recency = {nid: v.recency for nid, v in views.items()}
    z_rel = z_normalize(candidates)
    z_imp = z_normalize(importance)
    z_sal = z_normalize(salience)
    z_rec = z_normalize(recency)
    scores: dict[int, float] = {}
    for nid in sorted(candidates):
        scores[nid] = (
            params.w_relevance * z_rel.get(nid, 0.0)
            + params.w_importance * z_imp.get(nid, 0.0)
            + params.w_salience * z_sal.get(nid, 0.0)
            + params.w_recency * z_rec.get(nid, 0.0)
        )
    return scores


def calibrate_rank(
    activation: Mapping[int, float],
    depth: Mapping[int, int],
    scores: Mapping[int, float],
    scene_map: Mapping[int, Optional[str]],
    seed_scene: Optional[str],
    params: Params,
) -> dict[int, float]:
    """**第六道闸门「口径闸门」**（§15.2-⑥）：把两阶段的分数放到同一把尺子上。

    病灶（`早期实验` 实测，见实验记录发现 1）：

    * 阶段一是 `ω₁ẑ(相关)+ω₂ẑ(重要)+ω₃ẑ(S₀)+ω₄·recency`，量纲 ~`O(1)`（实测 1.0–3.2）；
    * 阶段二是 `A_j × ŵ × γ`，而 `ŵ` 出度归一（⟨出度⟩ 30 ⇒ ≈1/30）⇒ 实测 **≈0.013**。

    两者差两个数量级 ⇒ 库里存在"同名的另一件事"（别的场景、同一个线索词）时，
    它作为种子**永远压在正确答案上面**（早期夹具上反复出现）。

    规则（两条，都只作用于**排序**，不增删任何节点）：

    1. **各自按本阶段 top-1 归一**（`base ∈ [0,1]`），阶段二再乘 `stage2_weight`：
       阶段内**保序**，阶段间可比；`stage2_weight < 1` 表示"同分时优先信种子"。
    2. **跨场景在排序上打折**：节点场景 ≠ `seed_scene` 时乘 `cross_scene_rank_factor`。
       `scene is None` 的**语义层**节点豁免（与 §15-⑤ 的语义层豁免一致）；
       这条就是"排序时重新应用场景先验"，也是把「题外压它 1 → 0」的那一条。

    关掉这道闸门：`Params(stage2_weight=1.0, cross_scene_rank_factor=1.0)`
    （此时仍会归一，但两阶段的相对高低回到"种子恒赢"，即修复前的行为）。
    """

    if not activation:
        return {}
    seed_ids = [nid for nid, dep in depth.items() if dep == 0 and nid in activation]
    span_ids = [nid for nid, dep in depth.items() if dep >= 1 and nid in activation]
    s1_max = max((max(0.0, scores.get(nid, activation[nid])) for nid in seed_ids), default=0.0)
    s2_max = max((max(0.0, activation[nid]) for nid in span_ids), default=0.0)
    out: dict[int, float] = {}
    for nid, value in activation.items():
        if depth.get(nid, 0) == 0:
            base = max(0.0, scores.get(nid, value)) / (s1_max + params.rank_eps)
        else:
            base = params.stage2_weight * max(0.0, value) / (s2_max + params.rank_eps)
        node_scene = scene_map.get(nid)
        if node_scene is not None and seed_scene is not None and node_scene != seed_scene:
            base *= params.cross_scene_rank_factor
        out[nid] = base
    return out


def retrieve(
    snapshot: GraphSnapshot,
    query: Query,
    d: float,
    params: Optional[Params] = None,
    *,
    relevance: Optional[Mapping[int, float]] = None,
) -> RetrieveResult:
    """跑一次两阶段检索。**纯函数**：不写盘、不改图、无随机。"""

    p = params or Params()
    dp = params_for(d, p)
    if query.seed_nodes:
        # §19.1 随机检索器：**显式种子**，跳过阶段一的词面匹配（不是"相关"，是"自发回忆"）。
        seeds = [int(nid) for nid in query.seed_nodes if int(nid) in snapshot.nodes]
        if not seeds:
            return _empty_result(dp, snapshot)
        scores = {nid: 1.0 for nid in seeds}
    else:
        scores = seed_scores(snapshot, query, p, relevance=relevance)
        if not scores:
            return _empty_result(dp, snapshot)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        seeds = [nid for nid, s in ranked[: max(1, p.seed_topk)] if s >= p.seed_threshold]
        if not seeds:
            return _empty_result(dp, snapshot)
    seed_set = set(seeds)
    seed_scene = query.scene if query.scene is not None else snapshot.scene_of(seeds[0])

    if not p.diffusion_enabled:  # 空对照臂：只做阶段一（§15.4）
        scene_map = {nid: n.scene for nid, n in snapshot.nodes.items()}
        seed_only = {nid: scores[nid] for nid in seeds}
        zero_depth = {nid: 0 for nid in seeds}
        return RetrieveResult(
            d=dp.d,
            dparams=dp,
            seeds=tuple(seeds),
            activation=seed_only,
            depth=zero_depth,
            seed_scene=seed_scene,
            expanded_raw=len(seeds),
            pruned=0,
            capped=False,
            seed_scores=seed_only,
            rank_scores=calibrate_rank(seed_only, zero_depth, scores, scene_map, seed_scene, p),
            scene_map=scene_map,
            semantic_ids=frozenset(
                nid for nid, n in snapshot.nodes.items() if n.kind is NodeKind.SEMANTIC
            ),
        )

    w_hat = out_degree_normalized(snapshot.edges)
    # ⚠️ 边增益门槛的"悬崖"保护（早期）：`min_gain` 是绝对量，而单边能分到的份额是
    #    `γ / ⟨出度⟩`。图一变稠（出度 30），阈值一旦超过那个份额，扩散就**整条归零**
    #    （实测 253 → 9）。所以这里给它加一个与出度挂钩的上限，并把生效值暴露出来便于审计。
    #    正解应当写成相对量（"至少分到平均单边份额的 x%"），列在待办里。
    avg_deg = (sum(len(v) for v in snapshot.adj.values()) / max(1, len(snapshot.adj))) if snapshot.adj else 1.0
    # ⚠️ 只算**真的能走**的边类型：PLAY 默认关着，若把它的 γ（0.2）算进来会把夹具压低一半
    active_gammas = [v for k, v in dp.gamma.items() if p.enable_play_edges or k is not EdgeKind.PLAY]
    gamma_floor = min(active_gammas) if active_gammas else 0.0
    # 上限取**平均单边份额**（γ/⟨出度⟩）：阈值顶多等于它 —— 那样至少还能保住"平均以上"的边，
    # 而不是把所有边一起剪掉（取 0.5 倍会把门槛压得永远不生效，实测过）。
    eff_min_gain = min(p.min_gain, gamma_floor / max(1.0, avg_deg))
    activation: dict[int, float] = {nid: scores[nid] for nid in seeds}
    depth: dict[int, int] = {nid: 0 for nid in seeds}
    ever_expanded: set[int] = set(seeds)
    pruned = 0
    capped = False

    frontier = sorted(seeds, key=lambda nid: (-activation[nid], nid))
    # 🆕 §19.2 随机跳（teleport）：**只有开了才建 RNG** ⇒ 关着时与旧实现逐位相同。
    rng = random.Random(query.seed) if p.teleport_prob > 0.0 else None
    teleport_budget = (max(1, int(p.n_max * max(0.0, p.teleport_budget_ratio)))
                       if rng is not None else 0)
    teleports = 0
    prev_teleport = False
    for hop in range(1, max(1, dp.depth_max) + 1):
        delta: dict[int, float] = {}
        for j in frontier:
            if j not in activation:
                continue
            a_j = activation[j]
            # ⚠️ **先过滤、再取 top-M**（早期实验踩到的坑）：
            # 如果先切 top-M，`co_occur` 边在数量上永远压倒 `derived_from`
            # （共享标签到处都是，冷启动时尤其如此）⇒ **唯一的跨场景通道被挤出前 M 名**，
            # §9.3 与 §15-⑤ 那套"情景 → 抽象"在检索侧会**静默失效**。
            # 排序也按"这一跳能传多少"（增益）排，而不是按裸权重排。
            gains: list[tuple[float, int, Edge]] = []
            for edge in snapshot.adj.get(j, ()):
                if not edge_allowed(edge, snapshot, dp, p):
                    continue
                gain = edge_gain(edge, snapshot, dp, p, w_hat)
                if gain <= 0.0 or gain < eff_min_gain:
                    continue
                gains.append((gain, edge.dst, edge))
            gains.sort(key=lambda t: (-t[0], t[1]))
            for gain, _dst, edge in gains[: max(1, dp.top_m)]:
                delta[edge.dst] = delta.get(edge.dst, 0.0) + a_j * gain
        if not delta:
            break
        # ① 每轮剪：**相对**阈值（绝对阈值在量纲变化后会失效，§15.1①）。
        #    两种口径见 config.Params.beta_mode —— 早期实验证明它们不是一回事：
        #    "global"（文档字面）按全局 max(A) 剪，"hop" 按本跳增量剪。
        if p.beta_mode == "hop":
            hop_threshold = dp.beta * max(delta.values())
            admitted = {nid: v for nid, v in delta.items() if v >= hop_threshold}
            pruned += len(delta) - len(admitted)
        else:
            admitted = dict(delta)
        for nid in sorted(admitted):
            # `N_max` 是**硬**上限（§15.2）：合并到满就停，不允许"整跳超一点"。
            if len(ever_expanded) >= p.n_max and nid not in ever_expanded:
                capped = True
                break
            activation[nid] = activation.get(nid, 0.0) + admitted[nid]
            if nid not in depth:
                depth[nid] = hop
            ever_expanded.add(nid)
        if capped:
            break
        if p.beta_mode in ("global", "seed"):
            if p.beta_mode == "global":
                reference = max(activation.values())
            else:
                reference = max(scores[nid] for nid in seeds)
            threshold = dp.beta * reference
            for nid in sorted(activation):
                if nid in seed_set and p.retain_seeds:
                    continue
                if activation[nid] < threshold:
                    del activation[nid]
                    pruned += 1
        # 🆕 §19.2 **随机跳**（teleport / restart）：以概率 `p` 瞬时传送到远处一个节点，
        #    下一跳从它继续小范围扩散 —— 这是"自发回忆"里"突然想到八竿子打不着的事"的机制。
        # 🔴 三条硬约束在这里逐条落实：
        #    ① **计入预算**：`teleports < teleport_budget`（= 10% × n_max），且**不许连续跳**；
        #    ② **只服务闲聊路**：由调用方决定开不开（`teleport_prob` 默认 0），事实问答路不许开；
        #    ③ **只读不写**：这里只动内存里的 activation/depth，**不建边、不强化**。
        # ⚠️ 位置很关键：必须在**算 frontier 之前**跳，否则跳过去的节点不会进入下一跳的扩散。
        if rng is not None and not prev_teleport and teleports < teleport_budget and frontier:
            if rng.random() < p.teleport_prob:
                pool = sorted(nid for nid in snapshot.nodes if nid not in activation)
                if pool:
                    pick = pool[rng.randrange(len(pool))]
                    activation[pick] = max(1e-9, min(activation.values()))
                    depth[pick] = hop
                    ever_expanded.add(pick)
                    teleports += 1
                    prev_teleport = True
        else:
            prev_teleport = False
        frontier = sorted(
            [nid for nid, dep in depth.items() if dep == hop and nid in activation],
            key=lambda nid: (-activation[nid], nid),
        )
        if not frontier:
            break

    scene_map = {nid: n.scene for nid, n in snapshot.nodes.items()}
    return RetrieveResult(
        d=dp.d,
        dparams=dp,
        seeds=tuple(seeds),
        activation=dict(activation),
        depth=dict(depth),
        seed_scene=seed_scene,
        expanded_raw=len(ever_expanded),
        pruned=pruned,
        capped=capped,
        seed_scores={nid: scores[nid] for nid in seeds},
        rank_scores=calibrate_rank(activation, depth, scores, scene_map, seed_scene, p),
        effective_min_gain=round(eff_min_gain, 6),
        scene_map=scene_map,
        semantic_ids=frozenset(
            nid for nid, n in snapshot.nodes.items() if n.kind is NodeKind.SEMANTIC
        ),
        teleports=teleports,          # 🆕 §19.2 审计：这次跳了几次
        seed=query.seed,             # 🆕 随机源种子（可复现的凭据）
    )


@dataclass(frozen=True)
class SceneSetResult:
    """§24.9.2「场景集」的结果（**只读投影**，没有 depth/seed 这类属于单次检索的字段）。"""

    scenes: tuple[str, ...]
    rank_scores: Mapping[int, float]
    activation: Mapping[int, float]

    def top(self, k: int = 5) -> list[tuple[int, float]]:
        table = self.rank_scores or self.activation
        return sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def as_dict(self) -> dict:
        return {
            "scenes": list(self.scenes),
            "rank_scores": {str(k): round(v, 9) for k, v in sorted(self.rank_scores.items())},
            "activation": {str(k): round(v, 9) for k, v in sorted(self.activation.items())},
        }


def random_seed_nodes(snapshot: GraphSnapshot, k: int, *, seed: int = 0) -> tuple[int, ...]:
    """§19.1「随机检索器」的种子：**从库里随机撒 k 个点**（不是相关检索）。

    ⚠️ **必须带 `seed`**：§16.5 的验收要求"同参数跑两遍逐位相同"，
    随机源不可复现 = 那条验收直接失效。
    """

    ids = sorted(snapshot.nodes)
    if not ids:
        return ()
    kk = max(1, min(int(k), len(ids)))
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(ids, kk)))


def random_recall(
    snapshot: GraphSnapshot,
    k: int,
    d: float,
    params: Optional[Params] = None,
    *,
    seed: int = 0,
    scene: Optional[str] = None,
    now: float = 0.0,
) -> RetrieveResult:
    """**随机检索器**：随机撒点当种子 → 走**同一套**扩散（§19.1「两种检索器都有深度检索」）。

    它解决的是**信息茧房**：纯关联检索永远取回"最近、最相关"的那批，
    系统于是反复想到同样的事；随机撒点让沉寂的、弱关联的记忆有机会浮上来 ——
    人的**自发回忆 / 走神**就是"随机激活 + 关联放大"。

    ⚠️ **只服务闲聊 / 主动关怀路**（§19.2 约束②）：事实问答路**不许**用它，
    否则就是"答非所问"（问"我昨天说啥了"，她回"我想起小学同桌"）。
    """

    q = Query(terms=(), scene=scene, now=now,
              seed_nodes=random_seed_nodes(snapshot, k, seed=seed), seed=seed)
    return retrieve(snapshot, q, d, params)


def scene_candidates(
    snapshot: GraphSnapshot,
    query: Query,
    params: Params,
    *,
    top_scenes: int = 6,
) -> list[str]:
    """**从查询自己的候选里**推出"该搜哪些场景"（§24.9.2 的**可实现**版本）。

    为什么需要它：§24.9.2 最初报的"覆盖率 0.34 → 1.00"用的是**上帝视角**
    （两个探针都直接拿了"答案所在的场景"）。真实系统不知道答案在哪，
    所以只能从**自己撒出来的候选**里推：候选节点落在哪些场景，
    按"该场景里最高候选分"排序，取前 `top_scenes` 个。

    实测（早期实验）：这一条可实现规则明显高于"只看查询自带场景"的现状。

    ⚠️ 查询自带的 `scene` **一定排第一**（它是 `seed_scene`，必须搜）。
    """

    q_any = replace(query, scene=None)          # 先不带场景撒一遍，看候选落在哪
    cand = seed_scores(snapshot, q_any, params)
    best: dict[str, float] = {}
    for nid, sc in cand.items():
        node = snapshot.nodes.get(nid)
        if node is None or not node.scene:
            continue
        best[node.scene] = max(best.get(node.scene, 0.0), float(sc))
    ranked = [s for s, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]
    ranked = ranked[: max(1, int(top_scenes))]
    if query.scene and query.scene not in ranked:
        ranked.insert(0, query.scene)
    return ranked


def retrieve_scene_set(
    snapshot: GraphSnapshot,
    query: Query,
    d: float,
    params: Params,
    *,
    top_scenes: int = 6,
) -> SceneSetResult:
    """**逐场景各检索一次，再按 `rank_scores` 取 max 合并**（§24.9.2）。

    它解决的是"她反复做过的事"被第六道闸门误伤：跨出 `seed_scene` 的节点在**排序**上打
    `cross_scene_rank_factor` 折，而"每周三去游泳"每次都是独立场景，
    于是"我游泳都在哪天"只能想起 1–2 次（实测覆盖率 0.320）。

    关键：**不必放开那道惩罚**（0.10 保持不变）—— 每一次子检索里，被搜的那个场景
    自己就是 `seed_scene`，所以那些节点**根本不吃折扣**。

    ⚠️ 与 `retrieve` 的关系：本函数**不改 `retrieve`**，每个子检索仍是纯函数，
    合并在内存里做 ⇒ **同 seed 同结果**（可复现性不受影响）。
    ⚠️ 它**不是默认路径**：要显式调用。是否把它设成默认，是一个**效果 vs 成本**的决定
    （每个查询变成 `len(scenes)` 次检索），留给使用者拍板。
    """

    scenes = scene_candidates(snapshot, query, params, top_scenes=top_scenes)
    rank: dict[int, float] = {}
    act: dict[int, float] = {}
    for sc in scenes:
        r = retrieve(snapshot, replace(query, scene=sc), d, params)
        for nid, v in (r.rank_scores or r.activation).items():
            rank[nid] = max(rank.get(nid, 0.0), float(v))
        for nid, v in r.activation.items():
            act[nid] = max(act.get(nid, 0.0), float(v))
    return SceneSetResult(scenes=tuple(scenes), rank_scores=rank, activation=act)


def _empty_result(dp: DParams, snapshot: GraphSnapshot) -> RetrieveResult:
    return RetrieveResult(
        d=dp.d,
        dparams=dp,
        seeds=(),
        activation={},
        depth={},
        seed_scene=None,
        expanded_raw=0,
        pruned=0,
        capped=False,
        seed_scores={},
        rank_scores={},
        scene_map={nid: n.scene for nid, n in snapshot.nodes.items()},
        semantic_ids=frozenset(
            nid for nid, n in snapshot.nodes.items() if n.kind is NodeKind.SEMANTIC
        ),
    )
