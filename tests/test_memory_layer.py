#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆层的回归测试。

**每一个测试都对应设计文档里的一条硬规矩**（不是为了覆盖率而写）：
冻结、范数不抵消、冷静不喂 S、同场景限制、`derived_from` 是唯一跨场景通道、
可复现、`D` 单调、闸门真的在做功、`N_max` 是硬上限。

跑法：`python -m unittest discover -s tests -v`
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from memlayer import (  # noqa: E402
    MemoryStore,
    Modulators,
    NodeKind,
    Observation,
    Params,
    Query,
    advance_time,
    commit_access,
    freeze_signature,
    keep_ratio,
    persistence,
    reflect,
    SensitiveContentError,
    looks_sensitive,
    remember,
    retrieve,
    salience0,
    SelfTrace,
    params_for_path,
    ReflectionCandidate,
    ReflectionGate,
    ReflectionPolicy,
)
from memlayer.synth import build_bank  # noqa: E402
from memlayer.types import EdgeKind  # noqa: E402
from memlayer.ingest import BASE_PERSIST, induce_preferences  # noqa: E402
from memlayer.store import inspect_snapshot  # noqa: E402
from memlayer.retrieval import calibrate_rank  # noqa: E402
from memlayer.retrieval import retrieve_scene_set, scene_candidates  # noqa: E402
from memlayer import random_recall, random_seed_nodes  # noqa: E402
from memlayer.lifecycle import recency_of, view_of  # noqa: E402
from memlayer.types import MemoryState  # noqa: E402
from memory_backup import (  # noqa: E402
    BackupVerificationError,
    export_memory,
    restore_memory,
    verify_memory_export,
)
from backup_memory import main as backup_main  # noqa: E402


class BaseCase(unittest.TestCase):
    """统一在测试结束时关掉 SQLite 连接（不然跑起来一堆 ResourceWarning 噪声）。"""

    def mkstore(self) -> MemoryStore:
        store = MemoryStore(":memory:")
        self.addCleanup(store.close)
        return store

    def mkbank(self, **kw) -> tuple[MemoryStore, dict]:
        store = self.mkstore()
        return store, build_bank(store, **kw)


def _store_with_bank(**kw) -> tuple[MemoryStore, dict]:
    store = MemoryStore(":memory:")
    stats = build_bank(store, **kw)
    return store, stats


class TestWeights(BaseCase):
    """§11–§14 的权重规矩。"""

    def test_norm_does_not_cancel(self) -> None:
        """§11.1：开心 0.8 + 期待 −0.8 相加 = 0 是错的，范数必须还能看出"情绪复杂"。"""

        p = Params()
        # ⚠️ 早期 修（代码审查抓到的空转测试）：原版用 `anticipation=−0.8` 当"难过"，
        #    但 anticipation 自 v4 起**不进 CHANNELS** ⇒ 那个断言实际只等于 `0.8 > 0.0`。
        #    现在用真的进范数的两条通道来表达"复杂 vs 平淡"。
        both = Observation(happy=0.8, worry=0.8, arousal=0.5)
        flat = Observation(happy=0.0, worry=0.0, arousal=0.5)
        self.assertGreater(salience0(both, Modulators(), p), salience0(flat, Modulators(), p))
        # 顺带钉住：`anticipation` 是派生值，**不许**影响 S₀
        self.assertAlmostEqual(
            salience0(Observation(happy=0.5, arousal=0.5, anticipation=1.0), Modulators(), p),
            salience0(Observation(happy=0.5, arousal=0.5, anticipation=-1.0), Modulators(), p),
            places=12,
        )

    def test_hope_and_worry_coexist_without_cancelling(self) -> None:
        """"既盼着考完解放、又担心考砸"必须**同时**被记住，不能被相减抵消。

        早期 设计说明：`期待` 不能做成「盼望 ↔ 担心」一根轴（−4~+4），
        否则"既盼又怕"会被抵消成 0 —— 和 §11.1 那个"开心+期待−0.8=0"是同一个错。
        ⇒ `hope` / `worry` 是**两条独立通道**，`anticipation = hope − worry` 只作派生值。
        """

        p = Params()
        both = Observation(hope=0.9, worry=0.9, arousal=0.5)
        only_hope = Observation(hope=0.9, arousal=0.5)
        only_worry = Observation(worry=0.9, arousal=0.5)
        self.assertGreater(salience0(both, Modulators(), p), salience0(only_hope, Modulators(), p))
        self.assertGreater(salience0(both, Modulators(), p), salience0(only_worry, Modulators(), p))
        # 派生值仍然能给表达层用（方向）
        self.assertAlmostEqual(both.hope - both.worry, 0.0, places=9)

    def test_importance_does_not_feed_salience(self) -> None:
        """§16.7-②：重要性只喂压缩阈值 ⇒ **不许**改动 S₀。（calm 见下一条，它已改口径）"""

        p = Params()
        base = Observation(happy=0.5, arousal=0.5)
        noisy = Observation(happy=0.5, arousal=0.5, importance=1.0)
        self.assertAlmostEqual(
            salience0(base, Modulators(), p), salience0(noisy, Modulators(), p), places=12
        )

    def test_calm_is_a_channel_since_2026_10_08(self) -> None:
        """🔴 口径变更（设计说明）：`calm` → **`anxiety`（不安）**，并成为**通道**。

        实际判据是「**内心有没有持续的不安/张力**」：纯等待本身是**平静**；
        有具体的"没底/慌"才高。人工标注按"不安度"打分，早期页面把方向写反了。
        ⇒ 它进 S₀；`calm = 1 − anxiety` 降为派生值（§13.5 的"表达抑制"那套已搁置）。
        """

        p = Params()
        anxious = Observation(happy=0.3, arousal=0.5, anxiety=1.0)   # 慌
        peaceful = Observation(happy=0.3, arousal=0.5, anxiety=0.0)  # 平静
        self.assertGreater(
            salience0(anxious, Modulators(), p), salience0(peaceful, Modulators(), p),
            "越不安 ⇒ 这件事在她心里越重（不安是通道，进范数）",
        )
        # 派生值：calm = 1 − anxiety（只为兼容旧代码，不进 CHANNELS）
        self.assertAlmostEqual(Observation(anxiety=0.25).calm, 0.75, places=12)

    def test_unfinished_boosts_salience_zeigarnik(self) -> None:
        """🆕 未完成的事记得更牢（设计说明："等待期不激动，可记得很久"）。

        `κ_unfinished=0` 时必须**与加入这一项之前逐位相同**（向后兼容）。
        """

        p = Params()
        done = Observation(happy=0.3, arousal=0.5, unfinished=0.0)
        pending = Observation(happy=0.3, arousal=0.5, unfinished=1.0)
        self.assertGreater(salience0(pending, Modulators(), p), salience0(done, Modulators(), p))
        self.assertAlmostEqual(
            salience0(done, Modulators(), p),
            salience0(pending, Modulators(), p.variant(kappa_unfinished=0.0)),
            places=12,
        )

    def test_decay_curve_power_has_a_long_tail(self) -> None:
        """§20：遗忘曲线的形状可切，而**幂律（艾宾浩斯式）有长尾** —— 这是"大事忘不掉"的关键。

        指数：`0.5^(t/h)` ⇒ 一年后 ≈ 0（严格归零，没有尾巴）。
        幂律：`1/(1+t/τ)^β` ⇒ 一年后仍剩几个百分点，且**大事仍留在 active 档**。
        """

        from memlayer.lifecycle import recency_of

        exp_p = Params(decay_curve="exponential")
        pow_p = Params(decay_curve="power", power_beta=0.5, power_tau=1.0)
        day = 365.0
        r_exp = recency_of(0.0, day, p=exp_p)
        r_pow = recency_of(0.0, day, p=pow_p)
        self.assertLess(r_exp, 0.001, "指数曲线一年后基本归零")
        self.assertGreater(r_pow, 0.04, "幂律曲线一年后仍留长尾")
        self.assertGreater(r_pow, r_exp * 50, "两者的差别要在量级上可见")
        # 两条曲线都必须**随时间单调不增**，且在 t=0 时为 1
        for prm in (exp_p, pow_p):
            vals = [recency_of(0.0, t, p=prm) for t in (0.0, 1.0, 7.0, 30.0, 365.0)]
            self.assertAlmostEqual(vals[0], 1.0, places=9)
            self.assertEqual(vals, sorted(vals, reverse=True), f"{prm.decay_curve} 应单调不增")

    def test_uncertainty_is_multiplicative_penalty(self) -> None:
        """§13.4：不确定的不该抢注意力 ⇒ S₀ 单调下降。"""

        p = Params()
        sure = Observation(happy=0.8)
        unsure = Observation(happy=0.8, uncertainty=0.5)
        self.assertLess(salience0(unsure, Modulators(), p), salience0(sure, Modulators(), p))

    def test_first_time_multiplies(self) -> None:
        """§11.3：调制器是**乘法**，抽象成一般机制而不是一条特例。"""

        p = Params()
        obs = Observation(happy=0.6, arousal=0.6)
        plain = salience0(obs, Modulators(), p)
        first = salience0(obs, Modulators(first_time=True), p)
        self.assertGreater(first, plain)

    def test_persistence_and_keep_ratio(self) -> None:
        """§13.3：类型先验与强度**两个独立输入**；§12.3：S₀ 越高丢的细节越少。"""

        p = Params()
        self.assertGreater(persistence(1.0, 0.5, p), persistence(1.0, 0.0, p))
        self.assertGreater(persistence(6.0, 0.0, p), persistence(1.0, 0.0, p))
        self.assertGreater(keep_ratio(0.8, p), keep_ratio(0.2, p))
        self.assertEqual(keep_ratio(0.0, p), 0.0)


class TestFreeze(BaseCase):
    """§14.3：写入之后，只有 4 个量是活的。"""

    def test_observations_frozen_after_access(self) -> None:
        store = self.mkstore()
        node = remember(
            store,
            summary="中午去测试场景甲买测试物品12重奏",
            obs=Observation(happy=0.9, arousal=0.8),
            scene="今天中午",
            tags=("测试物品", "测试场景甲"),
            modulators=Modulators(first_time=True),
            created_at=0.0,
        )
        before = freeze_signature(store.nodes())
        commit_access(store, [node.id], 3.0, Params())
        after = freeze_signature(store.nodes())
        self.assertEqual(before, after, "检索命中不许动观测量/场景锚/出处/S₀")
        touched = store.nodes()[0]
        self.assertEqual(touched.access_count, 1)
        self.assertEqual(touched.last_access, 3.0)

    def test_recency_and_s_now_move_with_time_but_do_not_write(self) -> None:
        """§16.7-①：`recency`/`S_now` 是现算的 ⇒ 时间推进不落盘。"""

        store = self.mkstore()
        node = remember(
            store,
            summary="很早以前的事",
            obs=Observation(happy=0.7, arousal=0.7),
            scene="过去",
            created_at=0.0,
        )
        from memlayer.lifecycle import s_now_of

        near = s_now_of(node, 0.0, Params())
        far = s_now_of(node, 30.0, Params())
        self.assertLess(far, near)
        # ⚠️ 早期 修：原版是 `freeze_signature(...) == freeze_signature(...)`（自己跟自己比，恒真）。
        before = freeze_signature(store.nodes())
        _ = advance_time(store.nodes(), 30.0, Params())
        self.assertEqual(before, freeze_signature(store.nodes()),
                         "时间推进不许改任何落盘字段（§14.3）")


class TestEdgeRules(BaseCase):
    """§15.1③ 的边类型规则（含早期实验踩到的 top-M 顺序 bug）。"""

    def test_cross_scene_co_occur_is_blocked_but_derived_from_crosses(self) -> None:
        """§9.3：情境 A 的记忆**不许**直接走到情境 B；抽象（derived_from）才可以。"""

        store = self.mkstore()
        # 关键：查询词只命中 A（否则 B 会因为**自己词面命中**而合法地成为种子，
        # 那就不是在测"跨场景扩散被挡"了）
        a = remember(store, summary="在测试场景甲买测试物品", obs=Observation(happy=0.5),
                     scene="场景A", tags=("测试场景甲", "测试物品"), created_at=0.0)
        b = remember(store, summary="在码头喂鸟也吃测试物品", obs=Observation(happy=0.5),
                     scene="场景B", tags=("测试物品", "鸟"), created_at=1.0)
        snap = store.snapshot()
        self.assertTrue(any(e.src == a.id and e.dst == b.id for e in snap.edges),
                        "共享实体的边**照建**（那是事实）")
        r = retrieve(snap, Query(terms=("测试场景甲",), now=1.0), 1.0, Params(beta_mode="hop"))
        self.assertIn(a.id, r.seeds)
        self.assertNotIn(b.id, r.activation, "跨场景 co_occur 默认掐断")

        sem = reflect(store, summary="他喜欢在零食上尝新", sources=[a, b], created_at=2.0)
        r2 = retrieve(store.snapshot(), Query(terms=("测试场景甲",), now=2.0), 1.0,
                      Params(beta_mode="hop"))
        self.assertIn(sem.id, r2.activation,
                      "derived_from 是**唯一**的跨场景通道，必须走得到（否则 §15-⑤ 白做）")

    def test_play_edges_are_off_by_default_and_d_gated(self) -> None:
        """§17：玩梗边默认关；就算开了，也只允许高 `D` 的检索走到。"""

        store = self.mkstore()
        # 两件事**不共享任何标签** ⇒ b 只能靠 play 边被"走到"。
        # （第一版让它们共享"测试物品"，结果 b 是靠 co_occur 到的 —— 那测的就不是 play 边了。）
        a = remember(store, summary="测试场景甲", obs=Observation(happy=0.5), scene="S",
                     tags=("测试场景甲", "测试物品"))
        b = remember(store, summary="鸟去测试场景甲整点测试物品", obs=Observation(happy=0.9), scene="S",
                     tags=("码头", "鸟"))
        store.add_edge(src=a.id, dst=b.id, kind=EdgeKind.PLAY, weight=5.0)
        store.add_edge(src=b.id, dst=a.id, kind=EdgeKind.PLAY, weight=5.0)
        snap = store.snapshot()
        off = retrieve(snap, Query(terms=("测试场景甲",), now=0.0), 1.0, Params(beta_mode="hop"))
        self.assertNotIn(b.id, off.activation, "默认禁用")
        on_low = retrieve(snap, Query(terms=("测试场景甲",), now=0.0), 0.2,
                          Params(beta_mode="hop", enable_play_edges=True))
        self.assertNotIn(b.id, on_low.activation, "低 D（事实问答路）不许走玩梗边")
        on_high = retrieve(snap, Query(terms=("测试场景甲",), now=0.0), 1.0,
                           Params(beta_mode="hop", enable_play_edges=True))
        self.assertIn(b.id, on_high.activation, "高 D 才允许浮现")


class TestRetrievalProperties(BaseCase):
    """§15.4 / §16.5 的结构性质：可复现、单调、有界。"""

    def setUp(self) -> None:
        self.store = self.mkstore()
        self.stats = build_bank(self.store, n_nodes=300, seed=7)
        self.snap = self.store.snapshot()
        self.q = Query(terms=(self.stats["sample_query"],), now=15.0)

    def test_deterministic_on_frozen_snapshot(self) -> None:
        """§16.5：同 `D` 同线索跑两遍**逐位相同**（在冻结快照上、关掉 Hebbian）。"""

        p = Params(beta_mode="hop")
        r1 = retrieve(self.snap, self.q, 0.7, p)
        r2 = retrieve(self.snap, self.q, 0.7, p)
        self.assertEqual(r1.as_dict(), r2.as_dict())

    def test_activation_monotone_in_d(self) -> None:
        """§16.5：激活集合随 `D` 单调不减（这条如果挂了，说明 `D` 没接对）。"""

        p = Params(beta_mode="hop")
        counts = [retrieve(self.snap, self.q, d / 10.0, p).activated for d in range(11)]
        for prev, cur in zip(counts, counts[1:]):
            self.assertLessEqual(prev, cur, f"单调性破了：{counts}")

    def test_topm_applied_after_edge_filter(self) -> None:
        """早期实验的 bug：top-M 必须在**过滤掉不允许的边之后**再取。

        否则 `co_occur` 会把 `derived_from` 挤出前 M 名，跨场景通道静默失效。
        """

        # 造一个"大量同场景 co_occur + 一条通往语义层的 derived_from"的节点
        store = self.mkstore()
        hub = remember(store, summary="枢纽", obs=Observation(happy=0.5), scene="S",
                       tags=("枢纽", "咖啡"))
        for i in range(30):
            remember(store, summary=f"同场景邻居{i}", obs=Observation(happy=0.1), scene="S",
                     tags=("枢纽",))
        # far 只通过"咖啡"与 hub 相连 ⇒ 只能靠**跨场景 co_occur** 到达，默认必须挡住
        far = remember(store, summary="另一个场景", obs=Observation(happy=0.5), scene="T",
                       tags=("咖啡",))
        sem = reflect(store, summary="抽象", sources=[hub], created_at=1.0)
        snap = store.snapshot()
        r = retrieve(snap, Query(terms=("枢纽",), now=1.0), 1.0,
                     Params(beta_mode="hop", top_m=2))
        self.assertIn(sem.id, r.activation,
                      "top-M 切早了 ⇒ derived_from 被挤出 ⇒ 跨场景通道失效")
        self.assertNotIn(far.id, r.activation, "跨场景 co_occur 仍然要挡住")

    def test_nmax_is_a_hard_cap(self) -> None:
        """§15.2：即使闸门全失败，`N_max` 也要把展开数摁住。"""

        # ⚠️ 早期：默认 `min_gain=0.02` 一开，扩散本来就很收敛 ⇒ 顶不到上限。
        #    要测"硬上限"就得把**其它闸门先都关掉**（min_gain=0、β=0、桥开），否则测的是别的东西。
        p = Params(beta_mode="hop", beta=0.0, min_gain=0.0, n_max=12, cross_scene_bridge=True)
        r = retrieve(self.snap, self.q, 1.0, p)
        self.assertLessEqual(r.expanded_raw, 12)
        self.assertTrue(r.capped)

    def test_gain_gate_does_the_work(self) -> None:
        """§15.1①（第 0 轮结论）：真正在收敛扩散的是**边的增益门槛**。

        ⚠️ 早期 改：原来拿随机合成库比"门槛 0.01 vs 0"，结果两边一样
        （那个库的边权分布太整齐，门槛在钳制范围内根本切不到东西）。
        改成**构造一个能确定性触发的图**：种子上挂一条强边（权重 20）和一条弱边（权重 1）。
        """

        store = self.mkstore()
        # ⚠️ 用 `link=False`：否则"共享标签自动建链"会额外建出 strong→weak 的边，
        #    弱节点会沿着第二条路绕进来（第一版就是这么翻车的）。
        # ⚠️ 查询词只能命中**种子**：第一版三个节点都带"甲"，于是弱节点自己就是种子（词面命中），
        #    根本不是"被扩散找到的" ⇒ 测不到门槛。这里给三个节点互不相同的标签。
        a = remember(store, summary="种子", obs=Observation(happy=0.5), scene="S", tags=("种子词",), link=False)
        strong = remember(store, summary="强邻居", obs=Observation(happy=0.5), scene="S",
                          tags=("乙",), link=False)
        weak = remember(store, summary="弱邻居", obs=Observation(happy=0.5), scene="S",
                        tags=("丙",), link=False)
        # 唯一的两条边都在种子上：a→强 = 20，a→弱 = 1
        store.add_edge(src=a.id, dst=strong.id, kind=EdgeKind.CO_OCCUR, weight=20.0)
        store.add_edge(src=a.id, dst=weak.id, kind=EdgeKind.CO_OCCUR, weight=1.0)
        snap = store.snapshot()
        q = Query(terms=("种子词",), now=0.0)

        ungated = retrieve(snap, q, 1.0, Params(min_gain=0.0, beta=0.0, n_max=10 ** 9))
        self.assertIn(strong.id, ungated.activation)
        self.assertIn(weak.id, ungated.activation)

        gated = retrieve(snap, q, 1.0, Params(min_gain=0.1, beta=0.0, n_max=10 ** 9))
        self.assertIn(strong.id, gated.activation, "强边必须过得去")
        self.assertNotIn(weak.id, gated.activation, "弱边必须被增益门槛剪掉")

    def test_beta_is_nearly_noop_once_gain_gate_is_on(self) -> None:
        """已知行为（写下来免得以后误以为 β 还在做功）：`min_gain>0` 时 β 的影响很小。"""

        base = Params(beta_mode="hop", cross_scene_bridge=True, n_max=10 ** 9)
        with_beta = retrieve(self.snap, self.q, 1.0, base)
        without_beta = retrieve(self.snap, self.q, 1.0, base.variant(beta=0.0))
        self.assertLessEqual(
            abs(without_beta.activated - with_beta.activated), max(3, int(0.2 * max(1, with_beta.activated))),
            "若这里差值变大，说明 β 又开始做功了 —— 那就得重新审一遍两个闸门的分工",
        )

    def test_default_params_actually_diffuse(self) -> None:
        """🔴 代码审查抓到的问题：`Params()` 的**默认**检索路径原先没有任何测试跑过，
        而它的默认值恰好是"扩散被剪掉 95%、行为≈空对照"的那一组。
        现在默认值已改成实验选出的那一组，这条测试就是那个承诺的守门人。
        """

        r = retrieve(self.snap, self.q, 0.8, Params())          # 故意**不传任何参数**
        self.assertGreater(r.activated, len(r.seeds), "默认参数下扩散必须在做事（不是只回种子）")

    def test_empty_control_is_seed_only(self) -> None:
        """§15.4 的空对照：关掉扩散 ⇒ 只剩种子（测试场景甲用例就该在这条臂上失败）。"""

        p = Params(beta_mode="hop", diffusion_enabled=False)
        r = retrieve(self.snap, self.q, 1.0, p)
        self.assertEqual(r.activated, len(r.seeds))
        self.assertEqual(r.expanded_raw, len(r.seeds))

    def test_query_with_no_match_returns_empty(self) -> None:
        r = retrieve(self.snap, Query(terms=("不存在的词xyz",), now=1.0), 1.0)
        self.assertEqual(r.activated, 0)
        self.assertEqual(r.seeds, ())

    def test_retrieval_is_read_only(self) -> None:
        """🔒 检索是**纯函数**：跑多少次、`D` 取什么值，图都不能被改动。

        这条对应查重结论里的优先项 ④（"检索只读 + 出处保全"）——
        它是"同 `D` 同线索跑两遍逐位相同"（§16.5）能成立的前提，
        也是把"她想起某件事"和"她改了自己的记忆"彻底分开的那道墙。

        指纹包含：观测量 / 场景锚 / 标签 / 出处 / `S₀`（节点侧）+ 每条边的权重与类型（边侧）。
        """

        def graph_fingerprint() -> tuple:
            nodes = freeze_signature(store.nodes())
            edges = tuple(
                sorted((e.src, e.dst, e.kind.value, round(e.weight, 12)) for e in store.edges())
            )
            return (nodes, edges)

        store = self.mkstore()
        build_bank(store, n_nodes=120, seed=3)
        before = graph_fingerprint()
        for d in (0.0, 0.5, 1.0):
            for _ in range(3):
                retrieve(store.snapshot(), Query(terms=("ent00003",), now=11.0), d,
                         Params(cross_scene_bridge=True))
        self.assertEqual(before, graph_fingerprint(),
                         "检索不许改图（改了就不是纯函数，可复现性也就没了）")


class TestRankCalibrationGate(BaseCase):
    """§15.2-⑥ 第六道「口径闸门」：两阶段分数对齐 + 排序时重加场景先验。

    病灶（`早期实验` 实测，见实验记录发现 1）：
    阶段一是 z 归一的加权和（~`O(1)`，实测 1.0–3.2），阶段二是出度归一后的
    `A_j · ŵ · γ`（~`O(0.01)`）⇒ 库里只要存在一个「同名的另一件事」（别的场景、
    同一个线索词），它作为种子**永远压在正确答案上面**（30/30）。

    这一族测试守的就是那条：**排序分可比 + 跨场景在排序上打折**，且**只改读法**。
    """

    def _fixture(self) -> tuple[MemoryStore, dict, str]:
        """线索词同时出现在两个场景；答案在场景 A 里、且**不是种子**（只能 2 跳走到）。"""

        store = self.mkstore()
        cue, m1, m2, ans = "cueX", "m1X", "m2X", "ansX"
        trigger = remember(
            store,
            summary=f"那天她说起 {cue}",
            obs=Observation(happy=0.6, arousal=0.9, importance=0.9),
            scene="A",
            tags=(cue, m1),
            created_at=10.0,
            source_quote="[t]",
        )
        mid = remember(
            store,
            summary="中间的经过",
            obs=Observation(arousal=0.5, importance=0.6),
            scene="A",
            tags=(m1, m2),
            created_at=10.0,
            source_quote="[m]",
        )
        gold = remember(
            store,
            summary=f"结局：{ans}",
            obs=Observation(happy=0.7, importance=0.8),
            scene="A",
            tags=(m2, ans),
            created_at=10.0,
            source_quote="[g]",
        )
        side = remember(
            store,
            summary=f"别的时候也说过 {cue}",
            obs=Observation(arousal=0.1, importance=0.1),
            scene="B",
            tags=(cue,),
            created_at=1.0,
            source_quote="[s]",
        )
        ids = {"trigger": trigger.id, "mid": mid.id, "gold": gold.id, "side": side.id}
        return store, ids, cue

    def test_cross_scene_lookalike_no_longer_outranks_the_answer(self) -> None:
        """🔴 核心回归：默认参数下，「同名的另一件事」必须排在答案**后面**。"""

        store, ids, cue = self._fixture()
        r = retrieve(store.snapshot(), Query(terms=(cue,), now=30.0), 1.0, Params(beta_mode="hop"))
        self.assertIn(ids["gold"], r.activation, "答案必须可达（2 跳）")
        self.assertEqual(r.depth[ids["gold"]], 2)
        order = [nid for nid, _ in r.top(50)]
        self.assertLess(
            order.index(ids["gold"]),
            order.index(ids["side"]),
            "同场景的链条必须排在「同名的另一件事」前面",
        )

    def test_gate_off_reproduces_the_old_ordering(self) -> None:
        """把场景先验关掉（`cross_scene_rank_factor=1.0`）⇒ 旧行为回来。

        这条**故意**断言那个坏的排序，用来证明修复确实作用在这一点上，
        而不是别的什么碰巧把它挤下去了（§15.2-⑥ 的对照臂）。
        """

        store, ids, cue = self._fixture()
        p = Params(beta_mode="hop", cross_scene_rank_factor=1.0)
        r = retrieve(store.snapshot(), Query(terms=(cue,), now=30.0), 1.0, p)
        order = [nid for nid, _ in r.top(50)]
        self.assertLess(order.index(ids["side"]), order.index(ids["gold"]))

    def test_gate_is_ranking_only(self) -> None:
        """闸门**只改排序**：开/关两种先验强度下，激活集合与每个激活值必须逐位相同。"""

        store, _ids, cue = self._fixture()
        snap = store.snapshot()
        q = Query(terms=(cue,), now=30.0)
        on = retrieve(snap, q, 1.0, Params(beta_mode="hop"))
        off = retrieve(snap, q, 1.0, Params(beta_mode="hop", cross_scene_rank_factor=1.0))
        self.assertEqual(on.activation, off.activation)
        self.assertEqual(on.depth, off.depth)
        self.assertEqual(on.seeds, off.seeds)
        self.assertEqual(on.expanded_raw, off.expanded_raw)
        self.assertNotEqual(on.rank_scores, off.rank_scores, "排序分必须真的不一样，否则闸门没接上")

    def test_semantic_layer_is_exempt_from_the_scene_penalty(self) -> None:
        """§15-⑤：`scene=None` 的语义层节点**不受**跨场景惩罚（它本来就是跨场景的抽象）。"""

        p = Params(beta_mode="hop")   # cross_scene_rank_factor = 0.1
        activation = {1: 1.0, 2: 0.5, 3: 0.5}
        depth = {1: 0, 2: 1, 3: 1}
        scores = {1: 2.0}
        scene_map = {1: "A", 2: "A", 3: None}
        rank = calibrate_rank(activation, depth, scores, scene_map, "A", p)
        self.assertAlmostEqual(rank[1], 1.0, places=9)                 # 种子 top-1 → 1.0
        self.assertAlmostEqual(rank[2], 0.8, places=9)                 # 同场景扩散 → stage2_weight
        self.assertAlmostEqual(rank[3], 0.8, places=9)                 # 语义层：不乘 0.1
        self.assertGreater(rank[2], p.cross_scene_rank_factor, "同场景不该被罚到先验系数以下")

    def test_rank_scores_are_bounded_and_stratified(self) -> None:
        """排序分有界（`[0,1]`）且**阶段内保序**：种子 top-1 恒为 1，扩散 top-1 恒为 `stage2_weight`。"""

        store, _ids, cue = self._fixture()
        r = retrieve(store.snapshot(), Query(terms=(cue,), now=30.0), 1.0, Params(beta_mode="hop"))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in r.rank_scores.values()), r.rank_scores)
        best_seed = max(r.rank_scores[nid] for nid in r.seeds)
        best_hop = max(r.rank_scores[nid] for nid, dep in r.depth.items() if dep >= 1)
        self.assertAlmostEqual(best_seed, 1.0, places=9)
        self.assertAlmostEqual(best_hop, Params().stage2_weight, places=9)

    def test_top_follows_the_calibrated_order(self) -> None:
        """`top()` 必须按排序分给序（它是要喂进 prompt 的那几条），裸激活另有出口。"""

        store, _ids, cue = self._fixture()
        r = retrieve(store.snapshot(), Query(terms=(cue,), now=30.0), 1.0, Params(beta_mode="hop"))
        self.assertEqual(
            [nid for nid, _ in r.top(50)],
            sorted(r.rank_scores, key=lambda n: (-r.rank_scores[n], n)),
        )
        self.assertNotEqual([nid for nid, _ in r.top(50)], [nid for nid, _ in r.top_by_activation(50)])


class TestIngest(BaseCase):
    """写入路径（§14.1 / §9.3）。"""

    def test_shared_tags_create_edges_both_ways(self) -> None:
        store = self.mkstore()
        a = remember(store, summary="A", obs=Observation(happy=0.5), scene="S", tags=("测试物品", "咖啡"))
        b = remember(store, summary="B", obs=Observation(happy=0.5), scene="S", tags=("测试物品",))
        edges = store.edges()
        self.assertTrue(any(e.src == a.id and e.dst == b.id for e in edges))
        self.assertTrue(any(e.src == b.id and e.dst == a.id for e in edges))
        w = [e.weight for e in edges if e.src == a.id and e.dst == b.id][0]
        self.assertEqual(w, 1.0, "权重 = 共享标签数")

    def test_reflect_only_adds_nodes(self) -> None:
        """§9.3：抽象**只能新增**，旧节点一个字节不改。"""

        store = self.mkstore()
        a = remember(store, summary="A", obs=Observation(happy=0.5), scene="S", tags=("测试物品",))
        before = freeze_signature(store.nodes())
        reflect(store, summary="抽象", sources=[a], created_at=1.0)
        after = freeze_signature(store.nodes())
        self.assertEqual(before, after[: len(before)])
        node = [n for n in store.nodes() if n.kind is NodeKind.SEMANTIC][0]
        self.assertIsNone(node.scene, "语义层节点没有场景锚（§9.3）")

    def test_preference_is_induced_from_repeated_events(self) -> None:
        """🆕「测试物品是刚炸的、脆」→ **新增**一条偏好节点，而不是把事件改判成偏好。

        调用方 早期：那个细节值得长期留 —— 但那条事件本身确实只是"今天的一件事"
        （机器问"这段文本是不是关于他的偏好"只给 0.41）。正解是 §9.3 的"抽象只能新增"：
        从**重复出现的**事件里归纳出「他喜欢刚炸的测试物品」，并 `derived_from` 指回它们。
        """

        store = self.mkstore()
        # 两条同标签事件，情绪一致偏正
        a = remember(store, summary="测试物品是刚炸的，脆", obs=Observation(happy=0.8, arousal=0.4),
                     scene="D1", tags=("测试物品", "测试场景甲"), source_quote="测试物品刚炸的")
        b = remember(store, summary="又买了刚炸的测试物品", obs=Observation(happy=0.7, arousal=0.3),
                     scene="D9", tags=("测试物品",), source_quote="刚炸的好吃")
        before = freeze_signature(store.nodes())

        created = induce_preferences(store)
        self.assertEqual(len(created), 1, "只该为「测试物品」归纳出一条偏好（测试场景甲只有 1 条事件）")
        pref = created[0]
        self.assertIs(NodeKind.SEMANTIC, pref.kind, "偏好是抽象 ⇒ 语义层节点")
        self.assertIsNone(pref.scene, "偏好没有场景锚")
        self.assertAlmostEqual(pref.base_persist, BASE_PERSIST["preference"], places=12,
                               msg="'偏好活得久'由**类型先验**给，不靠抬高 importance")
        # 旧节点一个字节都不许改（§14.3 / §9.3）
        self.assertEqual(before, freeze_signature(store.nodes())[: len(before)])
        # 两条来源都有 derived_from 指过来（§15.1③ 唯一的跨场景通道）
        edges = {(e.src, e.dst) for e in store.edges() if e.kind is EdgeKind.DERIVED_FROM}
        self.assertIn((a.id, pref.id), edges)
        self.assertIn((b.id, pref.id), edges)
        # 幂等：再跑一遍不会堆节点
        self.assertEqual(induce_preferences(store), [])

    def test_preference_not_induced_when_emotions_conflict(self) -> None:
        """情绪不一致就不许下结论（"不敢乱归纳"）——这是 v1 规则版的边界。"""

        store = self.mkstore()
        remember(store, summary="测试物品好吃", obs=Observation(happy=0.9), scene="D1", tags=("测试物品",))
        remember(store, summary="测试物品很难吃", obs=Observation(angry=0.9), scene="D2", tags=("测试物品",))
        self.assertEqual(induce_preferences(store), [])


class TestLifecycleState(BaseCase):
    """§3.3/§20.3 生命周期状态：**决定"要不要主动想起"，不是"还在不在"**（早期 实现）。

    四条对应的规矩：
    1. 高 `S_now` + 新鲜/被访问过 ⇒ `active`；中 ⇒ `cold`；低 ⇒ `archived`；
    2. **`active` 必须同时看"最近有没有被用到"** —— 否则幂律长尾会让远古旧事一直占着主动关怀；
    3. 碰一下（`commit_access`）能把旧的大事**重新拉回 active**；
    4. **归档 ≠ 删除**：状态是派生量，不写图；`archived` 的节点照样在快照里、出处也在。
    """

    def _pair(self):
        store = self.mkstore()
        big = remember(store, summary="大事", obs=Observation(angry=3.5, unfinished=3.0),
                       scene="s1", tags=("a",), created_at=0.0, source_quote="原文")
        small = remember(store, summary="小事", obs=Observation(happy=0.3),
                         scene="s1", tags=("b",), created_at=0.0, source_quote="原文")
        return store, big.id, small.id

    def test_three_states_by_salience(self) -> None:
        store, big, small = self._pair()
        p = Params()
        nodes = {n.id: n for n in store.nodes()}
        self.assertEqual(view_of(nodes[big], 0.0, p).state, MemoryState.ACTIVE)
        self.assertEqual(view_of(nodes[small], 0.0, p).state, MemoryState.COLD)
        self.assertEqual(view_of(nodes[small], 3650.0, p).state, MemoryState.ARCHIVED)

    def test_active_also_requires_recent_or_accessed(self) -> None:
        """大事放 365 天没人碰 ⇒ **掉出 active**（防"主动关怀永远是同一件旧事"）。"""

        store, big, _ = self._pair()
        p = Params()
        n = {x.id: x for x in store.nodes()}[big]
        self.assertEqual(view_of(n, 0.0, p).state, MemoryState.ACTIVE)
        self.assertEqual(view_of(n, 365.0, p).state, MemoryState.COLD,
                         "S_now 还高，但 365 天没被用到 ⇒ 不该再占主动关怀")

    def test_access_pulls_it_back_to_active(self) -> None:
        store, big, _ = self._pair()
        p = Params()
        commit_access(store, [big], 365.0, p)
        n = {x.id: x for x in store.nodes()}[big]
        self.assertEqual(view_of(n, 366.0, p).state, MemoryState.ACTIVE, "碰一下就该回来")

    def test_archived_is_not_deleted(self) -> None:
        store, _, small = self._pair()
        p = Params()
        before = freeze_signature(store.nodes())
        v = view_of({x.id: x for x in store.nodes()}[small], 3650.0, p)
        self.assertEqual(v.state, MemoryState.ARCHIVED)
        self.assertIn(small, {x.id for x in store.nodes()}, "归档的节点必须还在库里")
        self.assertEqual(before, freeze_signature(store.nodes()), "状态是派生量，不许写图")


class TestRandomRecall(BaseCase):
    """§19 随机检索器 / 随机跳（teleport）—— 早期 实现。"""
    """§19 随机检索器 / 随机跳（teleport）—— 早期 实现。

    规格（设计 §19.1–19.2）：
    * **随机检索器**：随机撒点当种子 → 走同一套扩散（治"信息茧房"/走神）；
    * **随机跳（teleport）**：扩散中瞬时传送到远处，从那儿继续 —— 三条硬约束必须守住：
      ① **计入同一份预算**（≤ 10% × `n_max`）**且不许连续跳两次**；
      ② **只服务闲聊路**（默认关，事实问答路不许开）；
      ③ **只读不写**。
    * **随机必须带 seed**，否则 §16.5 的"两遍逐位相同"当场失效。
    """

    def _snap(self):
        store = self.mkstore()
        build_bank(store, n_nodes=200, seed=7)
        return store.snapshot()

    # ---- 随机检索器 ----

    def test_random_seeds_are_reproducible_and_seed_dependent(self) -> None:
        snap = self._snap()
        a = random_seed_nodes(snap, 3, seed=42)
        self.assertEqual(a, random_seed_nodes(snap, 3, seed=42), "同 seed 必须一模一样")
        self.assertNotEqual(a, random_seed_nodes(snap, 3, seed=43), "换 seed 应当换一批")
        self.assertEqual(len(a), 3)

    def test_random_recall_uses_its_seeds_and_is_reproducible(self) -> None:
        snap = self._snap()
        r1 = random_recall(snap, 3, 0.5, Params(), seed=42)
        r2 = random_recall(snap, 3, 0.5, Params(), seed=42)
        self.assertEqual(r1.as_dict(), r2.as_dict(), "同参数两遍必须逐位相同")
        for nid in random_seed_nodes(snap, 3, seed=42):
            self.assertIn(nid, r1.activation, "自己的种子必须在激活集合里")

    def test_random_recall_reaches_across_scenes(self) -> None:
        """随机种子通常落在**互不相干的场景**里 —— 这正是治茧房要的效果。"""

        snap = self._snap()
        r = random_recall(snap, 3, 0.5, Params(), seed=42)
        scenes = {r.scene_map.get(nid) for nid in r.activation} - {None}
        self.assertGreaterEqual(len(scenes), 2, "随机撒的点不该全挤在一个场景里（实测 %s）" % scenes)

    # ---- 随机跳（teleport）----

    def test_teleport_is_off_by_default_and_identical_to_legacy(self) -> None:
        snap = self._snap()
        q = Query(terms=("她",), now=0.0)
        r = retrieve(snap, q, 0.5, Params())
        self.assertEqual(r.teleports, 0, "默认必须是关的")
        self.assertEqual(r.seed, 0)

    def test_teleport_respects_budget_and_never_jumps_twice_in_a_row(self) -> None:
        snap = self._snap()
        p = Params(teleport_prob=1.0, teleport_budget_ratio=0.10)
        r = retrieve(snap, Query(terms=("她",), now=0.0, seed=123), 0.5, p)
        budget = max(1, int(p.n_max * p.teleport_budget_ratio))
        self.assertGreater(r.teleports, 0, "概率 1.0 时应当至少跳一次")
        self.assertLessEqual(r.teleports, budget, "跳的次数必须 ≤ 预算 %d" % budget)
        # **不许连续跳**：深度 5 跳、概率 1.0 时，最多只能跳 ceil(5/2) = 3 次
        p2 = Params(teleport_prob=1.0, depth_max=5)
        r2 = retrieve(snap, Query(terms=("她",), now=0.0, seed=7), 0.5, p2)
        self.assertLessEqual(r2.teleports, 3, "连续跳被允许了（实测 %d 次）" % r2.teleports)

    def test_teleport_is_reproducible_and_read_only(self) -> None:
        snap = self._snap()
        p = Params(teleport_prob=0.5)
        q = Query(terms=("她",), now=0.0, seed=2026)
        before = freeze_signature(snap.nodes.values())
        a = retrieve(snap, q, 0.5, p).as_dict()
        b = retrieve(snap, q, 0.5, p).as_dict()
        self.assertEqual(a, b, "带随机的检索也必须逐位可复现（seed 记在结果里）")
        self.assertEqual(before, freeze_signature(snap.nodes.values()), "随机跳不许写图")


class TestSceneSet(BaseCase):
    """§24.9.2「场景集」：**同一个习惯出现在多个场景**时，别只搜一个场景。

    它修的是第六道闸门的副作用：跨出 `seed_scene` 的节点在排序上打 0.10 折，
    而"每周三去游泳"每次都是独立场景 ⇒ "我游泳都在哪天"只能想起 1–2 次。

    三条对应的规矩：
    1. **场景集从查询自己的候选里推**（不是上帝视角 —— 那两个探针当时拿了"答案所在场景"）；
    2. **不必放开跨场景惩罚**：被搜的那个场景自己就是 `seed_scene`，不吃折；
    3. **只读 + 可复现**：跑完图指纹不变，同参数两遍逐位相同。
    """

    def _fixture(self):
        store = self.mkstore()
        build_bank(store, n_nodes=60, seed=7)
        scenes = sorted({n.scene for n in store.snapshot().nodes.values() if n.scene})[:3]
        ids = []
        for k, sc in enumerate(scenes):
            n = remember(store, summary="%s 第 %d 次去游泳" % (sc, k + 1),
                         obs=Observation(happy=0.4, importance=0.6), scene=sc,
                         tags=("hubzz", "habitzz"), modulators=Modulators(),
                         created_at=float(5 + k), source_quote="[t] %d" % k)
            ids.append(n.id)
        return store, scenes, ids

    def test_scene_candidates_are_self_derived_and_include_query_scene(self):
        store, scenes, _ = self._fixture()
        snap = store.snapshot()
        q = Query(terms=("hubzz", "habitzz"), scene=scenes[0], now=0.0)
        got = scene_candidates(snap, q, Params())
        self.assertEqual(got[0], scenes[0], "查询自带的场景必须排第一")
        for sc in scenes:
            self.assertIn(sc, got, "同一习惯所在的场景都该被推出来（%s）" % sc)

    def test_scene_set_recovers_what_single_scene_loses(self):
        store, scenes, ids = self._fixture()
        snap = store.snapshot()
        p = Params()
        q = Query(terms=("hubzz", "habitzz"), scene=scenes[0], now=0.0)

        single = retrieve(snap, q, 0.5, p)
        single_hits = sum(1 for nid in ids if nid in [i for i, _ in single.top(5)])
        multi = retrieve_scene_set(snap, q, 0.5, p)
        multi_hits = sum(1 for nid in ids if nid in [i for i, _ in multi.top(5)])

        self.assertLessEqual(single_hits, multi_hits, "场景集不该比单场景更差")
        self.assertEqual(multi_hits, len(ids), "逐场景并集应当把三次出现都找回来（实测得到 %d/%d）"
                         % (multi_hits, len(ids)))
        self.assertEqual(multi.scenes[0], scenes[0])

    def test_scene_set_is_read_only_and_reproducible(self):
        store, scenes, _ = self._fixture()
        snap = store.snapshot()
        q = Query(terms=("hubzz", "habitzz"), scene=scenes[0], now=0.0)
        before = freeze_signature(snap.nodes.values())
        a = retrieve_scene_set(snap, q, 0.5, Params()).as_dict()
        b = retrieve_scene_set(snap, q, 0.5, Params()).as_dict()
        self.assertEqual(before, freeze_signature(snap.nodes.values()), "检索不许改图（含边）")
        self.assertEqual(a, b, "同参数两遍必须逐位相同")


class TestUseGrowth(BaseCase):
    """§27 复述**累积**权重（调用方 早期 澄清："重复提及会导致权重增加，增加幅度逐级递减"）。

    三条对应的规矩：
    1. **默认关**时 `s_use` 恒为 0 ⇒ 与旧实现逐位相同；
    2. 累积走 `commit_access`（命中路径），**只动 `s_use`，不碰 `s0`/`obs`/`scene`**；
    3. **冻结证明仍然成立** —— `freeze_signature()` 不受 `s_use` 影响（它只证"事实没变"）。
    """

    def test_disabled_by_default_grows_nothing(self):
        p = Params()
        self.assertFalse(p.use_growth_enabled, "新机制必须默认关（opt-in）")
        store = self.mkstore()
        n = store.add_node(kind=NodeKind.EPISODE, summary="x", scene="s1", tags=("a",),
                           obs=Observation(happy=1.0), modulators=Modulators(),
                           created_at=0.0, s0=1.0, base_persist=1.0)
        self.assertEqual(n.s_use, 0.0, "新节点从 0 开始")
        commit_access(store, [n.id], 10.0, p)          # 关着 → 不该涨
        self.assertEqual(store.nodes()[0].s_use, 0.0, "默认关时命中不产生累积")

    def test_commit_access_grows_s_use_but_not_s0(self):
        """命中一次 ⇒ `s_use` 上升，而 `s0` / 观测量**一个字节都不动**。"""

        store = self.mkstore()
        p = Params(use_growth_enabled=True, decay_curve="power")
        n = store.add_node(
            kind=NodeKind.EPISODE, summary="大事", scene="s1", tags=("a",),
            obs=Observation(happy=3.0), modulators=Modulators(),
            created_at=0.0, s0=2.0, base_persist=1.0,
        )
        before = freeze_signature(store.nodes())
        commit_access(store, [n.id], 10.0, p)
        after = store.nodes()[0]
        self.assertGreater(after.s_use, 0.0, "开了累积就该涨")
        self.assertEqual(after.s0, 2.0, "`s0`（当时有多重）必须冻结")
        self.assertEqual(before, freeze_signature(store.nodes()),
                         "冻结指纹不许被 `s_use` 影响（它只证'事实没变'）")

    def test_gate_makes_small_events_barely_grow(self):
        """**按权重分级**：大事每次涨得多，小事几乎不涨（调用方要的"逐级递减趋近 0"）。"""

        store = self.mkstore()
        p = Params(use_growth_enabled=True, decay_curve="power",
                   reset_weight_gate=0.80, reset_gate_power=3.0, reset_repeat_halflife=100.0)
        big = store.add_node(kind=NodeKind.EPISODE, summary="大事", scene="s1", tags=("a",),
                             obs=Observation(happy=3.0), modulators=Modulators(),
                             created_at=0.0, s0=2.0, base_persist=1.0)
        small = store.add_node(kind=NodeKind.EPISODE, summary="小事", scene="s1", tags=("b",),
                               obs=Observation(happy=0.3), modulators=Modulators(),
                               created_at=0.0, s0=0.2, base_persist=1.0)
        for day in (10.0, 20.0, 30.0):
            commit_access(store, [big.id, small.id], day, p)
        nodes = {n.id: n for n in store.nodes()}
        self.assertGreater(nodes[big.id].s_use, 0.1, "大事应当被明显巩固")
        self.assertLess(nodes[small.id].s_use, 0.01, "小事不该被反复提及救活")


class TestRehearsalResetDiscount(BaseCase):
    """§26.1 复述时钟的折扣（调用方 早期 提案）：**小事件碰了也等于没碰，大事才能达成动态平衡**。

    三条对应的规矩：
    1. **默认关时与旧实现逐位相同**（新机制一律 opt-in，不在默认路径上改行为）；
    2. **权重闸门**：`S₀` 低于 `reset_weight_gate` 的节点，复述几乎不拨回时钟；
    3. **逐级递减**：同一个节点被碰得越多，单次拨回幅度越小。
    """

    def _p(self, **kw) -> Params:
        base = dict(
            decay_curve="power",
            reset_discount_enabled=True,
            reset_weight_gate=0.80,
            reset_repeat_halflife=100.0,
            reset_gate_power=3.0,
        )
        base.update(kw)
        return Params(**base)

    def test_default_off_is_bit_identical_to_legacy(self):
        """开关关掉时，`recency` 必须与旧公式 `max(created_at, last_access)` **逐位相同**。"""

        p = Params(decay_curve="power")  # 默认：reset_discount_enabled=False
        for created, last, s0, n in ((0.0, 360.0, 0.2, 5), (10.0, 200.0, 2.0, 1), (0.0, 0.0, 0.5, 0)):
            anchor = max(created, last)
            age = max(0.0, 365.0 - anchor)
            legacy = max(1e-9, min(1.0, (1.0 + age / p.power_tau) ** (-p.power_beta)))
            got = recency_of(created, 365.0, last_access=last, s0=s0, access_count=n, p=p)
            self.assertEqual(got, legacy, "关掉开关时必须逐位相同")

    def test_weight_gate_keeps_small_events_forgettable(self):
        """小事（`S₀` 低于闸门）被碰一次 ⇒ `recency` 与**从没被碰过**几乎一样。"""

        p = self._p()
        never = recency_of(0.0, 365.0, last_access=0.0, s0=0.2, access_count=0, p=p)
        touched = recency_of(0.0, 365.0, last_access=360.0, s0=0.2, access_count=1, p=p)
        self.assertLess(touched / never, 1.10, "小事被碰一次不该明显变牢（闸门没起作用）")

    def test_weight_gate_lets_big_events_reach_equilibrium(self):
        """大事（`S₀` 过闸门）被碰一次 ⇒ `recency` 显著抬高（能对抗遗忘）。"""

        p = self._p()
        never = recency_of(0.0, 365.0, last_access=0.0, s0=2.0, access_count=0, p=p)
        touched = recency_of(0.0, 365.0, last_access=360.0, s0=2.0, access_count=1, p=p)
        self.assertGreater(touched / never, 5.0, "大事被反复提起应当能维持（否则等于把复述整个关掉）")

    def test_diminishing_returns_on_repeats(self):
        """逐级递减：同一个大事被碰 20 次，单次拨回幅度小于只碰 1 次。"""

        p = self._p()
        once = recency_of(0.0, 365.0, last_access=360.0, s0=2.0, access_count=1, p=p)
        many = recency_of(0.0, 365.0, last_access=360.0, s0=2.0, access_count=20, p=p)
        self.assertLess(many, once, "碰得越多，单次拨回应当越小")


class TestForgetAndPrivacy(BaseCase):
    """代码审查 早期 第 3 条：调用方要有"让她忘掉"的路径，凭据不许落盘。

    硬规矩：删除是**显式写入**，不是检索的副作用 ⇒ `retrieve()` 仍是纯函数。
    """

    def _bank(self):
        store = self.mkstore()
        a = remember(store, summary="在测试场景甲丢了脸", obs=Observation(shame=0.9),
                     scene="场景A", tags=("测试场景甲", "丢脸"), created_at=0.0,
                     source_quote="我在测试场景甲摔了一跤")
        b = remember(store, summary="在码头喂鸟吃测试物品", obs=Observation(happy=0.6),
                     scene="场景B", tags=("测试物品", "鸟"), created_at=1.0,
                     source_quote="码头的鸟很可爱")
        c = remember(store, summary="又在测试场景甲吃测试物品", obs=Observation(happy=0.4),
                     scene="场景C", tags=("测试场景甲", "测试物品"), created_at=2.0,
                     source_quote="今天又去了测试场景甲")
        return store, a, b, c

    def test_forget_removes_node_edges_tags_and_source_quote(self) -> None:
        store, a, b, c = self._bank()
        self.assertTrue(any(e.src == a.id or e.dst == a.id for e in store.snapshot().edges))
        out = store.forget([a.id], now=10.0)
        self.assertEqual(out["deleted"], [a.id])
        snap = store.snapshot()
        self.assertNotIn(a.id, set(snap.nodes))
        self.assertFalse(any(e.src == a.id or e.dst == a.id for e in snap.edges), "边要一起抹")
        self.assertEqual(store.node_ids_with_any_tag(("丢脸",)), [], "标签要一起抹")
        dump = " ".join(repr(r) for t in ("nodes", "edges", "tags", "tombstones")
                        for r in store.conn.execute(f"SELECT * FROM {t}").fetchall())
        self.assertNotIn("摔了一跤", dump, "出处原文不许残留在任何表里")
        self.assertNotIn("丢了脸", dump)

    def test_tombstone_keeps_no_content(self) -> None:
        store, a, *_ = self._bank()
        store.forget([a.id], now=10.0, reason="user_request")
        ts = store.tombstones()
        self.assertEqual(len(ts), 1)
        self.assertEqual(set(ts[0]), {"node_id", "kind", "deleted_at", "reason"})
        self.assertEqual(ts[0]["node_id"], a.id)

    def test_forget_leaves_other_nodes_frozen_and_untouched(self) -> None:
        store, a, b, c = self._bank()
        before = freeze_signature([n for k, n in store.snapshot().nodes.items() if k != a.id])
        store.forget([a.id], now=10.0)
        self.assertEqual(before, freeze_signature(store.snapshot().nodes.values()),
                         "删一个节点不许改动别的节点的冻结量")

    def test_retrieval_never_returns_forgotten_and_stays_pure(self) -> None:
        store, a, b, c = self._bank()
        store.forget([a.id], now=10.0)
        snap = store.snapshot()
        sig = (freeze_signature(snap.nodes.values()), tuple((e.src, e.dst, e.weight) for e in snap.edges))
        r = retrieve(snap, Query(terms=("测试场景甲",), now=10.0), 1.0, Params(beta_mode="hop"))
        self.assertNotIn(a.id, r.activation)
        self.assertIn(c.id, r.activation)
        snap2 = store.snapshot()
        self.assertEqual(sig, (freeze_signature(snap2.nodes.values()),
                               tuple((e.src, e.dst, e.weight) for e in snap2.edges)),
                         "检索前后图指纹必须一致（删除是显式写入，检索仍是纯函数）")

    def test_cascade_deletes_derived_only_when_all_sources_gone(self) -> None:
        store, a, b, c = self._bank()
        only_a = reflect(store, summary="他在人前容易难堪", sources=[a], created_at=3.0)
        mixed = reflect(store, summary="他喜欢测试物品", sources=[a, c], created_at=3.0)
        out = store.forget([a.id], now=10.0)
        ids = set(store.snapshot().nodes)
        self.assertIn(only_a.id, out["cascaded"])
        self.assertNotIn(only_a.id, ids, "只剩被删内容转述的派生节点要一并删")
        self.assertIn(mixed.id, ids, "还有存活来源的派生节点不替调用方猜，保留")
        self.assertEqual(out["tainted"], [mixed.id], "但必须告诉调用方它可能还带着被删内容")

    def test_cascade_can_be_turned_off(self) -> None:
        store, a, b, c = self._bank()
        only_a = reflect(store, summary="他在人前容易难堪", sources=[a], created_at=3.0)
        out = store.forget([a.id], now=10.0, cascade_derived=False)
        self.assertEqual(out["cascaded"], [])
        self.assertIn(only_a.id, set(store.snapshot().nodes))

    def test_forget_by_tags_and_unknown_ids_are_harmless(self) -> None:
        store, a, b, c = self._bank()
        out = store.forget_by_tags(["测试场景甲"], now=10.0)
        self.assertEqual(out["deleted"], sorted([a.id, c.id]))
        self.assertEqual(store.forget([9999], now=11.0)["deleted"], [])
        self.assertEqual(len(store.tombstones()), 2, "不存在的 id 不留墓碑")

    def test_forgotten_content_is_gone_from_the_db_file(self) -> None:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.db")
            store = MemoryStore(path)
            n = remember(store, summary="独一无二的秘密小事XQZJ", obs=Observation(shame=0.8),
                         scene="场景Z", tags=("秘密",), created_at=0.0,
                         source_quote="独一无二的原话XQZJ")
            remember(store, summary="另一件事", obs=Observation(happy=0.3), scene="场景Y",
                     tags=("别的",), created_at=1.0)
            store.forget([n.id], now=5.0)
            store.scrub()
            store.close()
            raw = b""
            for suffix in ("", "-wal", "-journal"):
                if os.path.exists(path + suffix):
                    with open(path + suffix, "rb") as fh:
                        raw += fh.read()
            self.assertNotIn("XQZJ".encode(), raw, "删完 + scrub 之后，文件里不许再有这段内容")

    def test_sensitive_content_is_rejected_and_not_written(self) -> None:
        store = self.mkstore()
        for bad in ("我的 password: hunter2 别告诉别人", "key sk-abcdefghijklmnopqrstuv",
                    "卡号 6222021234567890123"):
            with self.assertRaises(SensitiveContentError) as cm:
                remember(store, summary=bad, obs=Observation(), scene="s", tags=("x",))
            self.assertNotIn("hunter2", str(cm.exception), "异常信息里不许带原文")
        with self.assertRaises(SensitiveContentError):
            remember(store, summary="正常摘要", obs=Observation(), scene="s", tags=("x",),
                     source_quote="token=abcdef123456")
        self.assertEqual(len(store.snapshot().nodes), 0, "被拒的内容一条都不许落盘")

    def test_ordinary_text_passes_and_check_can_be_disabled(self) -> None:
        store = self.mkstore()
        remember(store, summary="测试日期事件", obs=Observation(happy=0.8),
                 scene="s", tags=("生日",))
        remember(store, summary="password: x", obs=Observation(), scene="s", tags=("t",),
                 reject_sensitive=False)
        self.assertEqual(len(store.snapshot().nodes), 2)

    # ---- 早期 加固：三个漏洞各自的回归（评审自查发现）----

    def test_id_card_with_X_suffix_is_caught(self) -> None:
        """漏洞①：末位是 **X** 的身份证原来整条漏掉（`\\b` 在 X 前没有边界）。"""

        self.assertEqual(looks_sensitive("身份证号 99999919000101100X 别外传"), "id_card")
        self.assertEqual(looks_sensitive("999999190001011237"), "id_card")   # 全数字的合法号
        # 反例：11 位手机号不是身份证，不许误判
        self.assertIsNone(looks_sensitive("我的手机号 13800138000"))

    def test_epoch_timestamps_are_not_false_positives(self) -> None:
        """漏洞②：16 位微秒时间戳原来被误杀（`\\d{15,19}` 一刀切）。"""

        self.assertIsNone(looks_sensitive("日志时间 1728451200123456 写入完成"), "微秒时间戳该放过")
        self.assertIsNone(looks_sensitive("ts=1728451200123"), "毫秒时间戳该放过")
        # 但**真卡号**不许放过：Luhn 通过 ⇒ 卡号；有语境词 ⇒ 卡号
        self.assertEqual(looks_sensitive("4111111111111111"), "card_luhn")
        self.assertEqual(looks_sensitive("卡号 6222021234567890123"), "card_context")
        self.assertEqual(looks_sensitive("这是身份证，号码 999999190001011237"), "id_card")

    def test_artifact_scan_and_scrub(self) -> None:
        """漏洞③：`forget()` 只覆盖 SQLite；**打标缓存里的原文**要靠 `scrub_files` 擦掉。"""

        import tempfile
        from pathlib import Path

        from memlayer import find_text_in_files, scrub_files

        needle = "中午本来只是想去买测试物品"
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "external_label_cache.jsonl"
            cache.write_text(
                '{"text": "无关的一条"}\n'
                '{"text": "中午本来只是想去买测试物品"}\n'
                '{"text": "另一条无关的"}\n',
                encoding="utf-8",
            )
            other = Path(td) / "other.json"
            other.write_text('{"ok": 1}\n', encoding="utf-8")

            # 先只读扫：应当只报出缓存那一个文件、命中 1 行
            found = find_text_in_files([cache, other], [needle])
            self.assertEqual(found, {str(cache): 1})

            # dry_run 不动盘
            dry = scrub_files([cache], [needle], dry_run=True)
            self.assertEqual(dry[str(cache)]["removed"], 1)
            self.assertIn(needle, cache.read_text(encoding="utf-8"), "dry_run 不许写盘")

            # 真擦：命中行消失、其它行保留、留 `.bak`
            report = scrub_files([cache], [needle])
            self.assertEqual(report[str(cache)]["removed"], 1)
            self.assertEqual(report[str(cache)]["kept"], 2)
            body = cache.read_text(encoding="utf-8")
            self.assertNotIn(needle, body)
            self.assertIn("无关的一条", body)
            bak = Path(report[str(cache)]["backup"])
            self.assertTrue(bak.exists(), "默认要留 .bak（可恢复）")
            self.assertIn(needle, bak.read_text(encoding="utf-8"))


class TestRuntimeHook(BaseCase):
    """对话运行时接缝（方案 v1，示例 runtime 接线前的全部逻辑）。"""

    def test_remember_turn_writes_and_skips(self) -> None:
        from runtime_hook import remember_turn

        store = self.mkstore()
        n = remember_turn(store, text="测试事件B：在会议上被当众指出错误，有点生气", now=1000.0)
        self.assertIsNotNone(n)
        self.assertGreater(n.obs.angry, 0.0, "规则版应当识别到愤怒词")
        self.assertTrue(n.modulators.public, "「当众」应当打开 public 调制器")
        self.assertEqual(n.obs_source, "runtime_rule")
        self.assertIn("月", n.scene or "", "运行时场景键 = 「M月D日·时段」")
        self.assertIsNone(remember_turn(store, text="嗯", now=1001.0), "过短不写")
        # 隐私闸：合法身份证号 → 拒写
        self.assertIsNone(remember_turn(store, text="顺手存一下 999999190001011237 别忘了", now=1002.0))
        self.assertEqual(len(store.nodes()), 1)

    def test_recall_gate_and_injection(self) -> None:
        from runtime_hook import recall_context, remember_turn

        store = self.mkstore()
        remember_turn(store, text="测试事件A：在游泳馆完成第二次练习", now=1000.0)
        self.assertIsNone(recall_context(store, user_text="今天天气不错", now=2000.0),
                          "无触发词不召回")
        hit = recall_context(store, user_text="上次游泳馆那事怎么样了", now=2000.0)
        self.assertIsNotNone(hit)
        self.assertIn("游泳馆", hit["text"])
        self.assertEqual(hit["path"], "fact")
        self.assertIn("不要编造", hit["text"], "注入模板要带防幻觉提醒")

    def test_query_terms_drop_stop_chars(self) -> None:
        from runtime_hook import extract_query_terms, should_recall

        terms = extract_query_terms("上次我们一起去的那个游泳馆")
        self.assertNotIn("上", terms)
        self.assertNotIn("的", terms)
        self.assertIn("游", terms)
        self.assertTrue(should_recall("你还记得那件事吗"))
        self.assertFalse(should_recall("帮我定个闹钟"))

    def test_query_expansion_adds_paraphrase_terms(self) -> None:
        from runtime_hook import extract_query_terms

        plain = extract_query_terms("她心情很差，我陪到很晚")
        expanded = extract_query_terms("她心情很差，我陪到很晚", expand=True)
        self.assertNotIn("难", plain)
        self.assertIn("难", expanded)
        self.assertIn("半", expanded)
        self.assertIn("夜", expanded)


class TestParamProfiles(BaseCase):
    """查询分路双档（§16.3 落地）：fact=相关性+最近性主导 / companion=现行默认。"""

    def test_profile_mechanics(self) -> None:
        base = Params()
        fact = params_for_path("fact", base)
        self.assertEqual((fact.w_importance, fact.w_salience), (0.0, 0.0))
        self.assertEqual(fact.w_relevance, base.w_relevance)
        self.assertEqual(fact.w_recency, base.w_recency)   # 最近性保留（"最近那次"排序）
        self.assertEqual(fact.beta_mode, base.beta_mode)
        self.assertEqual(params_for_path("companion"), base)  # 关怀档 = 现行默认
        self.assertEqual(params_for_path("default"), base)
        with self.assertRaises(ValueError):
            params_for_path("unknown_path")
        self.assertEqual(base.w_importance, 0.6, "不许改到 base 本体（frozen dataclass）")

    def test_fact_path_demotes_narrative_value(self) -> None:
        """行为用例：高显著度的不相关事件（默认档压过）在事实档让位给相关事件。"""

        store = self.mkstore()
        n_rel = remember(store, summary="在图书馆睡着，醒来外面在下雨", obs=Observation(happy=0.1),
                         created_at=200.0, tags=("图书馆", "睡着"))
        n_hot = remember(store, summary="回想起测试事件，身体有反应",
                         obs=Observation(angry=1.0, shame=1.0, arousal=1.0, importance=0.9),
                         created_at=250.0, tags=("手抖",))
        snap = store.snapshot()
        q = Query(terms=("图书馆", "睡着", "手抖"), now=300.0)
        top_default = [i for i, _ in retrieve(snap, q, 0.5, params_for_path("companion")).top(2)]
        top_fact = [i for i, _ in retrieve(snap, q, 0.5, params_for_path("fact")).top(2)]
        self.assertEqual(top_default[0], n_hot.id, "默认档：显著度高的叙事事件压过相关事件")
        self.assertEqual(top_fact[0], n_rel.id, "事实档：相关事件回到第一位")


class TestSelfTrace(BaseCase):
    """§22 自我轨迹（辅网 v0）：三条硬约束（不进检索/写完冻结/幂等）+ 提案-批准路 + 成本轴。"""

    def test_append_only_and_rejected_keeps_top2(self) -> None:
        store = self.mkstore()
        st = SelfTrace(store)
        s1 = st.record_retrieval(ts=1.0, query_terms=("测试场景甲",), d=0.5, params=Params(),
                                 produced_ids=(2, 1), rejected_ids=(3, 4, 5),
                                 activation_ids=(1, 2, 3), scene="吃饭")
        s2 = st.record_retrieval(ts=2.0, query_terms=("测试物品",), d=0.5, params=Params(),
                                 produced_ids=(1,), rejected_ids=())
        self.assertEqual((s1, s2), (1, 2))
        rows = store.trace_rows()
        self.assertEqual(rows[0]["payload"]["rejected"], [3, 4], "反事实只记 top-2（§22.8-3）")
        self.assertEqual(rows[0]["payload"]["produced"], [2, 1])
        # 结构性的 append-only：store 上没有 update/delete 轨迹的入口
        for bad in ("update_trace", "delete_trace", "remove_trace", "edit_trace"):
            self.assertFalse(hasattr(store, bad), bad)

    def test_freeze_under_main_net_ops(self) -> None:
        store = self.mkstore()
        st = SelfTrace(store)
        st.record_retrieval(ts=1.0, query_terms=("a",), d=0.5, params=Params(),
                            produced_ids=(1,), activation_ids=(1, 2))
        st.propose_cognition(ts=2.0, summary="测试提案A", evidence_ids=())
        sig_before = store.trace_signature()
        n1 = remember(store, summary="今天吃了测试场景甲", obs=Observation(happy=0.6),
                      created_at=10.0, tags=("吃", "测试场景甲"))
        n2 = remember(store, summary="上周游了泳", obs=Observation(happy=0.4),
                      created_at=11.0, tags=("运动",))
        store.add_edge(src=n1.id, dst=n2.id, kind=EdgeKind.CO_OCCUR, weight=2.0)
        store.touch_access([n1.id], now=12.0)
        store.add_s_use([(n1.id, 0.3)])
        store.strengthen_edges([n1.id, n2.id], 0.5)
        store.decay_edges(0.9)
        self.assertEqual(store.trace_signature(), sig_before, "主网写操作不许碰轨迹（写完冻结）")

    def test_not_in_retrieval(self) -> None:
        sA = self.mkstore()
        build_bank(sA, n_nodes=60)
        sB = self.mkstore()
        build_bank(sB, n_nodes=60)
        st = SelfTrace(sB)
        st.record_retrieval(ts=1.0, query_terms=("测试场景甲",), d=0.6, params=Params(),
                            produced_ids=(1, 2), rejected_ids=(3,))
        st.propose_cognition(ts=2.0, summary="x", evidence_ids=(1,))
        q = Query(terms=("测试场景甲",), now=3.0)
        p = Params(beta_mode="hop")
        rA = retrieve(sA.snapshot(), q, 0.6, p)
        rB = retrieve(sB.snapshot(), q, 0.6, p)
        key = lambda r: (tuple(r.seeds), dict(r.activation), dict(r.depth),
                         r.expanded_raw, r.pruned, r.capped)
        self.assertEqual(key(rA), key(rB), "有/无轨迹行 ⇒ 检索结果必须逐位相同")
        self.assertEqual(len(sA.nodes()), len(sB.nodes()), "提案（未批准）不许进主网")

    def test_proposal_needs_approval(self) -> None:
        store = self.mkstore()
        st = SelfTrace(store)
        e1 = remember(store, summary="测试事件C", obs=Observation(anxiety=0.6),
                      created_at=5.0, tags=("熬夜",))
        before = len(store.nodes())
        seq = st.propose_cognition(ts=6.0, summary="测试提案B",
                                   evidence_ids=(e1.id,), obs=Observation(anxiety=0.3))
        self.assertEqual(len(store.nodes()), before, "提案阶段主网不许变")
        node = st.approve_proposal(seq, created_at=7.0)
        self.assertEqual(len(store.nodes()), before + 1)
        self.assertEqual(node.kind.value, "semantic")
        derived = {(e.src, e.dst) for e in store.edges() if e.kind == EdgeKind.DERIVED_FROM}
        self.assertIn((e1.id, node.id), derived, "批下来的认知必须挂 derived_from 指回证据")
        self.assertEqual(store.trace_rows()[-1]["kind"], "approval")

    def test_idempotent_replay(self) -> None:
        def build() -> MemoryStore:
            s = self.mkstore()
            build_bank(s, n_nodes=40)
            st = SelfTrace(s)
            st.record_retrieval(ts=1.0, query_terms=("测试场景甲",), d=0.5, params=Params(),
                                produced_ids=(3, 1), rejected_ids=(2,), activation_ids=(1, 2, 3))
            st.propose_cognition(ts=2.0, summary="s", evidence_ids=(1, 2))
            return s

        self.assertEqual(build().trace_signature(), build().trace_signature())

    def test_change_filter_skips_duplicates(self) -> None:
        store = self.mkstore()
        st = SelfTrace(store)
        a = st.record_retrieval_if_changed(ts=1.0, query_terms=("测试场景甲",), d=0.5,
                                           params=Params(), produced_ids=(3, 1),
                                           rejected_ids=(2,))
        b = st.record_retrieval_if_changed(ts=2.0, query_terms=("测试场景甲",), d=0.5,
                                           params=Params(), produced_ids=(3, 1),
                                           rejected_ids=(2,))
        self.assertEqual(a, 1)
        self.assertIsNone(b, "结论没变 ⇒ 不记转变点（§22.2-2）")
        c = st.record_retrieval_if_changed(ts=3.0, query_terms=("测试场景甲",), d=0.5,
                                           params=Params(), produced_ids=(1, 3),
                                           rejected_ids=(2,))
        self.assertEqual(c, 2, "顺序变化也算结论变化")
        self.assertEqual(len(store.trace_rows()), 2)

    def test_narrative_deterministic_and_budgeted(self) -> None:
        store = self.mkstore()
        st = SelfTrace(store)
        e1 = remember(store, summary="测试事件C", obs=Observation(anxiety=0.6),
                      created_at=5.0)
        st.record_retrieval(ts=1.0, query_terms=("熬夜",), d=0.5, params=Params(),
                            produced_ids=(e1.id,), rejected_ids=())
        seq = st.propose_cognition(ts=2.0, summary="测试提案B",
                                   evidence_ids=(e1.id,))
        st.approve_proposal(seq, created_at=3.0)
        st.record_cognition_edit(ts=4.0, node_id=e1.id, reason="新证据：他这周早睡了")
        n1 = st.narrative()
        n2 = st.narrative()
        self.assertEqual(n1["text"], n2["text"], "叙事摘要必须确定性（同输入同输出）")
        self.assertEqual(n1["n_records"], 4)
        for marker in ("检索", "提案", "批准", "认知微调"):
            self.assertIn(marker, n1["text"])
        self.assertGreater(n1["est_tokens"], 0)
        small = st.narrative(max_chars=60)
        self.assertLessEqual(small["chars"], 60)
        self.assertGreaterEqual(small["n_rendered"], 1)
        self.assertLess(small["n_rendered"], 4, "预算截断应生效")

    def test_cost_report_storage_axis(self) -> None:
        store = self.mkstore()
        build_bank(store, n_nodes=40)
        st = SelfTrace(store)
        for t in range(1, 6):
            st.record_retrieval(ts=float(t), query_terms=("测试场景甲",), d=0.5, params=Params(),
                                produced_ids=(t,), rejected_ids=(t + 1,),
                                activation_ids=(t, t + 1))
        rep = st.cost_report()
        self.assertEqual(rep["n_trace"], 5)
        self.assertGreater(rep["n_nodes"], 5)
        self.assertIsNotNone(rep["storage_ratio"])
        self.assertLess(rep["storage_ratio"], 1.0, "轨迹应比节点便宜（存储轴）")


class TestMemoryBackup(BaseCase):
    """更新前导出、出问题可恢复；快照必须覆盖主网 + 辅网全部表。"""

    @staticmethod
    def _seed(path: Path) -> MemoryStore:
        store = MemoryStore(path)
        e1 = remember(store, summary="测试事件D",
                      obs=Observation(happy=0.8, importance=0.7),
                      created_at=1.0, tags=("蛋糕", "第一次"))
        e2 = remember(store, summary="测试事件E",
                      obs=Observation(worry=0.4), created_at=2.0, tags=("蛋糕",))
        e3 = remember(store, summary="我们一起吃了晚饭",
                      obs=Observation(happy=0.3), created_at=3.0, tags=("晚饭",))
        store.add_edge(src=e1.id, dst=e3.id, kind=EdgeKind.CO_OCCUR, weight=0.7)
        store.touch_access([e1.id], now=4.0)
        store.add_s_use([(e1.id, 0.2)])
        store.forget([e2.id], now=5.0, reason="backup-test")
        SelfTrace(store).record_retrieval(
            ts=6.0, query_terms=("蛋糕",), d=0.5, params=Params(),
            produced_ids=(e1.id,), rejected_ids=(e2.id,), activation_ids=(e1.id, e3.id))
        return store

    @staticmethod
    def _signature(store: MemoryStore) -> dict:
        return {
            "nodes": store.nodes(),
            "edges": store.edges(),
            "tombstones": store.tombstones(),
            "trace": store.trace_rows(),
            "counts": store.counts(),
        }

    def test_export_verify_restore_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "memory.sqlite3"
            snap = root / "memory.snapshot.sqlite3"
            target = root / "restored.sqlite3"
            source = self._seed(db)
            before = self._signature(source)
            source.close()
            source_bytes = db.read_bytes()

            result = export_memory(db, snap)
            self.assertEqual(result["integrity"], "ok")
            self.assertEqual(result["counts"], before["counts"])
            self.assertEqual(db.read_bytes(), source_bytes, "导出不许改源数据库")

            verified = verify_memory_export(snap)
            self.assertTrue(verified["verified"])
            self.assertEqual(verified["format"], "lunavender-memory-snapshot/v1")

            restore_memory(snap, target)
            restored = MemoryStore(target)
            try:
                self.assertEqual(self._signature(restored), before,
                                 "节点/边/标签/墓碑/自我轨迹必须逐项恢复")
            finally:
                restored.close()

    def test_in_memory_store_can_export_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td) / "memory.snapshot.sqlite3"
            store = self.mkstore()
            remember(store, summary="一条内存库记忆", obs=Observation(happy=0.5),
                     created_at=1.0, tags=("内存库",))
            result = store.export_snapshot(snap)
            self.assertEqual(result["counts"], store.counts())
            self.assertEqual(inspect_snapshot(snap)["counts"], store.counts())

    def test_cli_export_verify_restore(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "memory.sqlite3"
            snap = root / "memory.snapshot.sqlite3"
            target = root / "restored.sqlite3"
            source = self._seed(db)
            before = self._signature(source)
            source.close()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(backup_main(
                    ["export", "--db", str(db), "--out", str(snap)]), 0)
                self.assertEqual(backup_main(
                    ["verify", "--snapshot", str(snap)]), 0)
                self.assertEqual(backup_main(
                    ["restore", "--snapshot", str(snap), "--out", str(target)]), 0)
            restored = MemoryStore(target)
            try:
                self.assertEqual(self._signature(restored), before)
            finally:
                restored.close()

    def test_tampered_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "memory.sqlite3"
            snap = root / "memory.snapshot.sqlite3"
            source = self._seed(db)
            source.close()
            export_memory(db, snap)
            with snap.open("ab") as f:
                f.write(b"tampered")
            with self.assertRaises(BackupVerificationError):
                verify_memory_export(snap)

    def test_restore_refuses_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "memory.sqlite3"
            snap = root / "memory.snapshot.sqlite3"
            target = root / "occupied.sqlite3"
            source = self._seed(db)
            source.close()
            export_memory(db, snap)
            target.write_bytes(b"do not overwrite")
            with self.assertRaises(FileExistsError):
                restore_memory(snap, target)
            self.assertEqual(target.read_bytes(), b"do not overwrite")


class TestReflectionGate(BaseCase):
    """§25.1 主动反思三闸门：反事实 × 复现 × 配额/冷却；默认关。"""

    @staticmethod
    def policy(**overrides) -> ReflectionPolicy:
        base = dict(enabled=True, min_counterfactual_chars=8, min_recurrence=0.50,
                    daily_limit=3, source_cooldown_days=7.0)
        base.update(overrides)
        return ReflectionPolicy(**base)

    def seed(self, store: MemoryStore, n: int = 8) -> list[int]:
        ids: list[int] = []
        for i in range(n):
            node = remember(store, summary=f"事件 {i}", obs=Observation(happy=0.2),
                            created_at=float(i), tags=(f"t{i}",))
            ids.append(node.id)
        return ids

    def test_default_disabled_has_no_side_effect(self) -> None:
        store = self.mkstore()
        ids = self.seed(store)
        before = store.trace_signature()
        d = ReflectionGate(store).evaluate(
            ts=1.0, candidate=ReflectionCandidate((ids[0],), "下次先问他现在要不要说", 0.9))
        self.assertFalse(d.allowed)
        self.assertEqual((d.stage, d.reason), ("disabled", "policy_disabled"))
        self.assertEqual(store.trace_signature(), before, "默认关时不写轨迹")

    def test_counterfactual_and_recurrence_filters(self) -> None:
        store = self.mkstore()
        ids = self.seed(store)
        gate = ReflectionGate(store, self.policy())

        no_ev = gate.evaluate(
            ts=1.0, candidate=ReflectionCandidate((), "下次先问他现在要不要说", 0.9))
        self.assertEqual((no_ev.stage, no_ev.reason), ("evidence", "no_evidence"))
        unknown = gate.evaluate(
            ts=1.0, candidate=ReflectionCandidate((9999,), "下次先问他现在要不要说", 0.9))
        self.assertEqual((unknown.stage, unknown.reason), ("evidence", "unknown_evidence"))
        short = gate.evaluate(
            ts=1.0, candidate=ReflectionCandidate((ids[0],), "先问他", 0.9))
        self.assertEqual((short.stage, short.reason),
                         ("counterfactual", "counterfactual_too_short"))
        abstract = gate.evaluate(
            ts=1.0, candidate=ReflectionCandidate((ids[0],), "她一直在想那件事", 0.9))
        self.assertEqual((abstract.stage, abstract.reason),
                         ("counterfactual", "counterfactual_abstract"))
        low = gate.evaluate(
            ts=1.0, candidate=ReflectionCandidate((ids[0],), "下次先问他现在要不要说", 0.2))
        self.assertEqual((low.stage, low.reason),
                         ("recurrence", "recurrence_below_threshold"))
        self.assertFalse(any(d.allowed for d in (no_ev, unknown, short, abstract, low)))

    def test_accepts_concrete_recurring_case(self) -> None:
        store = self.mkstore()
        ids = self.seed(store)
        d = ReflectionGate(store, self.policy()).evaluate(
            ts=1.5, candidate=ReflectionCandidate((ids[1], ids[0]), "下次先问他现在要不要说", 0.8))
        self.assertTrue(d.allowed)
        self.assertEqual((d.stage, d.reason), ("allow", "optimization_value"))
        self.assertAlmostEqual(d.optimization_value, 0.8)
        self.assertEqual(d.evidence_key, tuple(sorted((ids[0], ids[1]))),
                         "证据集按集合处理，顺序不影响冷却键")
        self.assertEqual((d.used_today, d.remaining_today), (1, 2))
        self.assertAlmostEqual(d.cooldown_until, 8.5)

    def test_daily_quota_rolls_over(self) -> None:
        store = self.mkstore()
        ids = self.seed(store, 5)
        gate = ReflectionGate(store, self.policy(daily_limit=2, source_cooldown_days=0.0))
        a = gate.evaluate(ts=1.1, candidate=ReflectionCandidate((ids[0],), "下次先问他现在要不要说", 0.9))
        b = gate.evaluate(ts=1.2, candidate=ReflectionCandidate((ids[1],), "下次先等十分钟再回答", 0.8))
        c = gate.evaluate(ts=1.3, candidate=ReflectionCandidate((ids[2],), "下次先确认他是否想听建议", 0.7))
        d = gate.evaluate(ts=2.1, candidate=ReflectionCandidate((ids[2],), "下次先确认他是否想听建议", 0.7))
        self.assertEqual([x.allowed for x in (a, b, c, d)], [True, True, False, True])
        self.assertEqual((c.stage, c.reason), ("quota", "daily_quota"))
        self.assertEqual(c.remaining_today, 0)

    def test_source_cooldown_is_order_insensitive(self) -> None:
        store = self.mkstore()
        ids = self.seed(store, 5)
        gate = ReflectionGate(store, self.policy(daily_limit=5, source_cooldown_days=7.0))
        first = gate.evaluate(
            ts=1.0, candidate=ReflectionCandidate((ids[0], ids[1]), "下次先问他现在要不要说", 0.9))
        dup = gate.evaluate(
            ts=7.9, candidate=ReflectionCandidate((ids[1], ids[0]), "下次先等他讲完再回应", 0.9))
        later = gate.evaluate(
            ts=8.0, candidate=ReflectionCandidate((ids[0], ids[1]), "下次先等他讲完再回应", 0.9))
        self.assertTrue(first.allowed)
        self.assertEqual((dup.stage, dup.reason), ("cooldown", "source_cooldown"))
        self.assertFalse(dup.allowed)
        self.assertTrue(later.allowed, "冷却到期即可再次反思")

    def test_gate_does_not_mutate_main_net_or_retrieval(self) -> None:
        sA = self.mkstore()
        sB = self.mkstore()
        build_bank(sA, n_nodes=60)
        build_bank(sB, n_nodes=60)
        ids = [n.id for n in sB.nodes()]
        sig_a = freeze_signature(sA.nodes())
        nodes_a = len(sA.nodes())
        gate = ReflectionGate(sB, self.policy())
        d = gate.evaluate(
            ts=1.0, candidate=ReflectionCandidate(tuple(ids[:2]), "下次先问他现在要不要说", 0.9))
        self.assertTrue(d.allowed)
        self.assertEqual(freeze_signature(sB.nodes()), sig_a,
                         "闸门不许改主网冻结量")
        self.assertEqual(len(sB.nodes()), nodes_a)
        q = Query(terms=("测试场景甲",), now=5.0)
        rA = retrieve(sA.snapshot(), q, 0.5, Params(beta_mode="hop"))
        rB = retrieve(sB.snapshot(), q, 0.5, Params(beta_mode="hop"))
        key = lambda r: (tuple(r.seeds), dict(r.activation), dict(r.depth),
                         r.expanded_raw, r.pruned, r.capped)
        self.assertEqual(key(rA), key(rB), "闸门轨迹不许参与检索打分")
        self.assertEqual(sig_a, freeze_signature(sA.nodes()))

    def test_trace_stores_fingerprint_not_counterfactual_text(self) -> None:
        store = self.mkstore()
        ids = self.seed(store)
        text = "下次先问他现在要不要说"
        ReflectionGate(store, self.policy()).evaluate(
            ts=1.0, candidate=ReflectionCandidate((ids[0],), text, 0.9))
        row = store.trace_rows()[-1]
        self.assertEqual(row["kind"], "reflection_gate")
        self.assertNotIn(text, str(row["payload"]), "闸门轨迹不复制候选原文")
        self.assertEqual(len(row["payload"]["counterfactual_fp"]), 16)
        self.assertEqual(row["payload"]["counterfactual_chars"], len(text))
        nar = SelfTrace(store).narrative()
        self.assertIn("反思闸门", nar["text"])
        self.assertIn("放行", nar["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
