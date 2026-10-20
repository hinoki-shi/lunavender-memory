# Lunavender Memory

**夕苓记忆层 / Lunavender Memory** 是一个小型、可复核的长期记忆实验库。

> 状态：**research preview**。这是一个仍在迭代的研究原型，不是生产级记忆服务。
> 仓库只包含核心代码、合成示例和测试；不包含任何真实对话、个人标注或模型权重。
>
> **时间单位：核心 API 的 `now` / `created_at` / `ts` 都是「天」，不是 epoch 秒。**
> 用 `time.time()` 时请先除以 `86400`；`examples/runtime_hook.py` 是唯一的秒→天边界。

## 这是什么

它把记忆表示为 SQLite 中的节点、标签和边，并提供：

- 冻结的事件观测量，避免复述时把当时的情绪越推越高；
- 词面/标签检索，以及带深度、边类型、时间衰减和硬上限的扩散检索；
- 幂律遗忘曲线、生命周期状态和可选的复述巩固；
- 显式访问提交：检索本身只读，不偷偷改图；
- 隐私闸、硬删除、墓碑和备份/恢复工具；
- 主网 / 辅网：自我轨迹（正常路径 append-only，另备显式擦除）与显式提案-批准路径；见
  [docs/dual-network.md](https://github.com/hinoki-shi/lunavender-memory/blob/main/docs/dual-network.md)；
- 可选的向量检索 sidecar（实验原型）：主网不变，sidecar 可重建；见
  [docs/vector_sidecar.md](https://github.com/hinoki-shi/lunavender-memory/blob/main/docs/vector_sidecar.md)；
- 近期工作记忆缓冲的纯逻辑实现。

核心包只依赖 Python 标准库；适配器、模型和外部 API 不属于核心包。

## 不是什么

- 不包含 LLM 事实抽取器；`examples/runtime_hook.py` 里的规则提取只是接线示例。
- 不包含向量数据库或 embedding 服务。
- 不包含真实事件、个人标注、模型权重或运行时数据库。
- 不承诺当前参数适用于所有语料；没有跨库 benchmark 结论。
- 不与其他记忆系统做功能或性能对比；这里只记录本实现的设计与测试。

## 安装

当前尚未发布到 PyPI。可以直接从 GitHub 安装：

```bash
python -m pip install "git+https://github.com/hinoki-shi/lunavender-memory.git"
```

或者下载/克隆仓库后本地安装：

```bash
python -m pip install .
```

也可以从 [Releases](https://github.com/hinoki-shi/lunavender-memory/releases) 下载 wheel/sdist 后本地安装。

Python 3.10+。核心包没有第三方运行时依赖；可选的 Laya adapter 需要另外安装 `laya` 并自行获取权重。

## 最小用法

```python
from memlayer import MemoryStore, Observation, Query, remember, retrieve, Params

store = MemoryStore(":memory:")
a = remember(
    store,
    summary="在咖啡馆和朋友聊到周末的计划",
    obs=Observation(happy=0.6, hope=0.4, importance=0.5),
    scene="某天·下午",
    tags=("咖啡馆", "朋友", "周末"),
    source_quote="我们周末去看看吧",
)
b = remember(
    store,
    summary="周末去了书店",
    obs=Observation(happy=0.7, importance=0.4),
    scene="某天·周末",
    tags=("书店", "周末"),
    source_quote="周末去了书店",
)

result = retrieve(
    store.snapshot(),
    Query(terms=("周末", "朋友"), now=1.0),  # days, not epoch seconds
    d=0.5,
    params=Params(),
)
print(result.top(k=5))
```

在源码仓库中可以运行 `python examples/quickstart.py`。

## Optional Laya adapter

`examples/laya_obs_adapter.py` is an optional adapter for
[Laya](https://github.com/NandhaKishorM/laya). It is not bundled as a model and
does not include weights. The upstream project is by Convai Innovations under
Apache-2.0; see [docs/laya.md](https://github.com/hinoki-shi/lunavender-memory/blob/main/docs/laya.md) and
[THIRD_PARTY_NOTICES.md](https://github.com/hinoki-shi/lunavender-memory/blob/main/THIRD_PARTY_NOTICES.md) for attribution and download
links. The base/checkpoint output is only a starting point: fine-tune and
evaluate your own checkpoint on your data.

## 设计约束

1. **观测量写入后冻结**：复述只影响显式的访问量，不回写当时的情绪。
2. **检索是纯函数**：同一个快照、参数和查询应得到同样的结果。
3. **访问是显式写入**：需要更新 `access_count` 时调用 `commit_access()`。
4. **边类型有限**：同场景共现和派生关系分开；跨场景传播必须走明确的派生通道。
5. **隐私优先**：所有写入路径（节点、边证据、轨迹）都过敏感内容闸；删除会同时清理节点、边、标签、出处和引用这些节点的轨迹，并留下无内容墓碑。
6. **可恢复**：更新数据库前使用 `memory_backup.py` 做一致性快照。
7. **轨迹最小化**：检索轨迹只存查询/场景指纹；提案轨迹只存摘要指纹，批准时由调用方重新提供文本。

## 可复核的边界

这里的“可复核”指：能查到一条记忆是怎么来的 —— 观测量、出处、参数/激活指纹、批准链。

它**不是 tamper-evident**：`delete_trace()` / `redact_trace_payload()` 是公开 API，而且没有 hash chain；改一行 payload 不会被自动发现。需要防篡改时，请把导出快照写到自己的 WORM / append-only 存储。

## 测试

```bash
python -m unittest discover -s tests -v
```

当前核心与示例测试共 **121** 条。

## 目录

```text
memlayer/          核心包（仅标准库）
tests/             核心与示例回归测试
examples/          synthetic quickstart、规则版 runtime 接线、obs schema、可选 Laya adapter
memory_backup.py   一致性导出/校验/恢复
backup_memory.py   CLI 包装
SECURITY.md        漏洞报告与安全边界
docs/              架构与设计说明
```

## 已知限制

- 词面检索仍是默认实现，尚未接向量召回；
- 事件标签提取需要由调用方提供；
- 规则提取器只用于端到端接线，不应被当成准确的情绪模型；
- 记忆压缩、时间模糊和反馈验证仍是后续工作；
- 默认是明文 SQLite，不内置整库/字段加密；需要静态加密时请把库放在 BitLocker / LUKS 等加密卷，或使用 SQLCipher 兼容封装。文件权限只做 `0600` 尽力而为，Windows ACL 需自行配置；
- 向量 sidecar 自带的是确定性 `HashEmbedder`（测试替身，不是语义模型）；真实 BGE-M3 / e5 适配器尚未内置，且当前检索是纯 Python 线性扫描，规模上去后要换 Flat/HNSW/IVF-PQ；
- 新版提案轨迹只存文本指纹，批准时必须重新传入 `summary=`；旧版轨迹会在批准后擦掉自由文本字段；
- API 在 `0.x` 阶段可能调整。

## Privacy and data

本仓库不附带真实记忆或标注。测试和示例中的事件均为合成或通用样例。

运行时默认会落盘的内容：

- 节点：摘要、场景、标签、冻结的九通道观测量、出处原话/轮次/来源；
- 边：证据原话（如有）；
- 辅网轨迹：检索的查询/场景指纹、提案的摘要/原话/备注指纹与长度、节点 ID 与观测量；新版轨迹不存这些文本的原文。

`cognition_edit` 的 `reason` / `detail` 是调用方自述的审计备注，仍可能含自由文本；如含个人数据，请用 `SelfTrace.purge([seq])` 指定删除。

`MemoryStore.forget()` 会删除节点、边、标签、出处和引用这些节点的轨迹；文件库请随后调用 `scrub()` 做 `VACUUM`。外部缓存/日志不在库的管辖范围内，请用 `find_text_in_files()` 扫描、`scrub_files()` 擦除；后者默认不留 `.bak`。

更完整的落盘内容、删除边界和静态加密建议见 [docs/privacy.md](https://github.com/hinoki-shi/lunavender-memory/blob/main/docs/privacy.md)。

如果你把本库接入真实对话，请自行承担数据分级、脱敏、访问控制、告知/同意和备份责任。

## Responsible use

本库会从文本推断情绪相关的观测量。它是研究原型，不是医疗、心理诊断、招聘、教育评估或保险/信贷决策工具。

部署前请自行确认当地法律与场景要求，至少包括：

- EU AI Act：工作场所/教育场景的情绪识别属于禁止用途（Art. 5(1)(f)，医疗/安全等法定例外除外）；相关交互系统还可能触发透明度义务（Art. 50）。
- PIPL：自动化决策应提供说明，并给出拒绝/非自动化途径（Art. 24）。
- 需要合法基础、明确告知、最小化数据、保存期限、人工复核和删除通道；不要用于隐蔽监控、弱势人群或高风险决策。

本库不提供同意管理、年龄门槛或特定司法辖区的合规实现；上述不是法律意见，部署前请咨询合规/法律人员。

## Reproducibility

公开预览不包含作者的完整实验脚本、原始语料、人工标注和内部设计稿。代码注释里的 § 编号和实验数字来自内部夹具，不能仅凭本仓库独立复现；请把它们当作工程笔记，而不是 benchmark 结论。§ 编号也没有公开版对应文档（见 [CONTRIBUTING.md](https://github.com/hinoki-shi/lunavender-memory/blob/main/CONTRIBUTING.md)）。可复现的是本仓库的单元测试与行为约束。

## License

MIT. See [LICENSE](https://github.com/hinoki-shi/lunavender-memory/blob/main/LICENSE).
