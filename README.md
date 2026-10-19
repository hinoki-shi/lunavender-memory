# Lunavender Memory

**夕苓记忆层 / Lunavender Memory** 是一个小型、可审计的长期记忆实验库。

> 状态：**research preview**。这是一个仍在迭代的研究原型，不是生产级记忆服务。
> 仓库只包含核心代码、合成示例和测试；不包含任何真实对话、个人标注或模型权重。

## 这是什么

它把记忆表示为 SQLite 中的节点、标签和边，并提供：

- 冻结的事件观测量，避免复述时把当时的情绪越推越高；
- 词面/标签检索，以及带深度、边类型、时间衰减和硬上限的扩散检索；
- 幂律遗忘曲线、生命周期状态和可选的复述巩固；
- 显式访问提交：检索本身只读，不偷偷改图；
- 隐私闸、硬删除、墓碑和备份/恢复工具；
- 主网 / 辅网：自我轨迹（append-only）与显式提案-批准路径；见
  [docs/dual-network.md](docs/dual-network.md)；
- 近期工作记忆缓冲的纯逻辑实现。

核心包只依赖 Python 标准库；适配器、模型和外部 API 不属于核心包。

## 不是什么

- 不包含 LLM 事实抽取器；`examples/runtime_hook.py` 里的规则提取只是接线示例。
- 不包含向量数据库或 embedding 服务。
- 不包含真实事件、个人标注、模型权重或运行时数据库。
- 不承诺当前参数适用于所有语料；没有跨库 benchmark 结论。
- 不与其他记忆系统做功能或性能对比；这里只记录本实现的设计与测试。

## 安装

当前从仓库本地安装：

```bash
python -m pip install -e .
```

Python 3.10+。

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
    Query(terms=("周末", "朋友"), now=1.0),
    d=0.5,
    params=Params(),
)
print(result.top(k=5))
```

也可以直接运行 `python examples/quickstart.py`。

## Optional Laya adapter

`examples/laya_obs_adapter.py` is an optional adapter for
[Laya](https://github.com/NandhaKishorM/laya). It is not bundled as a model and
does not include weights. The upstream project is by Convai Innovations under
Apache-2.0; see [docs/laya.md](docs/laya.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and download
links. The base/checkpoint output is only a starting point: fine-tune and
evaluate your own checkpoint on your data.

## 设计约束

1. **观测量写入后冻结**：复述只影响显式的访问量，不回写当时的情绪。
2. **检索是纯函数**：同一个快照、参数和查询应得到同样的结果。
3. **访问是显式写入**：需要更新 `access_count` 时调用 `commit_access()`。
4. **边类型有限**：同场景共现和派生关系分开；跨场景传播必须走明确的派生通道。
5. **隐私优先**：敏感内容拒写；删除会同时清理节点、边、标签和出处，并留下无内容墓碑。
6. **可恢复**：更新数据库前使用 `memory_backup.py` 做一致性快照。

## 测试

```bash
python -m unittest discover -s tests -v
```

当前核心与示例测试共 **102** 条。

## 目录

```text
memlayer/          核心包（仅标准库）
tests/             核心与示例回归测试
examples/          synthetic quickstart、规则版 runtime 接线、obs schema、可选 Laya adapter
memory_backup.py   一致性导出/校验/恢复
backup_memory.py   CLI 包装
docs/              架构与设计说明
```

## 已知限制

- 词面检索仍是默认实现，尚未接向量召回；
- 事件标签提取需要由调用方提供；
- 规则提取器只用于端到端接线，不应被当成准确的情绪模型；
- 记忆压缩、时间模糊和反馈验证仍是后续工作；
- API 在 `0.x` 阶段可能调整。

## Privacy and data

本仓库不附带真实记忆或标注。测试和示例中的事件均为合成或通用样例。
如果你把本库接入真实对话，请自行承担数据分级、脱敏、访问控制和备份责任。

## License

MIT. See [LICENSE](LICENSE).
