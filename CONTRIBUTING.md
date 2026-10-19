# Contributing

Thanks for taking a look. This is a small research project; focused issues and
small patches are especially useful.

Before opening a pull request:

```bash
python -m unittest discover -s tests -v
python -m compileall -q memlayer examples tests
```

Please keep these boundaries:

- core `memlayer` stays free of network/model dependencies;
- tests use synthetic data, not personal conversations;
- retrieval remains deterministic on a frozen snapshot;
- new optional behavior should be opt-in and covered by a regression test;
- do not include API keys, model weights, or private event logs.

For larger design changes, open an issue first and describe the invariant you
want to preserve.
