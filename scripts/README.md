# Instructions for training and evaluating SAC controller

## Training 
- Ran with seeds=100,200,300

### Rule high-level + SAC low-level

```bash
uv run python baselines/train_rule_macro_sac.py algo=sac scenario=default experiment.seed=100
```

### Default high-level + SAC low-level

```bash
uv run python baselines/train_sac.py algo=sac scenario=default experiment.seed=100
```

### Random high-level + SAC low-level

```bash
uv run python baselines/train_lowsac_highrandom.py algo=sac scenario=default experiment.seed=100
```

### SAC high-level + Rule low-level

```bash
uv run python baselines/train_hierarchical_sac.py algo=sac scenario=default experiment.seed=100
```

## Evaluation

### Rule high-level + SAC low-level

```bash
uv run python evaluation/run_sac_benchmark.py --agents rule_macro --scenarios default --inner-agents sac --model_dir <path to saved model from training>
```

### Default high-level + SAC low-level

```bash
uv run python evaluation/run_defaultmacro_benchmark.py --agents default_macro+sac --scenarios default --model_dir <path to saved model from training>
```

### Random high-level + SAC low-level

```bash
uv run python evaluation/run_sac_benchmark.py --agents random_macro --scenarios default --inner-agents sac --model_dir <path to saved model from training>
```
