# Evaluation

Five thin Python adapters that drive each benchmark's native CLI on a
MAD-OPD student checkpoint.

| Adapter | Benchmark | Upstream CLI |
|---|---|---|
| `mbpp_plus_eval.py`     | MBPP+              | `evalplus.codegen` + `evalplus.evaluate` |
| `livecodebench_eval.py` | LiveCodeBench v6   | `lcb_runner.runner.main` |
| `bfcl_eval.py`          | BFCL v4            | `bfcl_eval generate` + `bfcl_eval evaluate` |
| `tau2_bench_eval.py`    | τ²-Bench           | `tau2 run` |
| `vitabench_eval.py`     | VitaBench          | `vita run` |

## Setup

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/evalplus/evalplus.git
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
git clone https://github.com/ShishirPatil/gorilla.git
git clone https://github.com/sierra-research/tau2-bench.git
git clone https://github.com/meituan-longcat/vitabench.git
cd ..

pip install -e third_party/evalplus
pip install -e third_party/LiveCodeBench
pip install -e third_party/gorilla/berkeley-function-call-leaderboard
pip install -e third_party/tau2-bench
pip install -e third_party/vitabench
```

Pin to a known commit with `git checkout <sha>` inside each subdir.

## Run

```bash
# Code
python eval/mbpp_plus_eval.py      --model /path/to/student.ckpt --n_samples 16 --max_new_tokens 16384
python eval/livecodebench_eval.py  --model /path/to/student.ckpt --release v6 --tp 4

# Agentic (needs a user-simulator LLM)
python eval/bfcl_eval.py        --model /path/to/student.ckpt --registry_name student-v1
python eval/tau2_bench_eval.py  --model /path/to/student.ckpt --agent_model_name student-v1 \
                                --user_llm claude-opus-4-6 --domains airline,retail,telecom
python eval/vitabench_eval.py   --model /path/to/student.ckpt --agent_model_name student-v1 \
                                --user_llm claude-opus-4-6 --evaluator_llm claude-opus-4-6 \
                                --domains delivery,instore,ota
```

## Defaults

- Code benchmarks: `temperature=0.7`, `top_p=0.8`, 16 samples per problem,
  and a 16,384-token generation cap. The adapters currently produce
  sample-based pass@1 estimates; they do not run 16 independent seeded
  benchmark invocations or compute the paper's BoN@16 selector.
- Agentic benchmarks: `temperature=0.7`, `top_p=0.8`, `num_trials=4`,
  `max_steps=300`.
- The adapters request non-thinking mode where the upstream harness exposes
  it. BFCL additionally requires the Qwen handler change described below.
- BFCL-v4 uses `temperature=0.7`, the vLLM `hermes` tool-call parser,
  automatic tool choice, and `max_model_len=40960`. Its score averages 8
  categories: `simple_python, multiple, parallel,
  parallel_multiple, multi_turn_base, multi_turn_miss_func,
  multi_turn_miss_param, multi_turn_long_context`.
- τ²-Bench: `airline + retail + telecom`; VitaBench: `delivery + instore + ota`.

BFCL's native CLI currently exposes temperature but not top-p or presence
penalty for its local OSS handler. Reproducing the paper's non-thinking Qwen
setup also requires a BFCL Qwen handler that inserts an empty
`<think></think>` block before generation; this handler-level change is not
implemented by the adapter.
