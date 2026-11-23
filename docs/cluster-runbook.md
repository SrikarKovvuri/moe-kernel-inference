# Reproduce the baseline on the target GPU

No cluster job has been submitted or cluster access verified. Use one allocated
L40S/A6000 and a private Python environment; adapt driver/wheel choices to that machine.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev,plots,gpu]"
python -m pytest -q
nvidia-smi -q
git status --short
bash scripts/run_midpoint.sh runs/cluster-baseline
```

The known development versions are in `requirements/validation-linux.txt`.
For Slurm, adapt `scripts/midpoint.slurm` to the real account/partition/GPU constraint,
activate the environment, and submit from the repository root. The template requests
one GPU, 16 GB host memory and one hour. It does not assume institutional resource names.

Inspect correctness, power/clocks, both timing clocks, source revision/hashes and
scope before drawing conclusions. Preserve failed artifacts as failed. Do not
reuse laptop numbers or hardware claims in a cluster report. The baseline uses
synthetic weights and routes and does not download the model or perform generation.

To build a report from a completed compatible baseline run:

```sh
python -m pip install -e ".[report]"
python scripts/build_midpoint_report.py --benchmark runs/cluster-baseline/baseline.json \
  --profile runs/cluster-baseline/profile.json --output runs/cluster-report
```

Review the generated narrative against the new telemetry. Render the Word file
with LibreOffice or Word and inspect every page before submission. No report build
or benchmark script commits, pushes, or uploads results.
