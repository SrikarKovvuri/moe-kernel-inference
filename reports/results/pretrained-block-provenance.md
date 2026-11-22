# Pretrained block source audit

`pretrained-block.json` records source revision
`6600682c6d2e6356d958175df6a3c5a39efc9cfc` and `git_dirty=true`.
Completed graph results were being added to the report archive during this run.
The raw flag is preserved rather than rewritten as a clean run.

The benchmark/kernel source in that revision is identical to the next clean
checkpoint `cfa691043e0f315b6deefb8ecb5079901462a059` for the measured code:

```sh
git diff --exit-code 6600682 cfa6910 -- src/phasemoe scripts/benchmark_pretrained_block.py
```

This check completed with exit code 0 and no differences. The checkpoint weights
are independently identified by pinned Hugging Face revision and per-tensor hashes
inside the artifact. Later experiment metadata additionally records source-file
SHA256 hashes and the dirty-file list, avoiding ambiguity from concurrent report edits.
