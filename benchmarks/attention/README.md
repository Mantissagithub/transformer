# Transformer Lab Benchmark

- Collection: `Pradheep1647/transformer-lab-6a07fe3185f5728e217997e0`
- Dataset: `meetingbank` / `validation`
- Generation samples: `128`
- Precision: `fp32`
- Archived runs: `runs/<timestamp>/` under this folder.

| Repo | Status | Loss | PPL | Tok Acc | ROUGE-L | BLEU | Tok/s | Gen tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Pradheep1647/run_mha-meetingbank-bs8-e20-fp32-19` | ok | 2.764 | 15.86 | 0.5804 | 0.3951 | 24.71 | 4364 | 241.9 |
| `Pradheep1647/run_gqa_rope-meetingbank-bs8-e20-fp32-19` | ok | 2.78 | 16.12 | 0.5762 | 0.3799 | 24.18 | 8039 | 345.8 |
| `Pradheep1647/run_mqa-meetingbank-bs8-e20-fp32-19` | ok | 2.791 | 16.29 | 0.5722 | 0.3804 | 23.15 | 4972 | 224.6 |
| `Pradheep1647/run_gqa-meetingbank-bs8-e20-fp32-19` | ok | 2.811 | 16.63 | 0.5737 | 0.3843 | 23.11 | 4289 | 218.2 |
| `Pradheep1647/run_sliding_gqa-meetingbank-bs8-e20-fp32-19` | ok | 2.843 | 17.17 | 0.573 | 0.4149 | 26.24 | 4939 | 168.7 |
| `Pradheep1647/run_csa-meetingbank-bs8-e20-fp32-19` | ok | 10 | 2.213e+04 | 0.04232 | 0.07588 | 0.1682 | 2823 | 109.6 |
| `Pradheep1647/run_hca-meetingbank-bs8-e20-fp32-19` | ok | 10.16 | 2.585e+04 | 0.04029 | 0.0296 | 0.02163 | 6811 | 396 |
