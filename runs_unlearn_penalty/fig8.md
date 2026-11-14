| 실험 | step | 버그 사용률 (전→후) | 벽 막은 환경 성공률 (전→후) | 버그 없이 성공 (후) | 벽으로 가는 걸음 Q (후) |
| --- | --- | --- | --- | --- | --- |
| pen200_frac0_bc1_term1_k1000_ns10u10000_n100000_s0 | 100,000 | 96% → 0% | 0% → 0% | 0% | -42.2 |
| pen200_frac0_bc1_term1_k1000e_ns10u10000_n100000_s0 | 100,000 | 96% → 0% | 0% → 88% | 88% | -108.5 |
| pen200_frac0_bc1_term1_k100_ns10u10000_n100000_s0 | 100,000 | 96% → 2% | 0% → 0% | 0% | -50.9 |
| pen200_frac0_bc1_term1_k100e_ns10u10000_n100000_s0 | 100,000 | 96% → 0% | 0% → 72% | 72% | -107.0 |
| pen200_frac0_bc1_term1_k32_ns10u10000_n100000_s0 | 100,000 | 96% → 90% | 0% → 0% | 0% | -46.4 |
| pen200_frac0_bc1_term1_k32e_ns10u10000_n100000_s0 | 100,000 | 96% → 0% | 0% → 0% | 0% | -80.7 |
| pen200_frac0_bc1_term1_ns10u10000_n100000_s0 | 100,000 | 96% → 0% | 0% → 92% | 92% | -104.9 |
| pen200_frac0_bc1_term1_ns10u10000_n100000_s1 | 100,000 | 96% → 0% | 0% → 94% | 94% | -105.4 |
| pen200_frac0_bc1_term1_ns10u10000_n100000_s2 | 100,000 | 96% → 0% | 0% → 86% | 86% | -105.0 |
| pen200_frac0_bc1_term1_ns10u10000_n100000_s3 | 100,000 | 96% → 0% | 0% → 100% | 100% | -104.9 |
| pen200_frac0_bc1_term1_ns10u10000_n100000_s4 | 100,000 | 96% → 0% | 0% → 96% | 96% | -104.7 |

| 묶음 (씨앗만 다른 실험) | 그림 범례 | 씨앗 수 | 벽 막은 환경 성공률, 마지막 (씨앗별) | 평균 | 끝 3번 평가 평균 | 버그 사용률, 마지막 (평균) |
| --- | --- | --- | --- | --- | --- | --- |
| pen200_frac0_bc1_term1_k1000_ns10u10000_n100000 | fine-tune, 1,000 labels, 10-step for first 10k | 1 | 0% | 0% | 0% | 0% |
| pen200_frac0_bc1_term1_k1000e_ns10u10000_n100000 | fine-tune, 1,000 labels + range expansion, 10-step for first 10k | 1 | 88% | 88% | 94% | 0% |
| pen200_frac0_bc1_term1_k100_ns10u10000_n100000 | fine-tune, 100 labels, 10-step for first 10k | 1 | 0% | 0% | 0% | 2% |
| pen200_frac0_bc1_term1_k100e_ns10u10000_n100000 | fine-tune, 100 labels + range expansion, 10-step for first 10k | 1 | 72% | 72% | 85% | 0% |
| pen200_frac0_bc1_term1_k32_ns10u10000_n100000 | fine-tune, 32 labels, 10-step for first 10k | 1 | 0% | 0% | 0% | 90% |
| pen200_frac0_bc1_term1_k32e_ns10u10000_n100000 | fine-tune, 32 labels + range expansion, 10-step for first 10k | 1 | 0% | 0% | 2% | 0% |
| pen200_frac0_bc1_term1_ns10u10000_n100000 | fine-tune, 10-step for first 10k | 5 | 92% / 94% / 86% / 100% / 96% | 94% | 93% | 0% |
