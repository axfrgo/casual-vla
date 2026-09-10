# Measured symbolic contact results

Generated 2026-09-09T21:57:18.977Z. These are discrete-event tests, not robotics or Intel scores. Speech latency was not measured.

Paired seeds 1–100. Reloaded skill from disk. Held-out seeds 1001–1100 with five protected samples and amber_vial identity. Text-only ablation stores no executable rule; it is not an LLM baseline.

| Cohort | Passed | Episodes with violations | Candidates blocked |
|---|---:|---:|---:|
| before_learning | 26/100 | 74/100 | 0 |
| text_only_uncompiled | 26/100 | 74/100 | 0 |
| after_learning | 100/100 | 0/100 | 200 |
| retention_reload | 100/100 | 0/100 | 200 |
| generalization_and_identity_transfer | 100/100 | 0/100 | 500 |

One correction. No motor dynamics, perception noise, force, collision, VLA inference or real contamination sensing is simulated. Blocked candidate counts include candidate checks, not unique real-world accidents prevented.
