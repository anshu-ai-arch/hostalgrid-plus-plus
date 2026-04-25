# Final Run Gate

## Current Local Winner
- Preference-Distilled reward_mean: 13.008463

## Local Findings
- Merged preference distillation did not beat the current winner.
- Grouped preference distillation did not beat the current winner.
- GRPO-lite underperformed badly.

## Decision
- Stop local post-training tuning here.
- Use HF credits only for final serious runs.

## Final Runs
1. Offline HF-credit run
   - goal: beat 13.008463
2. Online HF-credit run
   - goal: beat base and approach offline winner
3. Held-out evaluation
   - compare final kept runs only
