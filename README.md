 uv run play Mjlab-Walk-Flat-B2Z1 --agent zero
 uv run play Mjlab-Walk-Flat-B2Z1 --agent random

 uv run play Mjlab-Walk-Flat-B2Z1 --wandb-run-path <wandb-run-path>

 CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Walk-Flat-B2Z1   --env.scene.num-envs 4096   --agent.max-iterations 15000

 uv run play Mjlab-Walk-Flat-B2Z1 --wandb-run-path 15632361677-none/mjlab/56d0jqiv # 臂不动
 uv run play Mjlab-Walk-Flat-B2Z1 --wandb-run-path 15632361677-none/mjlab/eyjzpiyo

 uv run play Mjlab-Walk-Flat-B2Z1 --checkpoint-file logs/rsl_rl/b2z1_velocity/2026-03-10_22-12-14


 wandb: 🚀 View run 2026-03-09_21-14-52 at: https://wandb.ai/15632361677-none/mjlab/runs/56d0jqiv