 uv run play Mjlab-Walk-Flat-B2Z1 --agent zero
 uv run play Mjlab-Walk-Flat-B2Z1 --agent random

 uv run play Mjlab-Walk-Flat-B2Z1 --wandb-run-path <wandb-run-path>

 CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Walk-Flat-B2Z1   --env.scene.num-envs 4   --agent.max-iterations 3000

 uv run play Mjlab-Walk-Flat-B2Z1 --wandb-run-path 15632361677-none/mjlab/56d0jqiv # 臂不动
 uv run play Mjlab-Walk-Flat-B2Z1 --wandb-run-path 15632361677-none/mjlab/attju8dw

 wandb: 🚀 View run 2026-03-09_21-14-52 at: https://wandb.ai/15632361677-none/mjlab/runs/56d0jqiv