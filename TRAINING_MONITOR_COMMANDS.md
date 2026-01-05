# UniLCD PPO Training - Monitoring Commands

## Training Status
- **Process ID**: 556649
- **Log file**: `train_unilcd_paper_robust.log`
- **Checkpoint directory**: `./checkpoints_ppo/`

## Robustness Features
- ✓ **Checkpoint saving**: Every 10 episodes
- ✓ **CARLA auto-restart**: Every 20 episodes (prevents deadlocks)
- ✓ **Auto-resume**: Training will resume from last checkpoint if interrupted

## Quick Progress Check

```bash
# See the last 5 completed episodes
grep "Episode.*1000" train_unilcd_paper_robust.log | tail -5
```

## Check if Training is Running

```bash
ps -p 556649 -o pid,%cpu,%mem,etime,cmd
```

If you see "no such process", the training stopped (check log for errors).

## View Live Training Output

```bash
tail -f train_unilcd_paper_robust.log
```
Press `Ctrl+C` to stop watching.

## Calculate Progress and ETA

```bash
python3 << 'EOF'
import re
import subprocess

log_file = "train_unilcd_paper_robust.log"
with open(log_file, 'r') as f:
    lines = f.readlines()

episode_lines = [l for l in lines if "Episode" in l and "/1000" in l]
if episode_lines:
    last_ep = episode_lines[-1]
    match = re.search(r'Episode (\d+)/1000', last_ep)
    if match:
        completed = int(match.group(1))

        result = subprocess.run(['ps', '-p', '556649', '-o', 'etimes'],
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            elapsed_sec = int(lines[1].strip())

            avg_time_per_ep = elapsed_sec / completed if completed > 0 else 0
            remaining_eps = 1000 - completed
            estimated_remaining_sec = avg_time_per_ep * remaining_eps

            hours = int(estimated_remaining_sec / 3600)
            minutes = int((estimated_remaining_sec % 3600) / 60)

            print(f"Episodes completed: {completed}/1000 ({completed/10:.1f}%)")
            print(f"Elapsed time: {elapsed_sec/3600:.1f} hours")
            print(f"Avg time per episode: {avg_time_per_ep:.1f} seconds")
            print(f"Estimated remaining: {hours}h {minutes}m")
        else:
            print("Training process not found")
    else:
        print("No episodes completed yet")
else:
    print("No episodes completed yet")
EOF
```

## Check Checkpoints

```bash
# List all checkpoints
ls -lh checkpoints_ppo/

# Count checkpoints
ls checkpoints_ppo/checkpoint_ep*.zip 2>/dev/null | wc -l
```

## GPU Usage

```bash
nvidia-smi
```

## If Training Crashes

The training will **automatically resume** from the last checkpoint. Just run:

```bash
nohup /media/shival/data4/shival/UniLCD/unilcd_env_py38/bin/python train_unilcd_paper_robust.py > train_unilcd_paper_robust.log 2>&1 &
```

It will detect the latest checkpoint and continue from there.

## Important Notes

1. **Checkpoints do not reduce episode count** - All 1,000 episodes will be trained
2. **CARLA restarts every 20 episodes** - This is automatic and prevents deadlocks
3. **Each checkpoint includes**:
   - Model weights (`checkpoint_epN.zip`)
   - Training state (`callback_state_epN.pkl`)
4. **Final model**: Saved as `unilcd_ppo_paper_1000ep.zip` after 1,000 episodes

## Expected Timeline

- With CARLA restarts, training may be slightly slower (~5% overhead)
- Estimated total time: **38-42 hours** for 1,000 episodes
- Progress saved every 10 episodes, so maximum loss from crash: ~10 episodes
