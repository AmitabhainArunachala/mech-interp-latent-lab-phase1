# Layer Sweep Debug Fixes

## Problem
Layer sweep experiment was hanging for 3+ hours without completing. Total runs: 300 (20 layers × 3 types × 5 prompts).

## Root Causes Identified

1. **No Progress Saving**: If the experiment crashed or hung, all progress was lost
2. **No Memory Clearing**: GPU memory accumulated across 300 runs without clearing
3. **No Intermediate Logging**: Couldn't tell which layer/prompt was causing the hang
4. **No Resume Capability**: Had to restart from scratch every time

## Fixes Applied

### 1. Progress Saving After Each Layer
- Results are saved to CSV after each layer completes
- Can resume from where it left off if interrupted
- Checks for existing results and skips completed runs

### 2. GPU Memory Clearing
- `torch.cuda.empty_cache()` called before each run
- Prevents memory accumulation that could cause OOM or slowdowns

### 3. Enhanced Logging
- Progress logged every 10 runs: `[Progress] Run X/300: L{layer} {type} prompt {idx}`
- Layer completion messages: `[Saved] Layer X complete. Total: Y/300`
- Better error messages with full traceback

### 4. Better Error Handling
- Patchers are always removed in `finally` blocks
- Errors are caught and logged without stopping the entire experiment
- Each failed run is recorded with error message

### 5. Code Structure Improvements
- Results directory created before saving
- Proper handling of existing results for resume capability
- Cleaner separation of concerns

## Expected Behavior Now

1. **Progress Tracking**: See exactly which run is executing
2. **Resume Capability**: Can restart and continue from last saved layer
3. **Memory Management**: GPU memory cleared between runs
4. **Fault Tolerance**: Individual failures don't stop the entire experiment

## Testing Recommendations

1. Run a small test first (e.g., L24-L27) to verify fixes work
2. Monitor GPU memory usage: `nvidia-smi -l 1`
3. Check progress file: `results/runs/*/layer_sweep_results.csv`
4. If it still hangs, check logs to see which specific run is stuck

## Next Steps

If the experiment still hangs:
1. Check which layer/prompt combination is causing the issue
2. Add more aggressive timeout (may require threading/multiprocessing)
3. Consider batching runs or reducing max_new_tokens
4. Profile GPU memory usage to identify leaks







