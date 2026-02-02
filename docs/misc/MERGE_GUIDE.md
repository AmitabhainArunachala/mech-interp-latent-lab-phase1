# Git Worktree Merge & Sync Guide

## Current Status

✅ **Both repos are synced at commit `1d646ab`** ("OPERATION SAMURAI: Complete refoundation")

- **Main repo**: `/Users/dhyana/mech-interp-latent-lab-phase1` (branch: `main`)
- **Worktree**: `/Users/dhyana/.cursor/worktrees/mech-interp-latent-lab-phase1/xce` (detached HEAD)
- **Remote**: `origin` → `https://github.com/AmitabhainArunachala/mech-interp-latent-lab-phase1.git`

## Option 1: Sync Everything to Main Repo (Recommended)

Since both repos are at the same commit, you just need to handle untracked files and ensure everything is committed.

### Step 1: Go to Main Repo
```bash
cd /Users/dhyana/mech-interp-latent-lab-phase1
```

### Step 2: Check Status
```bash
git status
```

### Step 3: Handle DEC10_LEARNING_DAY (if you want to keep it)
```bash
# Option A: Add to boneyard (recommended for experimental code)
git mv DEC10_LEARNING_DAY boneyard/DEC10_LEARNING_DAY

# Option B: Add as-is
git add DEC10_LEARNING_DAY/
```

### Step 4: Commit Any Changes
```bash
git add .
git commit -m "Add DEC10_LEARNING_DAY experiments"
```

### Step 5: Push to Remote
```bash
git push origin main
```

## Option 2: Create a Branch from Worktree (If You Have Uncommitted Changes)

If you have changes in the worktree that aren't in main:

### Step 1: In Worktree, Create a Branch
```bash
cd /Users/dhyana/.cursor/worktrees/mech-interp-latent-lab-phase1/xce
git checkout -b sync-refoundation
```

### Step 2: Add and Commit Changes
```bash
git add .
git commit -m "Sync refoundation work"
```

### Step 3: Switch to Main Repo and Merge
```bash
cd /Users/dhyana/mech-interp-latent-lab-phase1
git checkout main
git merge sync-refoundation
```

### Step 4: Push to Remote
```bash
git push origin main
```

## Option 3: Force Sync Worktree to Match Main (If Main Has Changes)

If the main repo has changes you want in the worktree:

### Step 1: In Worktree, Reset to Main
```bash
cd /Users/dhyana/.cursor/worktrees/mech-interp-latent-lab-phase1/xce
git fetch origin
git reset --hard origin/main
```

## Quick Sync Commands

### Check if repos are in sync:
```bash
# Compare commits
cd /Users/dhyana/.cursor/worktrees/mech-interp-latent-lab-phase1/xce
git rev-parse HEAD
cd /Users/dhyana/mech-interp-latent-lab-phase1
git rev-parse HEAD
# Should match if synced
```

### Sync worktree to main:
```bash
cd /Users/dhyana/.cursor/worktrees/mech-interp-latent-lab-phase1/xce
git fetch origin
git reset --hard origin/main
```

### Sync main to remote:
```bash
cd /Users/dhyana/mech-interp-latent-lab-phase1
git fetch origin
git merge origin/main  # or git pull
git push origin main
```

## Current Differences (Non-Critical)

These exist but don't affect the refoundation work:
- `__pycache__/` files (generated, can be ignored)
- `.DS_Store` files (macOS metadata, should be in .gitignore)
- Some result CSV files in main repo (not in worktree)

## Recommended Next Steps

1. **Ensure .gitignore includes cache files**:
   ```bash
   echo "__pycache__/" >> .gitignore
   echo "*.pyc" >> .gitignore
   echo ".DS_Store" >> .gitignore
   ```

2. **Commit and push from main repo**:
   ```bash
   cd /Users/dhyana/mech-interp-latent-lab-phase1
   git add .gitignore
   git commit -m "Update .gitignore for cache files"
   git push origin main
   ```

3. **Sync worktree**:
   ```bash
   cd /Users/dhyana/.cursor/worktrees/mech-interp-latent-lab-phase1/xce
   git fetch origin
   git reset --hard origin/main
   ```

## Notes

- The worktree is in **detached HEAD** state - this is normal for worktrees
- Both repos share the same `.git` directory, so commits are automatically shared
- The refoundation work (`src/`, `prompts/`, `reproduce_results.py`) is already committed and synced
- Only untracked files need to be handled
