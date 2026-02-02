#!/bin/bash
# Non-interactive sync script - safe operations only

set -e  # Exit on error

MAIN_REPO="/Users/dhyana/mech-interp-latent-lab-phase1"
WORKTREE="/Users/dhyana/.cursor/worktrees/mech-interp-latent-lab-phase1/xce"

echo "=========================================="
echo "AUTO SYNC WORKTREE → MAIN → REMOTE"
echo "=========================================="
echo ""

# Step 1: Check current status
echo "[1/6] Checking worktree status..."
cd "$WORKTREE"
WORKTREE_COMMIT=$(git rev-parse HEAD)
WORKTREE_BRANCH=$(git branch --show-current 2>/dev/null || echo "detached HEAD")
echo "  Worktree commit: $WORKTREE_COMMIT"
echo "  Worktree branch: $WORKTREE_BRANCH"

cd "$MAIN_REPO"
MAIN_COMMIT=$(git rev-parse HEAD)
MAIN_BRANCH=$(git branch --show-current)
echo "  Main repo commit: $MAIN_COMMIT"
echo "  Main repo branch: $MAIN_BRANCH"

if [ "$WORKTREE_COMMIT" != "$MAIN_COMMIT" ]; then
    echo "  ⚠️  WARNING: Commits differ!"
    echo "  Worktree: $WORKTREE_COMMIT"
    echo "  Main:     $MAIN_COMMIT"
    echo "  → Will sync worktree to main"
else
    echo "  ✓ Commits match"
fi

# Step 2: Check for uncommitted changes in worktree
echo ""
echo "[2/6] Checking for uncommitted changes in worktree..."
cd "$WORKTREE"
if [ -n "$(git status --porcelain)" ]; then
    echo "  ⚠️  Uncommitted changes found:"
    git status --short
    echo "  → Skipping commit (use interactive script to commit)"
else
    echo "  ✓ No uncommitted changes"
fi

# Step 3: Fetch latest from remote
echo ""
echo "[3/6] Fetching latest from remote..."
cd "$MAIN_REPO"
git fetch origin
echo "  ✓ Fetched from remote"

# Step 4: Check if main is ahead/behind remote
echo ""
echo "[4/6] Checking sync status with remote..."
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "none")
if [ "$REMOTE" != "none" ] && [ "$LOCAL" != "$REMOTE" ]; then
    BEHIND=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo "0")
    AHEAD=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo "0")
    
    if [ "$BEHIND" -gt 0 ]; then
        echo "  ⚠️  Main repo is $BEHIND commits behind remote"
        echo "  → Pulling from remote..."
        git pull origin main --no-edit
        echo "  ✓ Pulled from remote"
    elif [ "$AHEAD" -gt 0 ]; then
        echo "  ✓ Main repo is $AHEAD commits ahead of remote"
        echo "  → Will push to remote"
    fi
else
    echo "  ✓ Main repo is in sync with remote"
fi

# Step 5: Push to remote (if ahead)
echo ""
echo "[5/6] Pushing to remote..."
AHEAD=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo "0")
if [ "$AHEAD" -gt 0 ]; then
    echo "  → Pushing $AHEAD commit(s) to remote..."
    git push origin main
    echo "  ✓ Pushed to remote"
else
    echo "  ✓ Nothing to push (already in sync)"
fi

# Step 6: Sync worktree to match main
echo ""
echo "[6/6] Syncing worktree to match main..."
cd "$WORKTREE"
git fetch origin
git reset --hard origin/main
echo "  ✓ Worktree synced to main"

echo ""
echo "=========================================="
echo "✓ AUTO SYNC COMPLETE"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Main repo: $MAIN_REPO"
echo "  - Worktree:  $WORKTREE"
echo "  - Both are now at: $(cd "$MAIN_REPO" && git rev-parse HEAD)"
echo ""
echo "Note: If you had uncommitted changes, use sync_to_main.sh for interactive mode"
