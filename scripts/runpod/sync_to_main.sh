#!/bin/bash
# Sync worktree changes to main repo and push to remote

set -e  # Exit on error

MAIN_REPO="/Users/dhyana/mech-interp-latent-lab-phase1"
WORKTREE="/Users/dhyana/.cursor/worktrees/mech-interp-latent-lab-phase1/xce"

echo "=========================================="
echo "SYNC WORKTREE → MAIN → REMOTE"
echo "=========================================="
echo ""

# Step 1: Check current status
echo "[1/5] Checking worktree status..."
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
    read -p "  Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "  ✓ Commits match"
fi

# Step 2: Check for uncommitted changes in worktree
echo ""
echo "[2/5] Checking for uncommitted changes in worktree..."
cd "$WORKTREE"
if [ -n "$(git status --porcelain)" ]; then
    echo "  ⚠️  Uncommitted changes found:"
    git status --short
    read -p "  Commit these changes? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "Sync changes from worktree"
        echo "  ✓ Changes committed"
    fi
else
    echo "  ✓ No uncommitted changes"
fi

# Step 3: Fetch latest from remote
echo ""
echo "[3/5] Fetching latest from remote..."
cd "$MAIN_REPO"
git fetch origin
echo "  ✓ Fetched from remote"

# Step 4: Check if main is ahead/behind remote
echo ""
echo "[4/5] Checking sync status with remote..."
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "none")
if [ "$REMOTE" != "none" ] && [ "$LOCAL" != "$REMOTE" ]; then
    echo "  ⚠️  Main repo is not in sync with remote"
    echo "  Local:  $LOCAL"
    echo "  Remote: $REMOTE"
    read -p "  Pull from remote first? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git pull origin main
        echo "  ✓ Pulled from remote"
    fi
else
    echo "  ✓ Main repo is in sync with remote"
fi

# Step 5: Push to remote
echo ""
echo "[5/5] Pushing to remote..."
read -p "  Push main branch to origin? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin main
    echo "  ✓ Pushed to remote"
else
    echo "  ⊘ Skipped push"
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
echo "✓ SYNC COMPLETE"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Main repo: $MAIN_REPO"
echo "  - Worktree:  $WORKTREE"
echo "  - Both are now at: $(cd "$MAIN_REPO" && git rev-parse HEAD)"
echo ""
