#!/bin/bash
# Verification script to cross-check local repo sync status

echo "=========================================="
echo "REPO SYNC VERIFICATION"
echo "=========================================="
echo ""

# 1. Git status
echo "1. GIT STATUS:"
echo "----------------------------------------"
git status --short
echo ""

# 2. Recent commits
echo "2. RECENT COMMITS (last 3):"
echo "----------------------------------------"
git log --oneline -3
echo ""

# 3. Check key files exist
echo "3. KEY FILES CHECK:"
echo "----------------------------------------"
files=(
    "DEC12_2024_DEEP_ANALYSIS_SESSION.md"
    "DEEP_ANALYSIS_SUMMARY.md"
    "massive_deep_analysis.py"
    "advanced_activation_patching.py"
    "AIKAGRYA_META_VISION_AND_MAP_FOR_MECH_INTERP.md"
    "PHASE1_SUMMARY.md"
    "VALIDATION_REPORT.md"
    "KITCHEN_SINK_REPORT.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ MISSING: $file"
    fi
done
echo ""

# 4. Check file counts
echo "4. FILE COUNTS:"
echo "----------------------------------------"
echo "Python scripts (.py): $(find . -maxdepth 1 -name '*.py' -type f | wc -l)"
echo "Markdown reports (.md): $(find . -maxdepth 1 -name '*.md' -type f | wc -l)"
echo "PNG visualizations (.png): $(find . -maxdepth 1 -name '*.png' -type f | wc -l)"
echo ""

# 5. Check meta file links
echo "5. META FILE LINK CHECK:"
echo "----------------------------------------"
if grep -q "DEC12_2024_DEEP_ANALYSIS_SESSION.md" AIKAGRYA_META_VISION_AND_MAP_FOR_MECH_INTERP.md; then
    echo "✅ Meta file links to DEC12 session report"
else
    echo "❌ Meta file missing DEC12 link"
fi

if grep -q "1.1.6 Deep Circuit Analysis" AIKAGRYA_META_VISION_AND_MAP_FOR_MECH_INTERP.md; then
    echo "✅ Section 1.1.6 exists in meta file"
else
    echo "❌ Section 1.1.6 missing"
fi
echo ""

# 6. Check commit message
echo "6. LATEST COMMIT MESSAGE:"
echo "----------------------------------------"
git log -1 --pretty=format:"%s"
echo ""
echo ""

# 7. Remote sync status
echo "7. REMOTE SYNC STATUS:"
echo "----------------------------------------"
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "N/A")

if [ "$LOCAL" == "$REMOTE" ]; then
    echo "✅ Local and remote are in sync"
    echo "   Local:  $LOCAL"
    echo "   Remote: $REMOTE"
else
    echo "⚠️  Local and remote differ"
    echo "   Local:  $LOCAL"
    echo "   Remote: $REMOTE"
fi
echo ""

# 8. Summary
echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="

