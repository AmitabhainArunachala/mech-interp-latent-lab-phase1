#!/usr/bin/env python3
"""
Extract individual files from the combined recovery file.
"""

import re
from pathlib import Path

INPUT_FILE = Path("/Users/dhyana/mech-interp-latent-lab-phase1/DEC_9_EMERGENCY_BACKUP/ONE_25_file_backup_with_py.md/25_RECOVERY_FILES.md")
OUTPUT_DIR = Path("/Users/dhyana/mech-interp-latent-lab-phase1/DEC_9_EMERGENCY_BACKUP/extracted")
OUTPUT_DIR.mkdir(exist_ok=True)

# Read the file
with open(INPUT_FILE, 'r') as f:
    lines = f.readlines()

# Find file boundaries
# Pattern: "File N: " or "N: " or "N:" at start of line
boundaries = []
for i, line in enumerate(lines):
    if re.match(r'^File \d+:\s*$', line) or re.match(r'^\d+:\s*$', line):
        # Extract file number
        match = re.search(r'(\d+)', line)
        if match:
            file_num = int(match.group(1))
            boundaries.append((i, file_num))

# Add implicit first file (starts at line 0)
if not boundaries or boundaries[0][0] > 0:
    boundaries.insert(0, (0, 1))

# Add end of file
boundaries.append((len(lines), 999))

print(f"Found {len(boundaries)-1} files")
print("Boundaries:", [(b[0]+1, b[1]) for b in boundaries[:-1]])

# Extract each file
saved_files = []
for idx in range(len(boundaries) - 1):
    start_line, file_num = boundaries[idx]
    end_line, _ = boundaries[idx + 1]
    
    # Skip the marker line itself (except for first file)
    if idx > 0:
        start_line += 1
    
    # Get content
    content_lines = lines[start_line:end_line]
    content = ''.join(content_lines).strip()
    
    if not content:
        continue
    
    # Determine file type
    if content.startswith('#!/usr/bin/env python3') or content.startswith('#!/usr/bin/env python'):
        ext = '.py'
        # Try to extract name from docstring
        docstring_match = re.search(r'"""[\s\n]*([^\n]+)', content)
        if docstring_match:
            name = docstring_match.group(1).strip()
            # Clean up the name
            name = re.sub(r'[^\w\s-]', '', name)
            name = name.lower().replace(' ', '_').replace('-', '_')
            name = re.sub(r'_+', '_', name).strip('_')[:60]
        else:
            name = f"file_{file_num}"
    else:
        ext = '.md'
        # Try to extract name from first heading
        heading_match = re.search(r'^#+ (.+)', content, re.MULTILINE)
        if heading_match:
            name = heading_match.group(1).strip()
            name = re.sub(r'[^\w\s-]', '', name)
            name = name.lower().replace(' ', '_').replace('-', '_')
            name = re.sub(r'_+', '_', name).strip('_')[:60]
        else:
            name = f"file_{file_num}"
    
    # Add file number prefix to ensure uniqueness
    filename = f"{file_num:02d}_{name}{ext}"
    filepath = OUTPUT_DIR / filename
    
    with open(filepath, 'w') as f:
        f.write(content + '\n')
    
    saved_files.append((filename, len(content_lines), ext))
    print(f"  Saved: {filename} ({len(content_lines)} lines)")

print(f"\n✅ Extracted {len(saved_files)} files to {OUTPUT_DIR}")

# Summary
py_files = [f for f in saved_files if f[2] == '.py']
md_files = [f for f in saved_files if f[2] == '.md']
print(f"\n📊 Summary:")
print(f"   Python files: {len(py_files)}")
print(f"   Markdown files: {len(md_files)}")

