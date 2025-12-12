#!/usr/bin/env python3
"""
Run both scripts 3 times each and compare results
"""

import subprocess
import sys
import re
from collections import defaultdict

def extract_results(output):
    """Extract results from script output"""
    results = {}
    for layer in [25, 27]:
        pattern = f'LAYER {layer}:.*?Rec R_V:\s+([\d.]+).*?Base R_V:\s+([\d.]+).*?Cohen\'s d:\s+([-\d.]+)'
        match = re.search(pattern, output, re.DOTALL)
        if match:
            results[layer] = {
                'rec_rv': float(match.group(1)),
                'base_rv': float(match.group(2)),
                'cohens_d': float(match.group(3))
            }
    return results

def run_script(script_name, run_num):
    """Run a script and capture output"""
    print(f"\n{'='*70}")
    print(f"RUN {run_num}: {script_name}")
    print('='*70)
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True,
        cwd='/workspace/mech-interp-latent-lab-phase1-1'
    )
    
    output = result.stdout + result.stderr
    print(output)
    
    if result.returncode != 0:
        print(f"ERROR: Script failed with return code {result.returncode}")
        return None
    
    return extract_results(output)

# Run both scripts 3 times
all_results = {
    'operation_restoration': [],
    'reproduce_nov16': []
}

print("\n" + "="*70)
print("COMPARISON TEST: Running both scripts 3 times each")
print("="*70)

for run in range(1, 4):
    # Run operation_restoration.py
    op_result = run_script('operation_restoration.py', run)
    if op_result:
        all_results['operation_restoration'].append(op_result)
    
    # Run reproduce_nov16_mistral.py
    nov_result = run_script('reproduce_nov16_mistral.py', run)
    if nov_result:
        all_results['reproduce_nov16'].append(nov_result)

# Analyze results
print("\n" + "="*70)
print("COMPARATIVE ANALYSIS")
print("="*70)

for script_name, results_list in all_results.items():
    if not results_list:
        print(f"\n{script_name}: No valid results")
        continue
    
    print(f"\n{script_name.upper()} ({len(results_list)} runs):")
    
    for layer in [25, 27]:
        rec_rvs = [r[layer]['rec_rv'] for r in results_list if layer in r]
        base_rvs = [r[layer]['base_rv'] for r in results_list if layer in r]
        cohens_ds = [r[layer]['cohens_d'] for r in results_list if layer in r]
        
        if rec_rvs:
            print(f"\n  Layer {layer}:")
            print(f"    Rec R_V:  {sum(rec_rvs)/len(rec_rvs):.4f} (range: {min(rec_rvs):.4f} - {max(rec_rvs):.4f})")
            print(f"    Base R_V: {sum(base_rvs)/len(base_rvs):.4f} (range: {min(base_rvs):.4f} - {max(base_rvs):.4f})")
            print(f"    Cohen's d: {sum(cohens_ds)/len(cohens_ds):.4f} (range: {min(cohens_ds):.4f} - {max(cohens_ds):.4f})")

# Direct comparison
print("\n" + "="*70)
print("DIRECT COMPARISON (Average across 3 runs)")
print("="*70)

for layer in [25, 27]:
    print(f"\nLAYER {layer}:")
    
    op_results = [r[layer] for r in all_results['operation_restoration'] if layer in r]
    nov_results = [r[layer] for r in all_results['reproduce_nov16'] if layer in r]
    
    if op_results and nov_results:
        op_rec = sum(r['rec_rv'] for r in op_results) / len(op_results)
        op_base = sum(r['base_rv'] for r in op_results) / len(op_results)
        op_d = sum(r['cohens_d'] for r in op_results) / len(op_results)
        
        nov_rec = sum(r['rec_rv'] for r in nov_results) / len(nov_results)
        nov_base = sum(r['base_rv'] for r in nov_results) / len(nov_results)
        nov_d = sum(r['cohens_d'] for r in nov_results) / len(nov_results)
        
        print(f"  Operation Restoration:")
        print(f"    Rec R_V:  {op_rec:.4f}, Base R_V: {op_base:.4f}, Cohen's d: {op_d:.4f}")
        print(f"  Nov16 Reproduction:")
        print(f"    Rec R_V:  {nov_rec:.4f}, Base R_V: {nov_base:.4f}, Cohen's d: {nov_d:.4f}")
        print(f"  Differences:")
        print(f"    Rec R_V diff:  {nov_rec - op_rec:.4f}")
        print(f"    Base R_V diff: {nov_base - op_base:.4f}")
        print(f"    Cohen's d diff: {nov_d - op_d:.4f}")

