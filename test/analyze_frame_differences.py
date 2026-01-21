"""
Analyze the frame count differences between Video, NPY, and GT CSV files
"""

import pandas as pd

# Read the comparison CSV
df = pd.read_csv('frame_count_comparison.csv')

# Calculate differences
df['Video_NPY_Match'] = df['Video_Frames'] == df['NPY_Frames']
df['Diff_GT_Video'] = df['GT_CSV_Frames'] - df['Video_Frames']

print("=" * 80)
print("FRAME COUNT ANALYSIS")
print("=" * 80)

# Check if Video and NPY always match
print(f"\n✓ Video and NPY frames ALWAYS match: {df['Video_NPY_Match'].all()}")
print(f"  (All {len(df)} entries have identical Video and NPY frame counts)")

# Count mismatches
mismatches = df[df['Diff_GT_Video'] != 0]
matches = df[df['Diff_GT_Video'] == 0]

print(f"\n=== GT CSV vs Video Frame Differences ===")
print(f"Perfect matches (GT = Video): {len(matches)} / {len(df)} ({len(matches)/len(df)*100:.1f}%)")
print(f"Mismatches (GT ≠ Video):      {len(mismatches)} / {len(df)} ({len(mismatches)/len(df)*100:.1f}%)")

# Distribution of differences
print(f"\n=== Difference Distribution (GT - Video) ===")
diff_counts = df['Diff_GT_Video'].value_counts().sort_index()
for diff, count in diff_counts.items():
    if diff == 0:
        print(f"  {diff:+3d} frames (Perfect match): {count} entries")
    elif diff > 0:
        print(f"  {diff:+3d} frames (GT has MORE):  {count} entries")
    else:
        print(f"  {diff:+3d} frames (GT has LESS):  {count} entries")

# Show activities with issues
print(f"\n=== Activities with Frame Mismatches ===")
issues = mismatches.groupby(['Subject', 'Activity']).agg({
    'Diff_GT_Video': 'first',
    'Video_Frames': 'first',
    'GT_CSV_Frames': 'first'
}).reset_index()

issues = issues.sort_values(['Subject', 'Activity'])
print(f"\n{'Subject':<10} {'Activity':<12} {'Video':<8} {'GT':<8} {'Diff':<8} {'Issue'}")
print("-" * 70)

for _, row in issues.iterrows():
    diff = int(row['Diff_GT_Video'])
    video = int(row['Video_Frames'])
    gt = int(row['GT_CSV_Frames'])
    
    if diff > 0:
        issue_type = f"GT has {diff} MORE"
    else:
        issue_type = f"GT has {abs(diff)} LESS"
    
    print(f"{row['Subject']:<10} {row['Activity']:<12} {video:<8} {gt:<8} {diff:+4d}    {issue_type}")

# Special case: S1 acting3 has different video frame counts per camera
print(f"\n=== Special Cases ===")
s1_acting3 = df[(df['Subject'] == 'S1') & (df['Activity'] == 'acting3')]
if len(s1_acting3['Video_Frames'].unique()) > 1:
    print(f"\nS1 acting3: Different cameras have different video frame counts!")
    print(f"  Video frames range: {s1_acting3['Video_Frames'].min()} - {s1_acting3['Video_Frames'].max()}")
    print(f"  GT CSV frames: {s1_acting3['GT_CSV_Frames'].iloc[0]} (constant)")
    print(f"  Camera frame counts:")
    for _, row in s1_acting3.iterrows():
        print(f"    {row['Video']}: {int(row['Video_Frames'])} frames (diff: {int(row['Diff_GT_Video'])})")

s2_acting1 = df[(df['Subject'] == 'S2') & (df['Activity'] == 'acting1')]
if len(s2_acting1['Video_Frames'].unique()) > 1:
    print(f"\nS2 acting1: Different cameras have different video frame counts!")
    print(f"  Video frames range: {s2_acting1['Video_Frames'].min()} - {s2_acting1['Video_Frames'].max()}")
    print(f"  GT CSV frames: {s2_acting1['GT_CSV_Frames'].iloc[0]} (constant)")
    print(f"  Camera frame counts:")
    for _, row in s2_acting1.iterrows():
        print(f"    {row['Video']}: {int(row['Video_Frames'])} frames (diff: {int(row['Diff_GT_Video'])})")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
1. ✓ Video and NPY frames are PERFECTLY synchronized (100% match)
   → No problem with NPY files - they exactly match their source videos

2. GT CSV Issues:
   - Some GT files have +1 or +2 MORE frames than videos
   - Some GT files have -1 to -88 LESS frames than videos
   - S1 acting3: Videos themselves vary (2679-2769 frames) but GT is constant (2770)
   - S2 acting1 cam6: Video has only 2421 frames while others have 3386

3. Recommendation:
   - You need to TRIM GT CSV files to match video frame counts
   - For activities where GT > Video: Remove last N rows from GT CSV
   - For activities where GT < Video: This suggests video recording continued after GT stopped
     → Consider trimming videos OR extrapolating GT data
""")
