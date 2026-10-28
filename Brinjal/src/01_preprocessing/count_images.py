import os

def count_images(directory):
    counts = {}
    for d in os.listdir(directory):
        dir_path = os.path.join(directory, d)
        if os.path.isdir(dir_path):
            count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            counts[d] = count
    return counts

print("Original Brinjal dataset:")
original_counts = count_images("Brinjal")
for cls, count in sorted(original_counts.items()):
    print(f"  {cls}: {count}")

print("\nSAM-cropped Brinjal dataset:")
sam_counts = count_images("Brinjal_leaf_cropped")
for cls, count in sorted(sam_counts.items()):
    print(f"  {cls}: {count}")

print("\nComparison:")
for cls in sorted(original_counts.keys()):
    orig = original_counts.get(cls, 0)
    sam = sam_counts.get(cls, 0)
    diff = sam - orig
    print(f"  {cls}: {orig} -> {sam} ({diff:+d})")
