import json

path = r"C:\Users\dream\Datasets\polyvore_kaggle\polyvore_outfits\disjoint\test.json"

with open(path) as f:
    data = json.load(f)

print("Total outfits:", len(data))
print("\nFirst outfit example:\n")
print(data[0])