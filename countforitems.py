import json

path = r"C:\Users\dream\Datasets\polyvore_kaggle\polyvore_outfits\disjoint\train.json"

with open(path) as f:
    data = json.load(f)

total_items = 0

for outfit in data:
    total_items += len(outfit["items"])

print("Total items:", total_items)