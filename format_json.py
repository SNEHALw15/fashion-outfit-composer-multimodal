import json

with open(r"C:\Users\dream\Datasets\polyvore_kaggle\polyvore_outfits\polyvore_item_metadata.json", "r") as f:
    data = json.load(f)

with open(r"C:\Users\dream\Datasets\polyvore_kaggle\polyvore_outfits\formatted_metadata.json", "w") as f:
    json.dump(data, f, indent=4)
    