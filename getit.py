import json

base_path = r"C:\Users\dream\Datasets\polyvore_kaggle\polyvore_outfits\disjoint"

for file in ["train.json", "valid.json", "test.json"]:
    full_path = base_path + "\\" + file
    
    with open(full_path) as f:
        data = json.load(f)
    
    print(file, len(data))