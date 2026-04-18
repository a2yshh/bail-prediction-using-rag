from datasets import load_dataset
ds = load_dataset("Exploration-Lab/IL-TUR", "bail")
print(ds)
print(ds["train_all"][0]) 
