import numpy as np

with np.load("flow_dataset-Copy.npz") as flow_dataset:
    maps = flow_dataset["maps"]
    labels = np.ones(maps.shape[0])

order2 = np.argsort(np.random.random(maps.shape[0]))
special_train_data = maps[order2]
special_train_labels = labels[order2]
print(special_train_data.shape[0])
print("----------------------shape[0]")
print(special_train_data.shape[1])
print("----------------------shape[1]")
print(special_train_data.shape[2])
print("----------------------shape[2]")
# print(special_train_labels[0])