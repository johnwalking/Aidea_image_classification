# Aidea_image_classification

1. rename.py 
for the merge of two class.
```
python3 rename.py
```

2. pyhton resnet18.py {batch_size} {data_ratio}
for training
example:
```
python3 resnet18.py 32 0.5
```

3. python vit.py  {batch_size} {data_ratio}
for training
example:
```
python3 vit.py 32 0.5
```

4. try the sam optimizer
watch  https://github.com/moskomule/sam.pytorch
for training
example:
```
python3 sam_vit.py 32 0.5
```

5.for final test:
```
python3 classift.py 32
```
