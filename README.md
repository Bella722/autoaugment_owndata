# autoaugment_owndata
train best augment policies on own dataset,and generate augmented images for kinds of using(src  trans1  trans2  trans3)  

<img src="https://github.com/Bella722/autoaugment_owndata/raw/main/test/bike0.png" width="180" height="105"><img src="https://github.com/Bella722/autoaugment_owndata/blob/main/results/bike0_0.jpg" width="180" height="105"><img src="https://github.com/Bella722/autoaugment_owndata/blob/main/results/bike0_1.jpg" width="180" height="105"><img src="https://github.com/Bella722/autoaugment_owndata/blob/main/results/bike0_2.jpg" width="180" height="105">  
<img src="https://github.com/Bella722/autoaugment_owndata/raw/main/test/bike1.png" width="180" height="105"><img src="https://github.com/Bella722/autoaugment_owndata/blob/main/results/bike1_0.jpg" width="180" height="105"><img src="https://github.com/Bella722/autoaugment_owndata/blob/main/results/bike1_1.jpg" width="180" height="105"><img src="https://github.com/Bella722/autoaugment_owndata/blob/main/results/bike1_2.jpg" width="180" height="105">  
<img src="https://github.com/Bella722/autoaugment_owndata/raw/main/test/bike2.png" width="180" height="105"><img src="https://github.com/Bella722/autoaugment_owndata/blob/main/results/bike2_0.jpg" width="180" height="105"><img src="https://github.com/Bella722/autoaugment_owndata/blob/main/results/bike2_1.jpg" width="180" height="105"><img src="https://github.com/Bella722/autoaugment_owndata/blob/main/results/bike2_2.jpg" width="180" height="105">
# how to use
1. put your data into dirctory like:  
    ```bash
   ├── run.py  
   ├── data_deal.py  
   ├── transformations.py  
   ├── utils_paths.py  
   └── own_data  
      ├── classA  
        ├── img1  
        ├── img2  
        └── ...  
      ├── classB  
        ├── img1  
        ├── img2  
        └── ...  
      └── ...
    ```
   
 2. reset your CHILD_BATCH_SIZE/CONTROLLER_EPOCHS/nclass/imagePaths
```python
  # run.py
  CHILD_BATCH_SIZE = 128
  CHILD_BATCHES = len(Xtr) // CHILD_BATCH_SIZE
  CHILD_EPOCHS = 120
  CONTROLLER_EPOCHS = 50 # 15000 or 20000
  nclass = 8
  # data_deal.py
  imagePaths = sorted(list(utils_paths.list_images('own_data')))
```

3. strat trainning
```bash
  python run.py
```
 
4. define your polisies like:
```python
  ###use your trainned policies in generate.py  
  def best_policy():
    policy = [
        # Subpolicy 1
        'Operation 14  P=0.600 M=0.044  Operation  1  P=0.900 M=-0.167',
        'Operation 12  P=0.400 M=1.500  Operation  0  P=0.700 M=-0.033',
        'Operation  3  P=0.300 M=-0.150  Operation 14  P=0.100 M=0.067',
        'Operation  7  P=0.400 M=0.111  Operation  7  P=0.400 M=0.222',
        'Operation  4  P=0.200 M=3.333  Operation  6  P=0.200 M=0.667',
        # Subpolicy 2
        'Operation  4  P=0.000 M=10.000  Operation 14  P=0.800 M=0.111',
        'Operation  5  P=0.300 M=0.778  Operation 13  P=0.600 M=0.500',
        'Operation  9  P=1.000 M=4.444  Operation  7  P=0.200 M=0.667',
        'Operation  4  P=0.800 M=16.667  Operation  8  P=0.500 M=170.667',
        'Operation  7  P=0.600 M=0.556  Operation 15  P=1.000 M=0.089',
        # Subpolicy 3
        'Operation 15  P=0.100 M=0.133  Operation 12  P=0.300 M=0.500',
        'Operation  5  P=0.700 M=0.667  Operation  9  P=0.100 M=7.556',
        'Operation  3  P=0.400 M=-0.350  Operation  3  P=0.500 M=0.150',
        'Operation  3  P=0.800 M=-0.150  Operation  5  P=0.400 M=1.000',
        'Operation  8  P=0.600 M=227.556  Operation  2  P=0.300 M=-0.150',
        # Subpolicy 4
        'Operation 13  P=0.100 M=1.500  Operation  4  P=0.100 M=-3.333',
        'Operation  0  P=0.600 M=0.233  Operation 12  P=0.100 M=0.700',
        'Operation  3  P=0.000 M=-0.450  Operation  2  P=0.800 M=-0.350',
        'Operation  5  P=0.100 M=0.222  Operation  9  P=1.000 M=5.333',
        'Operation  6  P=0.500 M=0.333  Operation  1  P=0.100 M=-0.033',
        # Subpolicy 5
        'Operation 13  P=0.400 M=1.300  Operation  6  P=0.200 M=0.111',
        'Operation 10  P=0.500 M=0.700  Operation  1  P=0.100 M=-0.100',
        'Operation 12  P=0.500 M=1.700  Operation  9  P=0.400 M=6.222',
        'Operation 15  P=0.600 M=0.133  Operation 11  P=0.400 M=1.500',
        'Operation 14  P=0.300 M=0.200  Operation  1  P=0.800 M=0.100',
    ]
    return policy
```

5. generate transformed images
```bash
python generate.py
```
