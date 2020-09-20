**About different dataset**
* BaseDataset: Common functions for preprocessing data.
    * np2tensor: Convert numpy data from HWC[0: 255] to CHW[0: 1].
    * augment: Randomly vertical flip, horizontal flip and 90 rorate data. 
    * get\_patch: Randomly crop a patch.
    * add\_noise: Add gaussion or possion noise. 
* DIV2KDataset: Train or validate on DIV2K.
* TrainLRHRDataset: Train or validate on your own dataset. You should keep HR and LR img name the same, or you can modify \_\_getitem\_\_().
* TestLRDataset: Test dataset, return only LR. 
* TestLRHRDataset: Test dataset, return LR, HR and img name.
