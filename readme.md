### clothes classification

This dir is for clothes-grained classfication

#### Before training

1. use resnet18 trained on imagenet to evaluate all the images

2. remove the problem images

3. compute the mean and std for parts of the images

4. split the dataset into trainset, testset and valset

#### training

1. try the single-label classfication with precision around 35%

2. try the multi-labels classfication with precision around 30%

3. using resnet18 to do this classfication

4. loss for multi-labels case is BCEloss

5. The result is disppointing
