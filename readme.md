### clothes classification

This dir is for clothes-grained classfication

#### Before training

1. use resnet18 trained on imagenet to evaluate all the images

2. remove the problem images

3. compute the mean and std for parts of the images

4. split the dataset into trainset, testset and valset
