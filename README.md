## Train a custom dlib shape predictor
### Thanks to [Adrian Rosebrock](https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/)
and [link](https://medium.com/datadriveninvestor/training-alternative-dlib-shape-predictor-models-using-python-d1d8f8bd9f5c)
this sample code is written to detect iris. 

Note that the model trained for dlib and saved at iris_predictor.dat is overtraiend. The hyperparameters need to change and the training data set
needs to contain more iris. In my model I have annotated and trained on almost 45 iris images.

Evaluating the model on trianing set results in 0 MAE. Applying it on test iris data set and two sample images shows low accuracy. I suggest reading 
[Adrian Rosebrock](https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/) blog, increasing the data set and tuning
the hyperparameters to resolve this issue.


## Initiate the training set

Annotate your training set by [imgab tool](https://imglab.in/#).

## Change the paths and run the following command to train your dlib shape predictor:

python training_dlib_shape_detector.py --training C:/Users/E17538/"OneDrive - Uniper SE"/Desktop/DailyActivities/FAD/acv6/ACS_S6/iris/iris.xml --model iris_predictor.dat

It will save your trained model in iris_predictor.dat. Depending on your CPU and the number of cores, it may take a few minutes. In my case with
tree_depth = 8 and 6 threads, it took around 6 minutes.

## Check the training loss and test loss by running the following command:

python evaluate.py --predictor iris_predictor.dat --xml C:/Users/E17538/"OneDrive - Uniper SE"/Desktop/DailyActivities/FAD/acv6/ACS_S6/iris_test/iris_test.xml

In my case, with my test data set, I got:
[INFO] evaluating shape predictor...
[INFO] error: 5.294233442980803

## Apply it on a sample image to see visually how well your model is working: Change the path to your sample image(s) and run iris_detector.py.




