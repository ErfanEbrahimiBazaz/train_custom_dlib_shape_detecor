## Train a custom dlib shape predictor
### Thanks to [Adrian Rosebrock](https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/)
and [link](https://medium.com/datadriveninvestor/training-alternative-dlib-shape-predictor-models-using-python-d1d8f8bd9f5c)
this sample code is written to detect iris. 

Note that the model trained for dlib and saved at iris_predictor.dat is overtraiend. The hyperparameters need to change and the training data set
needs to contain more iris. In my model I have annotated and trained on almost 45 iris images.

Evaluating the model on trianing set results in 0 MAE. Applying it on test iris data set and two sample images shows low accuracy. I suggest reading 
[Adrian Rosebrock](https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/) blog, increasing the data set and tuning
the hyperparameters to resolve this issue.

