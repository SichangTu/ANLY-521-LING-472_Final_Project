# ANLY-521-LING-472_Final_Project

This project aims to identify spam messages using Natural Language Processing (NLP) protocols. The goal is to fit and train models that will distinguish and filter spam SMS messages with a substantial accuracy.

Data is from the [SMS Spam Collection Data](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/).

## Requirements
The project will run on Python 3.6+. Required libraries:

* numpy
* pandas
* xgboost
* sklearn
* TensorFlow 2.X

You should be able to manually install these libraries via pip if necessary.

Example:

`$ python3 -m pip install --upgrade tensorflow`

For testing the installation, you can use:

`$ python3 -c 'import tensorflow; print(tensorflow.__version__)'`

## ML_models.py

`ML_models.py` contains 4 machine learning models, namely Support Vector Machine (SVM), Random forest (RF), Multinomial Naive Bayes (MNB) and XGBoost, mostly using the implementation in scikit-learn. Each model is trained with 4 different features, including CountVectorizer, TfidfVectorizer, length of message and length of punctuations, to examine the best combination. 

Example usage:

`$ python ML_models.py --datafile SMSSpamCollection.txt`

The running time depends on the CPU type on your computer. And RF model may take longer time to train.

## Neural_Networks.py

`Neural_Networks.py` contains 2 neural network models, namely TextCNN and TextRNN. Different from the previous machine learning approach, text massages are first cleaned and padded into sequences. Then they are fed into the two models to make the predictions.

Example usage:
`$ python Neural_Networks.py --datafile SMSSpamCollection.txt`