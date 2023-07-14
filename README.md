# Fake_news_detection
In this project, we will leverage machine learning techniques to build a model capable of distinguishing between real and fake news articles. The model will analyze various textual features of the news articles and learn patterns that differentiate genuine news from deceptive or misleading content. This project was done as part of Intel Unnati Industrial training programme.

Link to download dataset - https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/


You can clone this repository using the below code

```
git clone https://github.com/jobint001/intelunnati_ones_and_zeroes

```

You can clone the docker image using the below code

```
docker pull ghcr.io/jobint001/intelunnati_ones_and_zeroes/fake_news_detection_image:latest


```

# Libraries used

* ``pandas``: Used for data manipulation and analysis.

* ``seaborn``: A data visualization library built on top of Matplotlib.
* ``matplotlib.pyplot``: A plotting library for creating visualizations.
* ``tqdm``: A library for creating progress bars and monitoring iterations.
* ``re``: A module for regular expressions, used for pattern matching and text manipulation.
* ``nltk``: The Natural Language Toolkit, used for natural language processing tasks.
* ``os``: A module providing a way to interact with the operating system.
* ``pickle``: A module for serializing Python objects to disk and deserializing them back into memory.
* ``nltk.tokenize.word_tokenize``: A function for tokenizing text into individual words.
* ``sklearn``: The scikit-learn library, a popular machine learning library in Python.
* ``train_test_split``: A function for splitting data into training and testing sets.
* ``accuracy_score``: A function for calculating the accuracy of a classification model.
* ``sklearnex``: A library for patching scikit-learn to enable new features.
* ``sklearn.linear_model.LogisticRegression``: A class for logistic regression models.
* ``time``: A module for working with time-related functions.
* ``numpy``: A library for numerical computing in Python.
* ``TfidfTransformer``: Transforms a count matrix into a TF-IDF representation.
* ``CountVectorizer``: Converts text documents into a matrix of token counts.
* ``TfidfVectorizer``: Combines CountVectorizer and TfidfTransformer to convert text documents into a matrix of TF-IDF features.


## Collection of data

The data used is from  https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

The data is read as csv file using `` panda.read_csv() ``

## Data cleaning 

Unnecessary column in the data is removed as they do not affect the performance of the model. Fake news is labeled as 0 and True news is labeled as 1. Tiltle, text, label columns are used from the initial dataset.

To check for missing values (NaN) ``  Check_forNAN(data) `` function is used. If any missing values are found `` data.dropna()`` removes those rows.

``tokenize(column)``  takes a column of text data, tokenizes it into individual words, and returns a list of tokens containing only alphabetical words. It can be useful for tasks such as text preprocessing or building language models.

`` remove_stopwords(tokenized_column)``  function is designed to remove stopwords from a tokenized column of text. Stopwords are common words that often do not carry significant meaning in text analysis tasks. The function takes a list of tokenized words as input and removes any words that are present in a predefined set of stopwords for the English language.

``apply_stemming(tokenized_column)`` function is designed to apply stemming to a tokenized column of words. Stemming is a process that reduces words to their base or root form. The function utilizes the PorterStemmer algorithm from the NLTK library, which is commonly used for English stemming. It takes a list of tokenized words as input, applies stemming to each word using the PorterStemmer, and returns a new list containing the stemmed versions of the words.

``rejoin_words(tokenized_column)`` rejoin a tokenized column of words back into a single string. 
``cleaning(news)`` calls all the datacleaning functions.

Word cloud is created for the DataFrame. It shows the most frequent words in the given text data.

## Vectorization

``count_vectorizer.fit_transform(X)`` This line applies the fit_transform() method of the CountVectorizer class to the input X. It tokenizes and counts the words in the text data, generating a matrix of token counts.

``freq_term_matrix = count_vectorizer.transform(X)`` This line applies the transform() method of the CountVectorizer class to the input X. It uses the previously fitted CountVectorizer to transform the text data into a matrix of token counts.

``tfidf = TfidfTransformer(norm="l2")`` This line creates an instance of the TfidfTransformer class with the parameter norm set to "l2", indicating that the TF-IDF normalization should use the L2 norm.

``tfidf.fit(freq_term_matrix)`` This line applies the fit() method of the TfidfTransformer class to the freq_term_matrix, which is the matrix of token counts obtained from the previous step. It calculates the IDF (Inverse Document Frequency) values for each term based on the entire corpus.

``tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)`` This line applies the fit_transform() method of the TfidfTransformer class to the freq_term_matrix. It transforms the matrix of token counts into a TF-IDF representation, incorporating both term frequency and inverse document frequency.

## Splitting of data
The code ``train_test_split(tf_idf_matrix, y, random_state=21)`` is using the ``train_test_split`` function from scikit-learn to split the ``tf_idf_matrix (the feature matrix)`` and y (the target variable) into training and testing sets.

``x_train`` and ``y_train`` represent the training data, which will be used to train a machine learning model.
``x_test`` and ``y_test`` represent the testing data, which will be used to evaluate the performance of the trained model on unseen data.
## Model training, Evaluation, and Prediction
  ### 1. Logistic Regression
  It is a supervised learning algorithm used for binary classification tasks. It is used to predict the probability of an instance belonging to a particular class.
  It can be used when the dependent variable is binary or categorical.Logistic Regression uses the logistic function (sigmoid function) to map the linear 
  combination of the input features to a value between 0 and 1, representing the probability of the positive class.We first used unpatched sklearn for the model      trainning. Accuracy and time taken is calculated. Then the sklearn library is patched using ``patch_sklearn()`` which gives intel optiimisation to sklearn library
  Now the accuracy and time taken is compared and found that the patched sklearn library trained the model faster.

  Evaluations including confusion matrix and ROC curve are drawn based on the predicted data from the model. AUC of ROC graph is 0.98.

  ### 2.Naive Bayes

  Naive Bayes models work well when there are many features (words) and the assumption of feature independence holds reasonably well.
  In the case of fake news detection, Naive Bayes models can effectively capture patterns in the presence or absence of specific words or features that may 
  indicate,fake or legitimate news.With Naive Bayes model the accuracy was lower but the time for training was significantly lower. It has a higher RMSE value      which indicates it is not as good as logistic regression model. ROC graph and confusion matrix is drawn on the predicted data. The AUC of ROC graph came out to   be 0.94 which is lower when compared to 0.98 of logistic regression. 

  ### 3.Decision Tree

  Decision trees provide a clear and interpretable structure that can be easily visualized. Decision trees inherently rank the importance of features based on their ability to split the data effectively. By examining the splits and node impurities within the decision tree, you can identify the most important features for differentiating between fake and legitimate news. 

  Nonlinear correlations and interactions between features can be successfully captured by decision trees. The detection of fake news frequently requires intricate patterns and dependencies between multiple words or phrases. 

  Decision trees can naturally handle both categorical and numerical variables without the need for explicit feature encoding. In the detection of fake news, you may come across a combination of textual, category, and numerical data.
  
  The training accuracy of the decision tree has come out as 99.62 percent. AUC of ROC curve is 1 which indicates a perfect or ideal performance of the classification model. 

### 4.Passive-Aggressive Classifier

PAC is known for its computational efficiency and fast training times. It updates the model based on the disagreement between predictions and true labels, making it well-suited for large-scale dataset. PAC can handle both binary and multiclass classification problems. In fake news detection, you may encounter scenarios where you need to classify news articles into multiple categories or identify different types of fake news. PAC is designed to handle concept drift and can dynamically adjust to new patterns or characteristics in the data.

The training accuracy of the decision tree has come out as 99.58 percent, which is comparable to the accuracy of the Decision tree. The training time for this model has been just 0.42 which is much faster when compared to 23 second training time of decision tree. RMSE value is also lower indicates improved accuracy and reduced prediction errors.  Therefore Passive-Aggressive classifer is taken as the final model.
## Saving model

Using ``pickle `` library is used to save the selected model. This library is also used to load the saved model from the disk for subsequent prediction.

## Using the saved model for fake-news prediction from a new data
``fake_news_det(news)`` function takes an unseen news article and predicts whether it is true or fake using the saved model. The news article is processed in the same way as in the model training section. It has been able to correctly predict the validity of the news.
