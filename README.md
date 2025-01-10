# Application-of-NLP-in-a-Book-Review

## Introduction
In the science of artificial intelligence, natural language processing, or NLP, has become a vital area of study since it allows machines to comprehend, interpret, and produce human language. Natural Language Process (Natural Language Process) is the use of procedures, frameworks, and technologies that enable computers to comprehend and react to spoken and written language in a manner that closely resembles that of humans (Ramanathan, 2024). With the exponential growth of textual data on the internet and other digital platforms, NLP techniques have become indispensable for tasks ranging from sentiment analysis and text classification to machine translation and chatbots. Deep learning, machine learning, statistics, and computational linguistics are required to process NLP (Ramanathan, 2024).
Sentiment analysis, also known as opinion mining, is a crucial area within the field of Natural Language Processing (NLP) that focuses on identifying and extracting subjective information from text data. Sentiment analysis can be defined as the area of research that examines how individuals see, feel, and evaluate towards entities expressed in written form (Liu, 2015). 

This report aims to explore the application of NLP techniques in analyzing customer sentiment from Amazon’s top 100 bestselling books. The following sections will provide an in-depth examination of related work in the field, the dataset(s) utilized, the methodology employed, key findings, discussion of results, conclusions drawn, and recommendations for future research.

## Related Work
Sentiment analysis has been an active area of research within the field of Natural Language Processing (NLP) for several decades. Numerous methodologies and techniques have been developed to improve the accuracy and efficiency of sentiment analysis systems. This section reviews three significant studies that have contributed to the advancement of sentiment analysis.
- (Srujan et al., 2018): This study was talked about the classification of amazon book reviews based on sentiment analysis. In this study, Srujan et al. (2018) mentioned that they use some machine learning classifiers, such as k-Nearest Neighbour (KNN), Decision Tree, Support Vector Machine, Naïve Bayes, and Random Forest. Random forest algorithm got the best accuracy compared to others algorithm (Srujan et al., 2018). 
- (Fasha et al., 2022): Fasha et al. (2022) proposed a sentiment analysis model to examine the public’s  dominant feeling or reaction towards a book called The Way Through the Woods: Of Mushrooms and Mourning. Fasha et al. (2022) using TextBlob and VADER as the text mining tools. The results of opinion mining shows 89 positive sentiments, 5 neutral, and 28 negative sentiments using VADER while TextBlob results depicts 99 positive sentiments, 5 neutral, and 18 negative sentiments. Fasha et al. (2022) statet that these two sentiment analysis tools may have difficulties to comprehend some unique comments and humor.
- (Kastrati et al., 2020): Kastrati et al. (2020) proposed a sentiment analysis model to analyze student’s opinion and feedback towards MOOC (Massive Open Online Courses). ….

## About the Data
### Dataset description
The primary dataset used for this study is the Top 100 Bestselling Books Reviews on Amazon, sourced from Kaggle. There are two datasets and, in this study, only use the customer review dataset. In customer review datasets, there are 920 consumer’s review and provide 9 columns, such as:
-	Book name: The title of the book
-	Review title: The title for the review
-	Reviewer: The name who give the review
-	Reviewer rating: The rating of the book given by the reader
-	Review description : The whole review from the reader
-	Is_verified: Boolean is the review was verified or not
-	Date: Date the review was given
-	Timestamp : Time stamp the review was given
-	ASIN: Amazon Standard Identification Number
 
Figure 1. Dataset Info
### Data Preprocessing
To make sure that the dataset are effective to process, cleaning the data are necessary. There are no Missing Value on this dataset. Because this study only needs the review description, drop some columns like ASIN, timestamp, date, is_verified, reviewer, and review title are needed.

The next step is removing punctuation and URLs in the review description. Removing punctuation marks and URLs are important to do sentiment analysis because it will improve tokenization and reduce the noise, so the model will easier to learn and the text can be more cleaner. It also needed to convert the review text to be all lowercase. It is important to make the model learn better and more consistent.  

The polarity from each review are needed, so the model can classify which review sounds positive or negative. To get the polarity value, the TextBlob sentiment are applied on each review description. TextBlob works by giving three scores from each review. If the polarity scores are below 0, that is means that the comments tend to be negative. If the polarity scores are more than 0, that is means that the comments tend to be positive. And, if the polarity scores are 0, that is means that the comments tend to be neutral. After implement the polarity and give the sentiment for each review, it is shows that the positive, neutral, and negative values are imbalance. But because the total of the neutral reviews is really small, is it necessary to drop all the neutral sentiment so the dataset will be only containing positive reviews and negative reviews.
 
Figure 2. Positive sentiment and Negative Sentiment of the review
Before making a Machine Learning model and Deep Learning model, some preprocessing the text data is a critical step. Because of that, tokenization, generating bigrams and trigrams, implementing lemmatization, and removing stop words. Tokenization is the practice of substituting distinct identification symbols for sensitive data while maintaining all pertinent data attributes and maintaining data security (Lutkevich). It transforms a raw text into a format that can be analysed by machine learning models. It’s simplified text data and making it easier to handle by the model. Bigrams are pairs of consecutive words, and trigrams are triplets of consecutive words in a text. For example, from the sentence "I love eating", the bigrams are ["I love", "love eating"] and the trigrams are ["I love eating"]. Using bigrams and trigrams can improve the model’s understanding of the text by providing more context than unigrams (single words), leading to better performance in tasks like sentiment analysis. According to Cambridge Dictionary, lemmatization is the process of reducing the different forms of a word to one single form. In Natural Language Processing, lemmatization helps in reducing the dimensionality of the text data by grouping different forms of the same word into a single form. Stop words are a list of terms that are often used (Ganesan). Removing stop words helps in reducing the noise in the text data, allowing the model to focus on the more meaningful words that contribute to the sentiment. 
After preprocessing all the text reviews, word clouds are implemented to summarize the main sentiment from the book reviews.
 
Figure 3. Word cloud from the book review

## Methodology
### Feature Engineering
To convert the text data into a format that suitable for machine learning models and deep learning models, this study employed some various feature engineering techniques.
-	TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF was used to weigh the importance of words based on their frequency in a document relative to their frequency in the entire corpus.
-	SMOTE (Synthetic Minority Oversampling Technique)
SMOTE is an oversampling technique where the synthetic samples are generated for the minority class. This technique is useful for addressing class imbalance in the dataset.
### Machine Learning
Before creating the model, first the text data need to convert into TF-IDF and then split into train and test set. Since the data is not balanced yet, it is necessary to apply SMOTE to balance the training data.
- Logistic Regression Classifier
As the base model, first import sklearn libraries that is LogisticRegression. 
- Random Forest with Grid Search Hyperparameter Tuning
First, import sklearn libraries such as RandomForestClassifier and GridSearchCV to implement the random forest model. Next, choose the parameters for the random forest and then implement the grid search and set the scoring into accuracy.
- Naïve Bayes with Grid Search Hyperparameter Tuning
First, import sklearn libraries such as MultinomialNB to implement the naïve bayes model. Next, choose the parameters for the naïve bayes and then implement the grid search and set the scoring into accuracy.
- Support Vector Machine with Grid Search Hyperparameter Tuning
First, import sklearn libraries such as SVC and GridSearchCV to implement the SVM model. Next, choose the parameters for the SVM and then implement the grid search and set the scoring into accuracy.
- k-Nearest Neighbors with Grid Search Hyperparameter Tuning
First, import sklearn libraries such as KNeighborsClassifier and GridSearchCV to implement the KNN model. Next, choose the parameters for the KNN and then implement the grid search and set the scoring into accuracy.
### Deep Learning
- Convolutional Neural Networks
First, load the dataset and preprocess the text data using Tokenizer from Keras to convert text to sequences of integers and pad_sequences to ensure all the sequences have the same length. Then define a CNN model architecture with an embedding layer, a 1D convolutional layer, global max pooling, dense layers with ReLU activation, dropout for regularization, and a final dense layer with a sigmoid activation function for binary classification. Compile the model with the Adam optimizer and binary cross-entropy loss function. Then train the model on the training data for 10 epochs with a batch size of 64.
- Long Short-Term Memory Networks
First, load the dataset and preprocess the text data using Tokenizer from Keras to convert text to sequences of integers and pad_sequences to ensure all the sequences have the same length. Then define an LSTM model architecture with an embedding layer, an LSTM layer, dense layers with ReLU activation, and dropout for regularization. Compile the model with the Adam optimizer and binary cross-entropy loss function. Then train the model on the training data for 10 epochs with a batch size of 64.
- Reccurent Neural Networks
First, load the dataset and preprocess the text data using Tokenizer from Keras to convert text to sequences of integers and pad_sequences to ensure all the sequences have the same length. Then define an RNN model architecture with an embedding layer, a SimpleRNN layer, dense layers with ReLU activation, and dropout for regularization. Compile the model with the Adam optimizer and binary cross-entropy loss function. Then train the model on the training data for 10 epochs with a batch size of 64.
- Gated Reccurent Unit Networks
First, load the dataset and preprocess the text data using Tokenizer from Keras to convert text to sequences of integers and pad_sequences to ensure all the sequences have the same length. Then define a GRU model architecture with an embedding layer, a GRU layer, dense layers with ReLU activation, and dropout for regularization. Compile the model with the Adam optimizer and binary cross-entropy loss function. Then train the model on the training data for 10 epochs with a batch size of 64.


## Findings
### Machine Learning
#### Logistic Regression Classifier

 
Figure 4. Confusion Matrix of Logistic Regression Classifier
From the confusion matrix, we observe the following:
•	True Negatives (TN): The model correctly predicted 1 instance as False.
•	False Positives (FP): The model incorrectly predicted 11 instances as True when they were actually False.
•	False Negatives (FN): The model incorrectly predicted 0 instances as False when they were actually True.
•	True Positives (TP): The model correctly predicted 167 instances as True.
 
Figure 5. Classification Report of Logistic Regression Classifier
Using the confusion matrix, we calculated the following performance metrics:
•	Accuracy: The overall accuracy of the model is approximately 94%, indicating that the model correctly classified 94% of the instances.
•	Precision (for True class): The precision is approximately 94%, meaning that 94% of the instances predicted as True are actually True.
•	Recall (Sensitivity for True class): The recall is 100%, indicating that the model correctly identified all True instances.
•	F1-Score: The F1-Score is approximately 97%, which provides a balance between precision and recall.

#### Random Forest with Grid Search Hyperparameter Tuning
 
Figure 6. Confusion Matrix of Random Forest with Grid Search
From the confusion matrix, we observe the following:
•	True Negatives (TN): The model correctly predicted 0 instance as False.
•	False Positives (FP): The model incorrectly predicted 12 instances as True when they were actually False.
•	False Negatives (FN): The model incorrectly predicted 0 instances as False when they were actually True.
•	True Positives (TP): The model correctly predicted 167 instances as True.

 
Figure 7. Classification Report of Random Forest

Using the confusion matrix, we calculated the following performance metrics:
•	Accuracy: The overall accuracy of the model is approximately 93%, indicating that the model correctly classified 93% of the instances.
•	Precision (for True class): The precision is approximately 93%, meaning that 93% of the instances predicted as True are actually True.
•	Recall (Sensitivity for True class): The recall is 100%, indicating that the model correctly identified all True instances.
•	F1-Score: The F1-Score is approximately 96%, which provides a balance between precision and recall.

#### Naïve Bayes with Grid Search Hyperparameter Tuning
 
Figure 8. Confusion Matrix of Naive Bayes with Grid Search
From the confusion matrix, we observe the following:
•	True Negatives (TN): The model correctly predicted 0 instance as False.
•	False Positives (FP): The model incorrectly predicted 12 instances as True when they were actually False.
•	False Negatives (FN): The model incorrectly predicted 1 instances as False when they were actually True.
•	True Positives (TP): The model correctly predicted 166 instances as True.

 
Figure 9. Classification Report of Naive Bayes

Using the confusion matrix, we calculated the following performance metrics:
•	Accuracy: The overall accuracy of the model is approximately 93%, indicating that the model correctly classified 93% of the instances.
•	Precision (for True class): The precision is approximately 93%, meaning that 93% of the instances predicted as True are actually True.
•	Recall (Sensitivity for True class): The recall is 100%, indicating that the model correctly identified all True instances.
•	F1-Score: The F1-Score is approximately 96%, which provides a balance between precision and recall.

#### Support Vector Machine with Grid Search Hyperparameter Tuning
 
Figure 10. Confusion Matrix of SVM with Grid Search
From the confusion matrix, we observe the following:
•	True Negatives (TN): The model correctly predicted 0 instance as False.
•	False Positives (FP): The model incorrectly predicted 12 instances as True when they were actually False.
•	False Negatives (FN): The model incorrectly predicted 0 instances as False when they were actually True.
•	True Positives (TP): The model correctly predicted 167 instances as True.

 
Figure 11. Classification Report of SVM
Using the confusion matrix, we calculated the following performance metrics:
•	Accuracy: The overall accuracy of the model is approximately 93%, indicating that the model correctly classified 93% of the instances.
•	Precision (for True class): The precision is approximately 93%, meaning that 93% of the instances predicted as True are actually True.
•	Recall (Sensitivity for True class): The recall is 100%, indicating that the model correctly identified all True instances.
•	F1-Score: The F1-Score is approximately 97%, which provides a balance between precision and recall.

#### k-Nearest Neighbors with Grid Search Hyperparameter Tuning
 
Figure 12. Confusion Matrix of k-Nearest Neighbors with Grid Search
From the confusion matrix, we observe the following:
•	True Negatives (TN): The model correctly predicted 1 instance as False.
•	False Positives (FP): The model incorrectly predicted 11 instances as True when they were actually False.
•	False Negatives (FN): The model incorrectly predicted 2 instances as False when they were actually True.
•	True Positives (TP): The model correctly predicted 165 instances as True.

 
Figure 13. Classification Report of KNN
Using the confusion matrix, we calculated the following performance metrics:
•	Accuracy: The overall accuracy of the model is approximately 93%, indicating that the model correctly classified 93% of the instances.
•	Precision (for True class): The precision is approximately 94%, meaning that 94% of the instances predicted as True are actually True.
•	Recall (Sensitivity for True class): The recall is 100%, indicating that the model correctly identified all True instances.
•	F1-Score: The F1-Score is approximately 96%, which provides a balance between precision and recall.

### Deep Learning
#### Convolutional Neural Networks
 
Figure 14. Confusion Matrix of CNN
From the confusion matrix, we observe the following:
•	True Negatives (TN): The model correctly predicted 0 instance as False.
•	False Positives (FP): The model incorrectly predicted 12 instances as True when they were actually False.
•	False Negatives (FN): The model incorrectly predicted 0 instances as False when they were actually True.
•	True Positives (TP): The model correctly predicted 167 instances as True.
 
Figure 15. Model Accuracy and Model Loss of CNN
For the model accuracy, the training accuracy starts very high (close to 1.0) and remains almost constant throughout the epochs. This indicates that the model performs very well on the training data from the beginning. 

For the model loss, the training loss starts very low (close to 0) and remains almost constant, indicating that the model has a very low error on the training data. The validation loss starts higher and increases slightly over the epochs, remaining relatively high compared to the training loss.

The high validation loss compared to the training loss further indicates overfitting, as the model's performance on the unseen validation data is not improving and even slightly worsening over time.

Test Accuracy: 0.9329608678817749

The accuracy of this model is 93.2%.

#### Long Short-Term Memory Networks
 
Figure 16. Confusion Matrix of LSTM
From the confusion matrix, we observe the following:
•	True Negatives (TN): The model correctly predicted 0 instance as False.
•	False Positives (FP): The model incorrectly predicted 12 instances as True when they were actually False.
•	False Negatives (FN): The model incorrectly predicted 0 instances as False when they were actually True.
•	True Positives (TP): The model correctly predicted 167 instances as True.

 
Figure 17. Model Accuracy and Model Loss of LSTM
For the model accuracy, the training accuracy starts lower, then quickly rises and stabilizes at around 0.94 before increasing sharply at the last epoch. The validation accuracy starts and remains constant at around 0.94 throughout the epochs.

For the model loss, the training loss starts high, around 0.5, and decreases steadily to around 0.1 by the final epoch, indicating that the model is learning and minimizing error on the training data. The validation loss starts around 0.35, decreases slightly, and then fluctuates around 0.3, with a slight increase towards the end.
Test Accuracy: 0.9329608678817749

The accuracy of this model is 93.3%.

#### Reccurent Neural Networks
 
Figure 18. Confusion Matrix of RNN

From the confusion matrix, we observe the following:
•	True Negatives (TN): The model correctly predicted 0 instance as False.
•	False Positives (FP): The model incorrectly predicted 12 instances as True when they were actually False.
•	False Negatives (FN): The model incorrectly predicted 0 instances as False when they were actually True.
•	True Positives (TP): The model correctly predicted 167 instances as True.

 
Figure 19. Model Accuracy and Model Loss of RNN
For the model accuracy, the training accuracy starts lower, then quickly rises and stabilizes at around 0.94 before increasing at the last epoch. The validation accuracy starts and remains constant at around 0.93 throughout the epochs.
For the model loss, the training loss starts high, around 0.35, and decreases steadily to around 0.1 by the final epoch, indicating that the model is learning and minimizing error on the training data. The validation loss starts around 0.35, decreases slightly, and then fluctuates around 0.3, with a slight increase towards the end.
Test Accuracy: 0.9329608678817749

The accuracy of this model is 93.3%.

#### Gated Reccurent Unit Networks
 
Figure 20. Confusion Matrix of GRU
From the confusion matrix, we observe the following:
•	True Negatives (TN): The model correctly predicted 1 instance as False.
•	False Positives (FP): The model incorrectly predicted 11 instances as True when they were actually False.
•	False Negatives (FN): The model incorrectly predicted 0 instances as False when they were actually True.
•	True Positives (TP): The model correctly predicted 167 instances as True.

 
Figure 21. Model Accuracy and Model Loss of GRU

For the model accuracy, the training accuracy starts lower, then rises and stabilizes at around 0.94 before increasing at the last epoch. The validation accuracy starts and remains constant at around 0.93 and then rising a little bit until reaching 0.94.
For the model loss, the training loss starts high, around 0.6, and decreases steadily to around 0.1 by the final epoch, indicating that the model is learning and minimizing error on the training data. The validation loss starts around 0.35, decreases slightly, and then fluctuates around 0.3, with a slight increase towards the end.
Test Accuracy: 0.9385474920272827

The accuracy of this model is 93.9%.

## Discussion
### Machine Learning
From the accuracy of this model, the logistic regression model got the biggest rate, which is 94%. 

### Deep Learning
From the accuracy of this model, the Grated Recurent Networks got the biggest rate, which is 94%.

## Conclusions and Recommendations
In this study, we performed sentiment analysis on a book review dataset using a series of preprocessing steps and a variety of machine learning and deep learning models. The preprocessing steps included tokenization, generating bigrams and trigrams, lemmatization, and removing stop words. These steps were crucial in transforming raw text data into a clean and consistent format suitable for model training.

## Reference

`Cambridge Dictionary, Lemmatization.`

`Fasha, E. F. B. K., Keikhosrokiani, P. & Asl, M. P. (2022) Opinion Mining Using Sentiment Analysis: A Case Study of Readers’ Response on Long Litt Woon’s The Way Through the Woods in Goodreads, Advances on Intelligent Informatics and Computing. Lecture Notes on Data Engineering and Communications Technologies, 231-242.`

`Ganesan, K. What Are Stop Words? Available online: https://www.opinosis-analytics.com/knowledge-base/stop-words-explained/#:~:text=Stop%20words%20are%20a%20set,carry%20very%20little%20useful%20information. [Accessed 17/05].`

`Kastrati, Z., Imran, A. S. & Kurti, A. (2020) Weakly Supervised Framework for Aspect-Based Sentiment Analysis on Students’ Reviews of MOOCs. IEEE Access, 8, 106799-106810.`

`Liu, B. (2015) Sentiment Analysis: Mining Opinions, Sentiments, and Emotions. Cambridge: Cambridge University Press.`

`Lutkevich, B. Tokenization. Available online: https://www.techtarget.com/searchsecurity/definition/tokenization [Accessed 18/05].`

`Ramanathan, T. (2024) Natural Language Processing. Available online: https://www.britannica.com/technology/natural-language-processing-computer-science [Accessed 11 March].`

`Srujan, K. S., Nikhil, S. S., Raghav Rao, H., Karthik, K., Harish, B. S. & Keerthi Kumar, H. M. (2018) Classification of Amazon Book Reviews Based on Sentiment Analysis, Information Systems Design and Intelligent Applications. Advances in Intelligent Systems and Computing, 401-411.`


