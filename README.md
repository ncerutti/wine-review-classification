# Classification of wine reviews

Creation of a simple model to predict the score of a wine from the text in the reviews.
The dataset is constructed using *vv_get.py*. The requests library is used to download reviews from vivino. The code, as it is, is intended to be changed dynamically to select the specific type of wine/area or interest. Please, consider increasing the delay to weigh as little as possible on their servers!

*vv_analyze.py* uses a simple LSTM model to classify the downloaded reviews (sample dataset attached). The accuracy was already satisfactory as it is for the purposes of the original project, so we did not explore the use of more complex models or pre-processing of the data.
