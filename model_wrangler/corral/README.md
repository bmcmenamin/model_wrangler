# model_wrangler/corral

This is the model corral where a bunch of premade models live.


Each of the py files here defines a class for a type of model (e.g., Linear Regression) that has the guts of TensorFlow in it. That class gets passed into the ModelWrangler class, and it is suddenly give the powers of being able to save/load, batch train, predict, measure feature importance, etc.

Current models:
* `linear_regression`: Linear Regression
* `logistic_regression`: Logistic Regression
* `dense_autoencoder`: Is an autoencoder built with densely connected layers
* `convolutional_autoencoder`: [NOT WORKING] Is an autoencoder built with convolutional connected layers
* `dense_feedforward`: Generic densely-connected feedforward model



Future models:
* general conv NN
* siamese training?
* recurrent nets