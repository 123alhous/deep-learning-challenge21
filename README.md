# deep-cleaning-challenge

Charity Funding Predictor
Project Overview
This project builds a neural network model to predict the success of charity funding applications. The dataset used in this project is the charity_data.csv, which contains various features related to charity funding applications. The model is built using TensorFlow and Keras and is trained to classify if a charity application will be successful or not.

Dataset
The dataset used for this project can be found here. It includes features such as application type, classification, and several other attributes that contribute to predicting the success of charity applications.

Features:
APPLICATION_TYPE: The type of the application.
CLASSIFICATION: The classification of the application.
IS_SUCCESSFUL: The target variable indicating whether the funding was successful (1) or not (0).
Files
Starter_Code.ipynb: Jupyter Notebook containing the code to preprocess the dataset, build, train, and evaluate the neural network model.
charity_model.keras: Saved neural network model in the Keras format.
Requirements
To run the code, you will need to have the following libraries installed:

tensorflow
pandas
scikit-learn
You can install them using pip:

bash
Copy code
pip install tensorflow pandas scikit-learn
Model Overview
The neural network used in this project consists of three layers:

Input Layer: Takes the preprocessed feature data with 80 units and uses ReLU activation.
Hidden Layer: Contains 30 units with ReLU activation.
Output Layer: Contains 1 unit with a sigmoid activation function to output binary classification (success or failure).
The model is trained using binary cross-entropy loss and the Adam optimizer for 100 epochs.

How to Run the Project
Clone the repository or download the notebook file.
Ensure you have installed the necessary libraries (TensorFlow, Pandas, and Scikit-learn).
Run the Jupyter notebook (Starter_Code.ipynb).
The notebook will:
Load and preprocess the dataset.
Train a neural network model.
Evaluate the model using test data.
Save the trained model as charity_model.keras.
Results
The model was evaluated on test data, achieving an accuracy of approximately 73%. Further tuning or alternative models could improve this result.

Conclusion
This project demonstrates the use of neural networks in predicting charity funding success based on various features. The model shows potential, and future improvements may include hyperparameter tuning and feature engineering for better performance.
