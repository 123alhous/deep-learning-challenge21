# deep-learning-challenge

### Scenario

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

- `EIN` and `NAME` — Identification columns
- `APPLICATION_TYPE` — Alphabet Soup application type
- `AFFILIATION` — Affiliated sector of industry
- `CLASSIFICATION` — Government organization classification
- `USE_CASE` — Use case for funding
- `ORGANIZATION` — Organization type
- `STATUS` — Active status
- `INCOME_AMT` — Income classification
- `SPECIAL_CONSIDERATIONS` — Special considerations for application
- `ASK_AMT` — Funding amount requested
- `IS_SUCCESSFUL` — Was the money used effectively

------

### Preprocess the Data

Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

- Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
    - What variable(s) are the target(s) for your model?
    - What variable(s) are the feature(s) for your model?
- Drop the `EIN` and `NAME` columns.
- Determine the number of unique values for each column.
- For columns that have more than 10 unique values, determine the number of data points for each unique value.
- Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.
- Use `pd.get_dummies()` to encode categorical variables.
- Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.
- Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.

------

### Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

- Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
- Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
- Create the first hidden layer and choose an appropriate activation function.
- If necessary, add a second hidden layer with an appropriate activation function.
- Create an output layer with an appropriate activation function.
- Check the structure of the model.
- Compile and train the model.
- Create a callback that saves the model's weights every five epochs.
- Evaluate the model using the test data to determine the loss and accuracy.
- Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

-------

### Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    - Dropping more or fewer columns.
    - Creating more bins for rare occurrences in columns.
    - Increasing or decreasing the number of values for each bin.
    - Add more neurons to a hidden layer.
    - Add more hidden layers.
    - Use different activation functions for the hidden layers.
    - Add or reduce the number of epochs to the training regimen.



## <a id="Final-Report"></a> Final Analysis and Report on the Neural Network Model

Given below is my Final Report and Analysis of the Neural Network Model along with answers to the questions posed in the assignment:

1. **Overview** of the analysis: Explain the purpose of this analysis. 
**ANSWER** - The purpose of the model was to create an algorithm to help Alphabet Soup, predict whether or not applicants for funding will be successful. The model was a binary classifier that was able to predict with a fairly high degree of accuracy if the funding will be successful or not.

2. **Results**: Using bulleted lists and images to support your answers, address the following questions.

  * Data Preprocessing
    * What variable(s) are considered the target(s) for your model?
    **ANSWER** - The variable for the Target was identified as the column `IS_SUCCESSFUL`.
    * What variable(s) are considered to be the features for your model?
    **ANSWER** - The following columns were considered as features for the model:
        * `NAME`
        * `APPLICATION_TYPE`
        * `AFFILIATION`
        * `CLASSIFICATION`
        * `USE_CASE`
        * `ORGANIZATION`
        * `STATUS`
        * `INCOME_AMT`
        * `SPECIAL_CONSIDERATIONS`
        * `ASK_AMT`
    * What variable(s) are neither targets nor features, and should be removed from the input data?
    **ANSWER** - The column or variable that can be removed is `EIN` as it is an identifier for the applicant organization and has no impact on the behavior of the model.
    
  * Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    **ANSWER** In the Optimized version of the model, I used **3 hidden layers** each with multiple neurons which increased the accuracy to <75% to 79%. The Initial model had 
    only **2 layers**. Although the number of `epochs` did not change between the Initial and the Optimized Model, adding a 3rd Layer increased the accuracy of the model.
    
    * Were you able to achieve the target model performance?
    **ANSWER** - Yes by optimizing the model, I was able to increase the accuracy from **72%** a little over **79%**.
    
    * What steps did you take to try and increase model performance?
    **ANSWER** The following steps were taken to optimize and increase the performance of the model:
     * Instead of dropping both the `EIN` and `Name` columns, only the `EIN` column was dropped. However, only the names which appeared more than 5 times were considered.
     * Added a 3rd Activation Layer to the model in the following order to boost the accuracy to > 75% :
       * 1st Layer - `relu`
       * 2nd Layer - `tanh`
       * 3rd Layer - `sigmoid`
     * It was observed that instead of both the 2nd and 3rd Layer to be `sigmoid`, when I used the 2nd Layer as `tanh` and the 3rd Layer as `sigmoid` it boosted the performance to beyond **79%**.

3. **Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.

    **Summary and Recommendation**
      * Overall, by optimizing the model we are able to increase the accuracy to above 79%.
      * This means we are able to correctly classify each of the points in the test data 79% of the time. In other words an applicant has a close to 80% chance of being 
        successful if they have the following:

         - The NAME of the applicant appears more than 5 times (they have applied more than 5 times)
         - The type of APPLICATION is one of the following: T3, T4, T5, T6 and T19   
         - The application has the following values for CLASSIFICATION: C1000, C1200, C2000,C2100 and C3000.

    **Alternative Method**
      * Although this model worked very well and provided a great deal of accuracy, an alternative approach to recommend is the `Random Forest` model as it is also suited for 
        classification problems. Using the Random Forest model we can achieve close to 78% accuracy.
