# Summary
This is a neural network model for the application of predicting the spread bias of multiple instruments such as MTGEFNCL Index, LRC30APR Index, 5yr Treasury, and 5yr Eris SOFR swap. The .ipynb/py ML model code begins with taking a dataframe that does not contain the yields spreads of the instruments but utilizes the daily historical data that is present to calculate them individually for each pair of instruments being compared. A new dataframe is created that incorporates the newly-calculated yield spreads to be at the beginning with n amount of columns (where n = i x j with i being the # of instruments and j the # of spreads for the respective instrument). The rest of the columns of the new dataframe contain other historical and sentimental analysis data. The data is converted to a NumPy array that is split into the yield spreads and market information (i.e. not the yield spreads) but is then joined to be converted to a tensor and split into training, validation, and testing sets. Scaling of something like scikit-learn MinMaxScaler for training, validation, and test sets does not happen as features that will me inputted are similar.

The neural network for spread bias is then created; the neural network incoporates the columns of the dataframe where yield spreads are first indluded and 2 hidden linear layers that are subject to change manually given number of dataframe columns; there are also 2 rectified linear unit (ReLU) activation functions that take the inputs and outputs of the layers; when the neural network is called it takes the training, validation, or test data of the concatenated yield spreads and market info to ouput the yield spreads (i.e. manually input n in the hyperparameters). 

The Neural network, a criterion using Mean Squared Error loss, and a optimizer utulizing Adam are assigned (or created). Test set is fed through the training loop where the very standard items of gradients, modeling, outputs, and loss are calculted. Similary, the validation test is also fed through the training loop but without the calculation gradients as it is not needed. Test set is run by similar way of the validation. Ultimately, the trained model is used to make predictions on the test data. Average predicted yield spread for each instrument is calculated. Instruments are ranked based on the average predicted yield spreads. Best and worst performing instruments are identified and printed.

At some point throughout the model, it is saved along with a checkpoint that takes many parameters for later to be loaded again without dealing with the hassle of relearning.

# Issues
Calculation of yield spreads has to be done manually as dataframe has to find all the columns by the concatanation of the field name and field mnemomic. Incomplete columns where data does not start collecting until much much later (i.e. multiple 5yr Treasury and 5yr Eris SOFR swap tickers) means that only one instrument was picked for each and not a aggregation of all of them, but were instead utilized as market information that helps predict the yield spreads of the main instruments that do contain most (if not all) of the data. Hyperparameters section of the model needs to be manually tuned in-regard to the explicit values of hidden_size_1, hidden_size_2, and learning_rate on the basis of the csv that is pulled from.

# Future
Calculation of yield spreads to be done in a more intuitive way. Incorporate some sort of imputation such as MICE to take care of data that starts later in a better way by filling hundreds of rows of missing data in a non-linear way to possibly be incorporated into a aggregation of a single instrument or just better maket information. Find a way to pull articles that use tf-idf to keen in on terms that may negatively or postively affect the prediction the yield spreads.

# How-to-use
Note that if on google.colab, want to enable GPU utilization for faster processing by going to Edit -> Notebook Settings -> Hardware Accelerator -> GPU

Before anything, have a google drive with a folder such as BBGdatasets where keep csv datasets wanting to load, and within the BBGdatasets, have another folder MLmodels where states of the neural network model will be saved. Will want to copy paths and replace with my ipynb/py code.

First, have a proper formatted csv file with the daily historical and sentimental analysis data such as the 'PX_LAST' and 'PX_SETTLE' for each instrument, security, and violotality rates among others. The csv should be indexed by the daily dates and be multi-header with columns having first header being the Field and the second one being the Field Mnemomic. Make sure to group by field. Showing dates such as beginning and end time will not allow for data to be grabbed and transformed. I have attached a sample csv that contains 389 rows and 52 columns of daily historical and sentimental analysis data to see the format of the csv that should be introduced.

In actual step by step terms, if calculating yield spreads of just MTGEFNCL/LRC30APR/GT5/USGG5YR, input correctly formated csv -> yield spreads for each instrument is automatically calculated -> the number of yield spreads is automatically inputed n and everything else is taken care of -> possibly, adjust hidden_size_1, hidden_size_2, and learning_rate to better optimize model to the yield spreads -> output will be a combination of epochs, training loss, validation loss, and test loss as well as ranking of which instrument is predicted to perform best in-regard to their respective yield spreads.

In actual step by step terms, if calculating yield spreads different from just (MTGEFNCL/LRC30APR/GT5/USGG5YR), input correctly formated csv -> calculate yield spreads of instruments that are not just MTGEFNCL/LRC30APR/GT5/USGG5YR similar to code in ml model, for example, for 6 instruments number of yield spreads will be 6 * 5 = 30 where this is n mentioned above, so should have 30 different columns at end of original dataframe -> input the number of yield spreads into n and everything else for the most part will be taken care of -> adjust hidden_size_1, hidden_size_2, and learning_rate to optimize model to the new yield spreads -> output will be a combination of epochs, training loss, validation loss, and test loss as well as ranking of which instrument is predicted to perform best in-regard to their respective yield spreads.

# Postface
Many complicated and incorrect ml models among other types were created and tested for the main goal of spread performance prediction, also, data was plotted and regressed for the purpose of investigating trends; however, the model necessitated a very straightford and direct approach as in essence it is just taking yield spreads of a multitude of instruments and utilzing other market information to see which instrument performs best.

# Dependecies
* google.colab
* Python 3
* Numpy
* Pandas
* Scikit-Learn
* torch
