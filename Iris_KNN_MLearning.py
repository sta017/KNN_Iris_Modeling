import pandas as pd
from sklearn import datasets   #could not use statsmodels.api for import

    ##1. Importing data set
iris = datasets.load_iris()
    ## this data set is a list of dictionary like 'data' : containing sepal and petal dimensions
                                         ##     'target' : containing the label assignment 0,1,2 for setosa, versicolor and virginica.
                                        ##   'target_names': the names of the correspondig flowers.
                                        ##    and more other dictionary.

    ## 2. creating a data frame from the dictionary
species= [iris.target_names[x] for x in iris.target]    #making a list of the labels in the 'Species' column
iris = pd.DataFrame(iris['data'], columns= ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris['Species'] = species

    ## finding the data types of each column.
print(iris.dtypes)

    ## 3. Deter number of unique categories and number of cases for each category,
        ## for the label variable.
iris['count'] = 1          #CREATING  a new column 'count' listing number of unique row element.
print(iris[['Species','count']].groupby('Species').count())  ## counting each case of flower.

def plot_iris(iris, col1, col2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.lmplot(x=col1, y=col2,
               data = iris,
               hue = 'Species',
               fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Iris species shown by color')
    plt.show()

##plot_iris(iris, 'Petal_Width', 'Sepal_Length')
##plot_iris(iris, 'Sepal_Width', 'Sepal_Length')


        ## 4. SCALING THE DATA
from sklearn.preprocessing import scale
import pandas as pd
num_cols = ['Sepal_Length', 'Sepal_Width', 'Petal_Length','Petal_Width']
iris_scaled = scale(iris[num_cols])                                             ## recall, in R, to scale, iris_scale <- lapply(iris[,num_cols], scale)
iris_scaled = pd.DataFrame(iris_scaled, columns = num_cols)
print('\n Zscore Normalization results: \n'+ str(iris_scaled.describe().round(3)))

        ##Methods in the scikit-learn package requires numeric numpy arrays as arguments.
        ## thus, the strings for Species must be RECODED as numbers using
        ## using  DICTIONARY lookup.

levels = {'setosa':0, 'versicolor':1, 'virginica':2}
iris_scaled['Species'] = [levels[x] for x in iris['Species']]
iris_scaled.head()

        ## 5. SPLITTING the DATA: with BERNOULLI SAMPLING

from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(1234)
iris_split= train_test_split(np.asmatrix(iris_scaled),test_size=75)    #divided into equal halves of 75 cases each.
                                                        ## notice, the division is exact number, unlike in R
                                                        ## in R recall, partition= createDataPartition(iris, times= 1, p=0.4, list= FALSE ) 'p' being the
                                                        ## fraction of the whole dataset.  or simply  train=sample_frac(iris, p=0.4) and test = iris[-as.numeric(rownames(train.iris)),]
                                                         
iris_train_features = iris_split[0][:, :4]             ##extracting only the four columns with features
iris_train_labels = np.ravel(iris_split[0][:, 4])      ## taking only the column with the label 'Species'
                                                    ## np.ravel()    gives 75*4 =300 cases
iris_test_features = iris_split[1][:, :4]
iris_test_labels = iris_split[1][:, 4]

print('\n The Dimension of the training dataset: '+ str(iris_train_features.shape))
print(iris_train_labels.shape)
print('\n The Dimension of the test dataset: ' + str(iris_test_features.shape))
print(iris_test_labels.shape)



        ## 6. Defining and Training the KNN MODEL.

from sklearn.neighbors import KNeighborsClassifier
KNN_mod = KNeighborsClassifier(n_neighbors = 3)                                              ##R-equiv    knn_mod= kknn(Species ~ ., train=iris_train, test=iris_test, k=3)
KNN_mod.fit(iris_train_features, iris_train_labels)                                          


iris_test = pd. DataFrame(iris_test_features, columns = num_cols)
iris_test['predicted'] = KNN_mod.predict(iris_test_features)                                    ##R-equiv next step    test$predicted = predict(knn_mod)
iris_test['correct']= [1 if x==z else 0 for x, z in zip(iris_test['predicted'], iris_test_labels)]
accuracy = 100.0 * float(sum(iris_test['correct']))/float(iris_test.shape[0])
print(' \n The accuracy of the KNN model prediction= ' + str(round(accuracy,2))+ '\n')

levels = {0:'setosa', 1:'versicolor', 2:'virginica'}
iris_test['Species'] = [levels[x] for x in iris_test['predicted']]
markers = {1:'^', 0:'o'}                                            # a dictionary {}
colors= {'setosa':'blue', 'versicolor':'green', 'virginica':'red'}  # a dictionary {}
def plot_shapes(df, coL1, coL2, markers, colors):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = plt.figure(figsize = (6,6)).gca()  #defining plot axis
    for m in markers:        # iterate over marker dictionary keys
        for c in colors:    #iterate over  color dictionary keys
            df_temp = df[(df['correct']== m) & (df['Species']== c)]
            sns.regplot(x = coL1, y= coL2,
                        data= df_temp,
                        fit_reg = False,
                        scatter_kws = {'color': colors[c]},
                        marker = markers[m],
                        ax = ax)
    plt.xlabel(coL1)
    plt.ylabel(coL2)
    plt.title('Iris species by color')
    plt.show()
    return 'Done'


plot_shapes(iris_test,'Petal_Width','Sepal_Length',markers,colors)
plot_shapes(iris_test, 'Sepal_Width', 'Sepal_Length', markers, colors)














































