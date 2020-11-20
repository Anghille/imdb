<<<<<<< HEAD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np


def metricsReport(modelName, test_labels, predictions, verbose=True):
    """
    **Return a computation of `macro/micro f1_score`, `accuracy` and `hamming loss` for multi-class and multi-label model computation.**    

    **Macro**
    Macro-average will compute the metric independently for each class and then take the average (hence treating all classes equally).    

    **Micro**  
    Micro-average will aggregate the contributions of all classes to compute the average metric.  
    In a multi-class classification setup, micro-average is preferable if you suspect there might be class imbalance   
    (i.e you may have many more examples of one class than of other classes).  

    **Hamming Loss**  
    Hamming-Loss is the fraction of labels that are incorrectly predicted, i.e., the fraction of the wrong labels to the total number of labels.   
    Lower the value better the model

    # Parameters
    modelName (str): Name of the model    

    test_labels (numpy.ndarray): Array of binary matrix obtained using sklearn.preprocessing.LabelBinarize() or   
    sklearn.preprocessing.MultiBinarize()   

    predictions (numpy.ndarray): Array result of a sklearn model.predict(X_test)  

    verbose (bool): printout results     

    # Return  
    Accuracy (float):   
    Hamming Loss (float):   
    Macro Precision (float):   
    Micro Precision (float):   
    Macro Recall (float):   
    Micro Recall (float):  
    Macro f1 Score (float):   
    Micro f1 Score (float):   

    # Exemple:   
    > from sklearn.neighbors import KNeighborsClassifier  
    > from sklearn.preprocessing import LabelBinarizer  
    > from sklearn.model_selection import train_test_split  

    > data = pd.DataFrame([["Two imprisoned men bond over a number of years", "R"],
                    ["Nine noble families fight for control over", "PG-13"],
                    ["After training with his mentor", "R"],
                    ["Two detectives, a rookie and a veteran", "PG-13"],
                    ["In Nazi-occupied France during World War II", "R"], 
                    ["Luke Skywalker joins forces with a Jedi Knight" , "PG-13"]])  

    > cleanedTrainData , cleanedTestData = train_test_split(restriction, test_size=0.2, random_state=128, shuffle=True)  

    > vectorizer = TfidfVectorizer() # for converting text into tf-idf-based vectors  
    > vectorised_train_documents = vectorizer.fit_transform(cleanedTrainData["Restriction"])  
    > vectorised_test_documents = vectorizer.transform(cleanedTestData["Restriction"])  

    > le = LabelBinarizer()  
    > train_labels = le.fit_transform(train["Restriction"])  
    > test_labels = le.fit_transform(test["Restriction"])  

    > knnClf = KNeighborsClassifier()  
    > knnClf.fit(vectorised_train_documents, train_labels)  
    > knnPredictions = knnClf.predict(vectorised_test_documents)  

    > metricsReport("knn", test_labels, knnPredictions)  
    """

    model_performance = {}

    # Compute accuracy
    accuracy = accuracy_score(test_labels, predictions)

    # Compute macro f1
    macro_precision = precision_score(test_labels, predictions, average='macro')
    macro_recall = recall_score(test_labels, predictions, average='macro')
    macro_f1 = f1_score(test_labels, predictions, average='macro')

    # Compute micro f1
    micro_precision = precision_score(test_labels, predictions, average='micro')
    micro_recall = recall_score(test_labels, predictions, average='micro')
    micro_f1 = f1_score(test_labels, predictions, average='micro')

    # Compute Hamming loss
    hamLoss = hamming_loss(test_labels, predictions)

    # Printout
    if verbose:
        print("------" + modelName + " Model Metrics-----")
        print("Accuracy: {:.4f}\nHamming Loss: {:.4f}\nPrecision:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nRecall:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nF1-measure:\n  - Macro: {:.4f}\n  - Micro: {:.4f}".format(accuracy, hamLoss, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1))
    model_performance[modelName] = micro_f1

    return accuracy, hamLoss, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1


def plot_search_results(grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()
=======
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np


def metricsReport(modelName, test_labels, predictions, verbose=True):
    """
    **Return a computation of `macro/micro f1_score`, `accuracy` and `hamming loss` for multi-class and multi-label model computation.**    

    **Macro**
    Macro-average will compute the metric independently for each class and then take the average (hence treating all classes equally).    

    **Micro**  
    Micro-average will aggregate the contributions of all classes to compute the average metric.  
    In a multi-class classification setup, micro-average is preferable if you suspect there might be class imbalance   
    (i.e you may have many more examples of one class than of other classes).  

    **Hamming Loss**  
    Hamming-Loss is the fraction of labels that are incorrectly predicted, i.e., the fraction of the wrong labels to the total number of labels.   
    Lower the value better the model

    # Parameters
    modelName (str): Name of the model    

    test_labels (numpy.ndarray): Array of binary matrix obtained using sklearn.preprocessing.LabelBinarize() or   
    sklearn.preprocessing.MultiBinarize()   

    predictions (numpy.ndarray): Array result of a sklearn model.predict(X_test)  

    verbose (bool): printout results     

    # Return  
    Accuracy (float):   
    Hamming Loss (float):   
    Macro Precision (float):   
    Micro Precision (float):   
    Macro Recall (float):   
    Micro Recall (float):  
    Macro f1 Score (float):   
    Micro f1 Score (float):   

    # Exemple:   
    > from sklearn.neighbors import KNeighborsClassifier  
    > from sklearn.preprocessing import LabelBinarizer  
    > from sklearn.model_selection import train_test_split  

    > data = pd.DataFrame([["Two imprisoned men bond over a number of years", "R"],
                    ["Nine noble families fight for control over", "PG-13"],
                    ["After training with his mentor", "R"],
                    ["Two detectives, a rookie and a veteran", "PG-13"],
                    ["In Nazi-occupied France during World War II", "R"], 
                    ["Luke Skywalker joins forces with a Jedi Knight" , "PG-13"]])  

    > cleanedTrainData , cleanedTestData = train_test_split(restriction, test_size=0.2, random_state=128, shuffle=True)  

    > vectorizer = TfidfVectorizer() # for converting text into tf-idf-based vectors  
    > vectorised_train_documents = vectorizer.fit_transform(cleanedTrainData["Restriction"])  
    > vectorised_test_documents = vectorizer.transform(cleanedTestData["Restriction"])  

    > le = LabelBinarizer()  
    > train_labels = le.fit_transform(train["Restriction"])  
    > test_labels = le.fit_transform(test["Restriction"])  

    > knnClf = KNeighborsClassifier()  
    > knnClf.fit(vectorised_train_documents, train_labels)  
    > knnPredictions = knnClf.predict(vectorised_test_documents)  

    > metricsReport("knn", test_labels, knnPredictions)  
    """

    model_performance = {}

    # Compute accuracy
    accuracy = accuracy_score(test_labels, predictions)

    # Compute macro f1
    macro_precision = precision_score(test_labels, predictions, average='macro')
    macro_recall = recall_score(test_labels, predictions, average='macro')
    macro_f1 = f1_score(test_labels, predictions, average='macro')

    # Compute micro f1
    micro_precision = precision_score(test_labels, predictions, average='micro')
    micro_recall = recall_score(test_labels, predictions, average='micro')
    micro_f1 = f1_score(test_labels, predictions, average='micro')

    # Compute Hamming loss
    hamLoss = hamming_loss(test_labels, predictions)

    # Printout
    if verbose:
        print("------" + modelName + " Model Metrics-----")
        print("Accuracy: {:.4f}\nHamming Loss: {:.4f}\nPrecision:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nRecall:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nF1-measure:\n  - Macro: {:.4f}\n  - Micro: {:.4f}".format(accuracy, hamLoss, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1))
    model_performance[modelName] = micro_f1

    return accuracy, hamLoss, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1


def plot_search_results(grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()
>>>>>>> c485c9bae7741cdb37361aed20ef3e5a11bd0611
