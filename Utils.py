import numpy as np
import os
import glob
import pyvista as pv
import matplotlib.pyplot as plt
from Feature_extractor import features_extractor

def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height.
        https://matplotlib.org/3.3.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def autolabel_horizontal(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height.
        https://matplotlib.org/3.3.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        ax.annotate('{:.2f}'.format(rect.get_width()),
                    xy=(rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    xytext=(20,-6),
                    textcoords="offset points",
                    ha='center', va='bottom')

def load_features(classes,feat_size=None, type="DNA"):
    """Load the features specified in classes

    :param classes: list
        A list containing the names of the classes
    :param feat_size: int, optional
        The size of the features in output
    :param type: string, optional
        The type of the features loaded
    :return: list
        A list containing an array[n_samples, feat_size] for each class
    """
    # Load the features
    features = list()
    for c in classes:
        print("\t",c)
        t = list()
        files = glob.glob(os.path.join('.', 'Dataset', c, "*_{}.npy".format(type)))
        if not files:
            features_extractor([c,])
            files = glob.glob(os.path.join('.', 'Dataset', c, "*_{}.npy".format(type)))
        for file in files:
            f = np.load(file)
            if feat_size is None:
                t.append(f)
            else:
                t.append(f[:feat_size])

        features.append(np.array(t))
    return features



def confusion_matrix(prediction, gound_truth, labels, path, title=""):
    """ Compute the confusion matrix given the prediction and the corresponding ground truth. It also plots the confusion matrix at the path specified

    :param prediction: array[n_prediction]
        The result of the classification
    :param gound_truth: array[n_prediction]
        The gound truth labels in the same order of prediction
    :param labels: list
        A list containing the classes
    :param path: string or os.path
        The path where to save the plots
    :param title: string(optional)
        A string to add to the title of the saved plot
    :return: array[n_labels,n_labels]
        The resulting confusion matrix with prediction on the column and ground truth on rows
    """
    # Compute the confusion matrix
    cm = np.zeros((len(labels), len(labels)))
    for pr, y_te in zip(prediction, gound_truth):
        cm[y_te, pr] += 1.0
    # normalization
    cmm = np.copy(cm)
    for i in range(len(labels)):
        cmm[i,:] = cm[i,:] / sum(cm[i,:])

    # Plot the confusion matrix
    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(13.5, 10.5))
    plt.imshow(cmm, interpolation='nearest', cmap=cmap)


    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, "{:0.2f}".format(cmm[i, j]),
                     horizontalalignment="center",
                     color="white" if cmm[i, j] > 0.7 else "black")

    plt.xlabel("Predicted")
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.ylabel("Real")
    plt.yticks(np.concatenate((np.array([-0.5,]),np.arange(len(labels)),np.array([len(labels)-0.5,]))), ["",] + labels + ["",], rotation=45)
    plt.colorbar()
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(os.path.join(path,"Confusion_matrix"+title+".pdf"))
    plt.close()

    return cm

def metrics(cm,labels, path,title=""):
    """ Compute the accuracy, precision and recall from a confusione matrix. It also plots a bar plot with precision and recall for each class.

    :param cm: array[n_labels,n_labels]
        The confusion matrix
    :param labels: list
        A list containing the classes
    :param path: string or os.path
        The path where to save the plots
    :param title: string(optional)
        A string to add to the title of the saved plot
    :return: float, array[n_labels], array[n_labels]
        The accuracy and the precisions and recalls for each class
    """
    # Accuracy
    accuracy = 100 * np.trace(cm) / float(np.sum(cm))
    # Recall\Precision
    precision = np.ndarray(len(labels))
    recall = np.ndarray(len(labels))
    for i in range(len(labels)):
        if np.sum(cm[:, i]) == 0:
            precision[i] = 0
        else:
            precision[i] = cm[i, i] / np.sum(cm[:, i])
        if np.sum(cm[i,:]) == 0:
            recall[i] = 0
        else:
            recall[i] = cm[i, i] / np.sum(cm[i, :])

    # Plot
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    fig.set_figheight(13)
    fig.set_figwidth(15.5)
    rect = ax.bar(x - 0.35/2, precision, 0.35, label='Precision')
    autolabel(rect,ax)
    rect = ax.bar(x + 0.35/2, recall, 0.35, label='Recall')
    autolabel(rect, ax)
    ax.set_title('Accuracy : {:.2f}'.format(float(accuracy)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=45)
    ax.legend()
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(os.path.join(path, "Metrics"+title+".pdf"))
    plt.close()
    return accuracy,precision,recall

def plot_results(results, path):
    """ Plots the accuracy, average prediction and average recall for each method in results

    :param results: dict
        A dict containing different methods as keys and for each key an array with accuracy, average prediction and average recall.
    :param path: string or os.path
        The path where to save the plots
    :return:
    """
    # Plot
    methods = np.array(list(results.keys()))
    acc = np.ndarray((len(results)))
    precision = np.ndarray((len(results)))
    recall = np.ndarray((len(results)))
    for i in range(len(methods)):
        # r = results[methods[i]]
        acc[i] = results[methods[i]][0]
        precision[i] = results[methods[i]][1]
        recall[i] = results[methods[i]][2]
    x = np.arange(len(methods))

    fig, ax = plt.subplots()
    fig.set_figheight(13)
    fig.set_figwidth(15.5)
    rect = ax.bar(x - 0.2, acc, 0.2, label='Accuracy')
    autolabel(rect, ax)
    rect = ax.bar(x, precision, 0.2, label='Precision')
    autolabel(rect, ax)
    rect = ax.bar(x + 0.2, recall, 0.2, label='Recall')
    autolabel(rect, ax)
    ax.set_title('Results')
    ax.set_xticks(x)
    ax.set_xticklabels(methods,rotation=45)
    ax.set_ybound(0, 105)
    ax.legend()
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(os.path.join(path, "Results.pdf"))
    plt.close()
    return

def plot_results_classes(results, classes, path):
    """ Plots the f1 measure of the methods for each class in results

    :param results: dict
        A dict containing different methods as keys and for each key an array with accuracy, average prediction and average recall.
    :param path: string or os.path
        The path where to save the plots
    :return:
    """
    # Plot
    methods = np.array(list(results.keys()))
    f1_classes = np.ndarray((len(classes),methods.shape[0]))
    for i in range(len(methods)):
        # r = results[methods[i]]
        f1_classes[:,i] = results[methods[i]] * 100
    f1_classes = np.concatenate((f1_classes,np.mean(f1_classes,axis=0,keepdims=True)),axis=0)
    x = np.arange(len(classes)+1)

    fig, ax = plt.subplots()
    fig.set_figheight(13)
    fig.set_figwidth(15.5)
    for i in range(methods.shape[0]):
        rect = ax.barh(x - (1 / (methods.shape[0]+1)) * (i+1) +0.5 , f1_classes[:,i], 1 / (methods.shape[0]+1), label=methods[i])
        autolabel_horizontal(rect, ax)

    ax.set_title('F1 score per class')
    ax.set_yticks(x)
    ax.set_yticklabels(classes+["Average",],rotation=45)
    ax.set_xbound(0, 110)
    ax.legend()
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(os.path.join(path, "Results_class.pdf"))
    plt.close()

def plot_prediction(index, prediction, gound_truth, labels, path, title=""):
    """Plots a picture for each class with the mesh and the labels predicted for it in red if it is wrong and green if it is correct.

    :param index: array[n_labels]
        An array containing the index of the test meshes for each class
    :param prediction: array[n_prediction]
        The result of the classification
    :param gound_truth: array[n_prediction]
        The gound truth labels in the same order of prediction
    :param labels: list
        A list containing the classes
    :param path: string or os.path
        The path where to save the plots
    :return:
    """
    os.makedirs(os.path.join(path,"Qualitative"), exist_ok=True)
    i_prediction = 0
    err = prediction == gound_truth

    for inds,l in zip(index,labels):
        plotter = pv.Plotter(shape=(1,index[0].shape[0]), off_screen=False, window_size=[1024, 1024 // 2])
        plotter.set_background("white")
        j=0
        offs = glob.glob(os.path.join('.', 'Dataset', l, "*.off"))
        for i in inds:

            mesh = pv.read(offs[i])
            plotter.subplot(0,j)
            plotter.add_text(labels[prediction[i_prediction]], color="green" if err[i_prediction] else "red", font_size=10)
            plotter.add_mesh(mesh,smooth_shading=True, color="grey")
            j+=1
            i_prediction+=1

        plotter.save_graphic(os.path.join(path,"Qualitative", l +title+ '.pdf'))
    pv.close_all()
    return
