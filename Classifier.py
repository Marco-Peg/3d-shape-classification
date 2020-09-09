import argparse
from math import floor
import numpy as np
import os
import glob
from sklearn.decomposition import PCA

import Utils
import Models

KERNEL = ['linear', 'rbf', 'rbf']
C = [1, 1, 8]

METRICS = ['sqeuclidean', 'canberra', 'sqeuclidean']
K = [3,3,5]


def main(DNA_size=20, test_perc=20, PCA_size=0, n_partition=5):
    # Load the dataset
    print("Loading dataset ...", end='')
    classes = list()
    with open(os.path.join('.', 'Dataset', "Classes.txt"), "r") as fo:
        for line in fo:
            classes.append(line[:-1])
    print("done")

    # Load the features
    print("Loading features ...",end='')
    feats = Utils.load_features(classes,DNA_size, type="DNA")
    print("done")

    predictions = dict()

    # Classification on 4 different train and test set
    for n_test in range(n_partition):
        if n_test == 0:
            write_mode="w"
        else:
            write_mode = "a"
        print("\n###TEST", n_test, "###")
        i_train = list()
        i_test = list()
        shapes_test = list()
        first = True
        for i in range(len(classes)):
            f=feats[i]
            i_split = floor(f.shape[0] * (test_perc) / 100)
            i_test.append(np.arange(n_test * i_split,(n_test+1) * i_split))
            i_train.append(np.setdiff1d(np.arange(f.shape[0]), np.array(i_test)))
            offs = glob.glob(os.path.join('.', 'Dataset', classes[i], "*.off"))
            if first:
                feat_train = f[i_train[-1],:]
                feat_test = f[i_test[-1],:]
                labels_train = i * np.ones_like(i_train[-1])
                labels_test = i * np.ones_like(i_test[-1])
                shapes_test = [offs[j] for j in i_test[-1]]
                first=False
            else:
                feat_train = np.concatenate((feat_train,f[i_train[-1], :]))
                feat_test = np.concatenate((feat_test,f[i_test[-1], :]))
                labels_train = np.concatenate((labels_train,i * np.ones_like(i_train[-1])))
                labels_test = np.concatenate((labels_test,i * np.ones_like(i_test[-1])))
                shapes_test = shapes_test + [offs[j] for j in i_test[-1]]

        results_folder = 'Results'

        # PCA
        if PCA_size and PCA_size < DNA_size:
            # normalize
            import sklearn.preprocessing as pr
            print("PCA ...", end='')
            pca = PCA(PCA_size)
            pca.fit(feat_train)
            feat_train = pca.transform(pr.normalize(feat_train))
            feat_test = pca.transform(pr.normalize(feat_test))
            print("done")
            results_folder += 'PCA'

        # Classification
        for metric,k in zip(METRICS,K):
            print("\n###",metric,"_", k,"###")
            method = "KNN_" + metric + "_" + str(k)
            print("Classification ...", end='')

            predicted = Models.KNN_classifier(feat_train, labels_train, feat_test,metric=metric, K=k)
            print("done")

            print("Results...")
            dir_res = os.path.join('.', results_folder, method)
            os.makedirs(dir_res, exist_ok=True)

            # Confusion matrix
            cm = Utils.confusion_matrix(predicted,labels_test,classes,dir_res, title=str(n_test))
            accuracy, precision, recall = Utils.metrics(cm,classes,dir_res, title=str(n_test))
            f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))

            # Qualitative results
            Utils.plot_prediction(i_test,predicted,labels_test,classes,dir_res, title=str(n_test))
            print("done")

            # Log
            with open(os.path.join(dir_res,"LOG.txt"), write_mode) as output:
                output.write("##TEST" + str(n_test) + "##\n")
                output.write("MODEL:SVM\n")
                output.write("METRIC:"+metric+"\n")
                output.write("K:" + str(k) + "\n")
                output.write("TEST_SHAPES:{:s}\n".format(";".join(shapes_test)))
                output.write("ACCURACY:{:.2f}\n".format(accuracy))
                output.write("PRECISION(mean):{:.2f}\n".format(100 * np.mean(precision)))
                output.write("RECALL(mean):{:.2f}\n".format(100 * np.mean(recall)))
            print("ACCURACY:", accuracy)
            if n_test == 0:
                # results[method] = np.array([accuracy,100 * np.mean(precision),100 * np.mean(recall)])
                # results_f1[method] = f1
                predictions[method] = [ predicted, labels_test ]
            else:
                # results[method] = results[method] + np.array([accuracy,100 * np.mean(precision),100 * np.mean(recall)])
                # results_f1[method] = results_f1[method] + f1
                predictions[method] = [np.concatenate((predictions[method][0], predicted)), np.concatenate((predictions[method][1], labels_test))]

        for kernel,c in zip(KERNEL,C):
            print("\n###",kernel,str(c),"###")
            method = "SVM_" + kernel + "_" + str(c)
            print("Classification ...", end='')

            predicted = Models.SVM_multiclass(feat_train, labels_train, feat_test, C=c, kernel=kernel, max_iteration=1000)
            print("done")

            print("Results...")
            dir_res = os.path.join('.', results_folder, method)
            os.makedirs(dir_res, exist_ok=True)
            # Confusion matrix
            cm = Utils.confusion_matrix(predicted,labels_test,classes,dir_res, title=str(n_test))
            accuracy, precision, recall = Utils.metrics(cm,classes,dir_res, title=str(n_test))

            f1 = np.nan_to_num(2* (precision * recall) / (precision+recall))

            # Qualitative results
            Utils.plot_prediction(i_test,predicted,labels_test,classes,dir_res, title=str(n_test))
            print("done")

            # Log
            with open(os.path.join(dir_res,"LOG.txt"), write_mode) as output:
                output.write("##TEST"+str(n_test)+"##\n")
                output.write("MODEL:SVM\n")
                output.write("KERNEL:"+kernel+"\n")
                output.write("C:" + str(c) + "\n")
                if kernel == 'poly':
                    output.write("DEGREE:" + str(3) + "\n")
                output.write("TEST_SHAPES:{:s}\n".format(";".join(shapes_test)))
                output.write("ACCURACY:{:.2f}\n".format(accuracy))
                output.write("PRECISION(mean):{:.2f}\n".format(100 * np.mean(precision)))
                output.write("RECALL(mean):{:.2f}\n".format(100 * np.mean(recall)))
            print("ACCURACY:", accuracy)
            if n_test == 0:
                predictions[method] = [predicted, labels_test]
            else:
                predictions[method] = [np.concatenate((predictions[method][0], predicted)), np.concatenate((predictions[method][1], labels_test))]

    # Plot results
    with open(os.path.join('.', results_folder, 'LOG.txt'), "w") as output:
        if PCA_size and PCA_size<DNA_size:
            output.write("\Features type: ShapeDNA + PCA\n")
        else:
            output.write("\Features type: ShapeDNA\n")
        output.write("\Features Size:{:}\n".format(DNA_size))
        if PCA_size and PCA_size<DNA_size:
            output.write("\PCA Size:{:}\n".format(PCA_size))

    results = dict()
    results_f1 = dict()
    for k in predictions:
        dir_res = os.path.join('.', results_folder, k)
        cm = Utils.confusion_matrix(predictions[k][0], predictions[k][1], classes, dir_res)
        accuracy, precision, recall = Utils.metrics(cm, classes, dir_res)
        f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))

        print("\n###", k, "###")
        print("ACCURACY:", accuracy)

        results[k] = np.array([accuracy,100 * np.mean(precision),100 * np.mean(recall)])
        results_f1[k] = f1
        with open(os.path.join('.', results_folder, 'LOG.txt'), "a") as output:
            output.write(k + "\n")
            output.write("\tACCURACY:{:.2f}\n".format(accuracy))
            output.write("\tPRECISION(mean):{:.2f}\n".format(100 * np.mean(precision)))
            output.write("\tRECALL(mean):{:.2f}\n\n".format(100 * np.mean(recall)))

    Utils.plot_results(results,os.path.join('.', results_folder))
    Utils.plot_results_classes(results_f1, classes,os.path.join('.', results_folder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Features extractor')
    parser.add_argument("-d", "--DNA_size", default=11,  type=int, help="Size of the ShapeDNA desciptor.")
    parser.add_argument("-t", "--test_perc", default=20,  type=float,
                        help="Percentage of test samples.", dest="test_perc")
    parser.add_argument("-p", "--PCA_size", default=0, type=int,
                        help="Size of the PCA reduction. If 0, is not applied.", dest="PCA_size")
    parser.add_argument("-n", "--n_partition", default=5, type=int,
                        help="Number of partition for the cross-validation.", dest="n_partition")

    args = vars(parser.parse_args())
    main(**args)

