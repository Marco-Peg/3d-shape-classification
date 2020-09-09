# 3d shape classification

Code for the project of 3d shape classification.  The main packages used in this projects are: [sklearn](http://scikit-learn.org), [numpy](https://numpy.org/) and [igl](https://libigl.github.io/libigl-python-bindings/). The dataset is composed of shapes taken from: [TOSCA Non-rigid world](http://tosca.cs.technion.ac.il/book/resources_data.html) and [Princeton's Benchmark for 3D Mesh Segmentation](https://segeval.cs.princeton.edu/public/Download/off.zip)

**You will need Python3+ to use this project.**

My goal is to classify 10 different classes of 3d shapes saved as meshes. For each mesh I computed their ShapeDNA from the  Laplace–Beltrami operator. I tested the ShapeDNAs as features with 2 different models (KNN,SVM) each one with 3 variants of parameters. For each method, I summarised the results in a LOG file and I plotted the confusion matrix and 2 bar plot with accuracy, precision, recall and F1 score. 

### Precompute the features
```bash
python Feature_extractor.py -c Female Male Gorilla Armadillo Teddy Fourleg Ant Octopus Bird Glasses
```
This computes the ShapeDNA for each sample in the dataset and saves it in a .npy file in the same path of their mesh.

### Classify and get prediction results.
```bash
python Classifier.py
```
This loads or computes the features and tests different classification methods. It builds a folder for each classification variant contatining the confusions matrix, the bar plots of the metrics, a LOG file and the qualitative results. At the end it saves two bar plot and a LOG file to summarise all the tests.