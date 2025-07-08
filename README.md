 Project Description 
This project is designed to identify the most informative regions in images. It works by dividing images into patches of sizes 64×64, 128×128, or 256×256, and then vectorizing them using a convolutional autoencoder. The compressed representation is extracted from the bottleneck layer of the autoencoder.
After encoding, the dimensionality of each patch vector is reduced using PCA (we keep only the top 5 principal components with the highest variance). As a result, each image patch is represented as a point in a 5-dimensional space.
The most informative patches are detected as anomalies using the Local Outlier Factor (LOF) method.

    Training is handled in train.py

    Evaluation and anomaly detection in test.py

    The model is saved in convautoencodermodel.py

    Results are stored in the Results/ directory
