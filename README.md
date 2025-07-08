
### Project Description 

### This project is made to find the most informative parts of an image — the patches that "stand out" or seem unusual compared to the rest.

Here’s how it works:


 ### 1. Split the image into patches

Each input image is automatically split into smaller square patches.
The size of these patches can be:

* 64×64
* 128×128
* or 256×256 pixels

This allows the program to focus on smaller areas of the image.


### 2. Turn patches into vectors using a convolutional autoencoder

Each patch is encoded into a compressed vector using a convolutional autoencoder.
The autoencoder is a neural network that learns how to:

* Compress the patch to a smaller representation
* Then reconstruct it again

We only keep the compressed vector from the middle (bottleneck) layer, because it captures the most important features of the patch.


### 3. Reduce the vector size with PCA

The compressed vectors are still a bit large.
So, we use PCA (Principal Component Analysis) to reduce their size.
We only keep the top 5 components — the directions with the highest variance.
This gives us a 5-dimensional vector for each patch.

### 4. Detect which patches are "different" using LOF

Now, every patch is just a point in 5D space.

We assume that normal patches are similar to each other, and informative patches look different.

To find the different ones (anomalies), we use a method called LOF (Local Outlier Factor).

These are the most informative patches — maybe they show rare objects, details, or abnormalities.


###  Files used in this project:

* `train.py` – trains the autoencoder model
* `test.py` – runs inference: splits the image, encodes patches, applies PCA & LOF
* `convautoencodermodel.py` – defines the autoencoder architecture
* Output results are saved in the `Results/` folder

