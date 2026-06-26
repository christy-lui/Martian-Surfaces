# Mars HiRISE Image Classification

This repository contains the Jupyter Notebook and data processing pipeline for a machine learning midterm project. The focus of the code is parsing, validating, and preprocessing a dataset of HiRISE (High Resolution Imaging Science Experiment) planetary surface images (`map-proj-v3`). 

## Project Overview

This project uses TensorFlow and Keras to prepare image data for a deep learning model. The pipeline efficiently maps labels to images, cleans missing values, and utilizes TensorFlow's `tf.data.Dataset` API to batch data. This approach keeps the GPU properly utilized and improves overall training stability.

---

## Technologies Used

* **Deep Learning:** `tensorflow`, Keras (`ImageDataGenerator`, `Adam` optimizer, `layers`, `regularizers`)
* **Data Manipulation:** `pandas`, `numpy`, `os`
* **Machine Learning Utilities:** `scikit-learn` (`train_test_split`, `shuffle`, `confusion_matrix`, `classification_report`, `compute_class_weight`)
* **Data Visualization:** `matplotlib.pyplot`, `seaborn`

---

## Data Pipeline

The notebook executes the following data preparation steps:

1.  **Label Parsing:** Reads image class mappings from `labels-map-proj-v3.txt` into a Pandas DataFrame and casts the classes to integer types.
2.  **Data Validation:** Drops any `NaN` values and verifies that every image file physically exists in the directory and has a file size greater than 0.
3.  **Shuffling:** Randomizes the dataset using `scikit-learn`'s shuffle function with a fixed random state (`22`) to prevent order bias.
4.  **Image Processing:** Loads the raw files, decodes the RGB JPEG images, resizes them to `128x128` pixels, and normalizes the pixel values to a `0.0 - 1.0` range.
5.  **Dataset Optimization:** Builds a `tf.data.Dataset` pipeline that maps the processing functions, caches the data, shuffles it, applies a batch size of 128, and prefetches the images for maximum performance.

---

## Current Status & Next Steps

The data preprocessing pipeline is functioning and yields a processed batch of 128 images with a shape of `(128, 128, 128, 3)`. 

**Pending Tasks:** * The immediate next step identified in the project notes is to plot one sample photo from each distinct image class.
* Implement the neural network architecture, train the model using the batched dataset, and evaluate its performance.