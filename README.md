# Computer-Vision-Assignment-2

An image retrieval system is a technology that retrieves relevant images from a database based on user queries. There are two main types of image retrieval systems: text-based and content-based. Text-based systems use keywords or tags to index and search images. Content-based systems analyze the actual content of the image, to find similar images. It typically involves extracting features from images, such as color, texture, and shape, and indexing them for efficient retrieval. These systems employ techniques like color histograms, edge detection, and deep learning to analyze and compare images. Users input an image, and the system returns images with similar visual content. 

Image retrieval systems are used in a variety of applications, including:
      • Stock photography: Helping users find the perfect image for their project
      • E-commerce: Helping customers find the products they are looking for.
      • Medical imaging: Helping doctors diagnose diseases.
      • Law enforcement: Helping investigators identify suspects.
      • Personal use: Helping people find photos of their friends and family

This assignment aims to develop a functional CBIR system using different color features along with evaluaation of their performance

# Task one

 Build the CBIR system: Design and implement a system architecture for image retrieval using 
color features. Develop functionalities for loading images, extracting features, computing distances, and 
ranking results.

# Task Two

 Implement the CBIR system using Color Histogram as an image representation. Experiment with 
120 pins, 180 pins and with 360 pins. Use Euclidean as distant measure and compute precision, recall, F1 
score, and time for each experiment. Construct a Receiver Operating Characteristic (ROC) curve by varying 
the retrieval threshold. Calculate the Area Under the Curve (AUC) to measure the overall performance 
across different threshold settings. Note that you need to compute these measures as an average of at 
least 10 different quires. 

# Task Three

  3.1: Implement the CBIR system using Color Moments (mean, standard deviation, and skewness) as an 
       image representation. Use Euclidean as distant measure and assign equal weights to each moment. 
       Compute precision, recall, F1 score, and time. Calculate the Area Under the Curve (AUC) to measure the 
       overall performance across different threshold settings. Note that you need to compute these measures 
       as an average of at least 10 different quires. 
       
  3.2: Same as task 3.1 but with different weights. You need to give a weigh relative to the important of the 
       moment. 
       
  3.3: Same as task 3.2 but with the addition of more Moments including Median, Mode, and Kurtosis

# Task Four

Try to improve the performance of the CBIR system using other image representation techniques.

# Dataset

the dataset used for this assignment can be found in the following link: https://www.kaggle.com/datasets/theaayushbajaj/cbir-dataset
