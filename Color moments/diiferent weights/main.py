import cv2
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from scipy.stats import skew
import time


def calculate_color_moments(image, weights):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate mean, standard deviation, and skewness for each channel (H, S, V)
    mean_values = np.mean(hsv_image, axis=(0, 1))
    std_dev_values = np.std(hsv_image, axis=(0, 1))
    skewness_values = skew(hsv_image, axis=(0, 1))  # Use skew from scipy.stats

    # Apply weights to each moment
    mean_values *= weights[0]
    std_dev_values *= weights[1]
    skewness_values *= weights[2]

    # Concatenate the weighted feature vectors
    color_moments = np.concatenate((mean_values, std_dev_values, skewness_values))

    return color_moments


def calculate_metrics(true_labels, predicted_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    return precision, recall, f1

def calculate_auc(true_labels, predicted_labels):
    fp_rate, tp_rate, _ = roc_curve(true_labels, predicted_labels)
    avg_auc = auc(fp_rate, tp_rate)
    return avg_auc

def euclidean_distance(features1, features2):
    return np.linalg.norm(features1 - features2)

def rank_results(distances):
    distances.sort(key=lambda x: x[1])

def print_relevant_images(query_path,distances,threshold):
    print("Related images for "+query_path)
    #sort distances based on the euclidean distances
    rank_results(distances)
    j = 0
    while (1):
        if (distances[j][1] <= threshold):
            print(f"{distances[j][0]} - Euclidean Distance: {distances[j][1]}")
        else:
            break
        j += 1
    print("\n")

def print_metrics(true_labels,predicted_labels,execution_lists):
    # Calculate average metrics
    avg_precision, avg_recall, avg_f1 = calculate_metrics(true_labels, predicted_labels)
    # calculated average execution time (divide it by 60.0 to get a result in minutes)
    avg_execution = np.mean(execution_lists) / 60.0
    # Calculate AUC
    avg_auc = calculate_auc(true_labels, predicted_labels)
    # print results
    print("Average Precision: " + str(avg_precision))
    print("Average Recall: " + str(avg_recall))
    print("Average F1 Score: " + str(avg_f1))
    print("Average AUC: " + str(avg_auc))
    print("Average execution time: " + str(avg_execution) + " minutes")

def cbir(queryset_path, dataset_path, threshold):

    true_labels = []
    predicted_labels = []
    execution_lists = []

    for queryImage in os.listdir(queryset_path):
        start_time = time.time()
        if (queryImage.lower().endswith(('.jpg', '.jpeg', '.png'))):
            queryImage_path = os.path.join(queryset_path, queryImage)
            query_image = cv2.imread(queryImage_path)
            query_features = calculate_color_moments(query_image,(0.2,2.0,5.0))

            distances = []
            for image_file in os.listdir(dataset_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(dataset_path, image_file)
                    image = cv2.imread(image_path)

                    db_features = calculate_color_moments(image,(0.2,2.0,5.0))

                    # Use Euclidean distance as the similarity measure
                    distance = euclidean_distance(query_features,db_features)
                    distances.append((image_path,distance))

                    query_name = queryImage_path.split("\\")[len(queryImage_path.split("\\")) - 1]
                    image_name = image_path.split("\\")[len(image_path.split("\\")) - 1]

                    if (query_name == image_name):
                        true_label = 1
                        true_labels.append(true_label)
                    else:
                        true_label = 0
                        true_labels.append(true_label)

                    predicted_label = 1 if distance <= threshold else 0
                    predicted_labels.append(predicted_label)

            end_time = time.time()
            execution_time = end_time - start_time
            execution_lists.append(execution_time)
            #print_relevant_images(queryImage_path,distances,threshold)
    print_metrics(true_labels,predicted_labels,execution_lists)

##########################################################################

#main function starts here
print("\n-------------------------")
print("Welcome to my CBIR System")
print("-------------------------\n")

dataset_path = "data_set"
queries_path = "quires_set"

while(1):
    try:
        threshold = input("Enter the threshold: ")
        threshold = float(threshold)
        break
    except ValueError:
        print("Invalid input...\n")
cbir(queries_path, dataset_path, threshold)