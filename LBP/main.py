import cv2
import numpy as np
import os
import time
from sklearn.metrics import precision_recall_fscore_support, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from skimage.color import rgb2lab

def calculate_ccv(image, pins, spatial_radius=8, color_radius=8):
    lab_image = rgb2lab(image)
    height, width, _ = lab_image.shape

    # Reshape the image to a list of pixels in Lab color space
    pixels = lab_image.reshape((height * width, 3))

    # Quantize the Lab color space
    quantized_pixels = (pixels // pins).astype(np.uint8)

    # Compute the CCV
    ccv = np.zeros((pins, pins, pins))

    for i in range(height * width):
        x, y, z = quantized_pixels[i]
        x = min(x, pins - 1)
        y = min(y, pins - 1)
        z = min(z, pins - 1)
        ccv[x, y, z] += 1

    # Apply spatial and color radius to smooth the CCV
    ccv_smoothed = cv2.GaussianBlur(ccv, (2*spatial_radius + 1, 2*color_radius + 1), 0)

    # Flatten the CCV to a 1D vector
    ccv_flattened = ccv_smoothed.flatten()

    # Normalize the CCV
    ccv_normalized = ccv_flattened / np.sum(ccv_flattened)

    return ccv_normalized


def calculate_metrics(true_labels, predicted_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    return precision, recall, f1

def calculate_auc(true_labels, predicted_labels):
    fp_rate, tp_rate, _ = roc_curve(true_labels, predicted_labels)
    avg_auc = auc(fp_rate, tp_rate)
    return avg_auc, fp_rate, tp_rate

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
    avg_auc, fp_rate, tp_rate = calculate_auc(true_labels, predicted_labels)
    # print results
    print("Average Precision: " + str(avg_precision))
    print("Average Recall: " + str(avg_recall))
    print("Average F1 Score: " + str(avg_f1))
    print("Average AUC: " + str(avg_auc))
    print("Average execution time: " + str(avg_execution) + " minutes")

    # Plot ROC curve
    plt.figure()
    plt.plot(fp_rate, tp_rate, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(avg_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def cbir(queryset_path, dataset_path, pins, threshold):
    print("\nExperiment for pins = " + str(pins) + "\n")

    true_labels = []
    predicted_labels = []
    execution_lists = []

    for queryImage in os.listdir(queryset_path):
        start_time = time.time()
        if (queryImage.lower().endswith(('.jpg', '.jpeg', '.png'))):
            queryImage_path = os.path.join(queryset_path, queryImage)
            query_image = cv2.imread(queryImage_path)
            query_features = calculate_ccv(query_image,pins)

            distances = []
            for image_file in os.listdir(dataset_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(dataset_path, image_file)
                    image = cv2.imread(image_path)

                    db_features = calculate_ccv(image,pins)

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
            print_relevant_images(queryImage_path,distances,threshold)
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
        pins = input("\nEnter the pins value: ")
        pins = int(pins)
        threshold = input("Enter the threshold: ")
        threshold = float(threshold)
        cbir(queries_path, dataset_path,pins,threshold)
        break
    except ValueError:
        print("Invalid input...\n")
