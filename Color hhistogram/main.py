import cv2
import numpy as np
import os
import time
from sklearn.metrics import precision_recall_fscore_support, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def calculate_histogram(image, pins):
    hist = cv2.calcHist([image], [0, 1, 2], None, [pins, pins, pins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def calculate_metrics(true_labels, predicted_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    return precision, recall, f1

def calculate_auc(true_labels, predicted_labels):
    fp_rate, tp_rate, _ = roc_curve(true_labels, predicted_labels)
    avg_auc = auc(fp_rate, tp_rate)
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
    return avg_auc

def euclidean_distance(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)

def rank_results(distances):
    distances.sort(key=lambda x: x[1])

def image_retrieval(queryset_path, dataset_path, pins,threshold):
    i = 0
    execution_times = []

    true_labels_list = []
    scores_list = []

    for queryImage in os.listdir(queryset_path):
        start_time = time.time()
        if (queryImage.lower().endswith(('.jpg','.jpeg','.png'))):
            queryImage_path = os.path.join(queryset_path,queryImage)
            query_image = cv2.imread(queryImage_path)
            query_hist = calculate_histogram(query_image, pins)

            # Iterate through the images in the dataset
            distances = []
            for image_file in os.listdir(dataset_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(dataset_path, image_file)
                    image = cv2.imread(image_path)
                    image_hist = calculate_histogram(image, pins)

                    # Calculate Euclidean distance as a similarity measure
                    distance = euclidean_distance(query_hist, image_hist)
                    distances.append((image_path, distance))

                    query_name = queryImage_path.split("\\")[len(queryImage_path.split("\\")) - 1]
                    image_name = image_path.split("\\")[len(image_path.split("\\")) - 1]

                    if (query_name == image_name):
                        true_label = 1
                        true_labels_list.append(true_label)
                    else:
                        true_label = 0
                        true_labels_list.append(true_label)

                    predicted_label = 1 if distance <= threshold else 0
                    scores_list.append(predicted_label)


            # Sort the images based on distances
            rank_results(distances)

            end_time = time.time()
            execution_times.append(end_time - start_time)

            # Print relevant image paths based on the threshold
            j = 0
            while(1):
               if (distances[j][1] <= threshold):
                   print(f"{distances[j][0]} - Euclidean Distance: {distances[j][1]}")
               else:
                   break
               j += 1
            i += 1

    avg_auc = calculate_auc(true_labels_list,scores_list)
    avg_execution_time = np.mean(execution_times)

    avg_precision, avg_recall, avg_f1 = calculate_metrics(true_labels_list,scores_list)

    #avg_auc = np.mean(auc_values)
    print(f"Average time       = " +str(avg_execution_time / 60)+" minutes")
    print(f"Average precision  = " + str(avg_precision))
    print(f"Average recall     = " + str(avg_recall))
    print(f"Average f1 score   = " + str(avg_f1))
    print(f"Average AUC        = "+ str(avg_auc))



#main function starts here
print("\n-------------------------")
print("Welcome to my CBIR System")
print("-------------------------\n")
pins = input("\nEnter the pins value: ")
pins = int(pins)
threshold = input("Enter the threshold value: ")
threshold = float(threshold)
# Example usage:
queryset_path = "quires_set"
dataset_path = "data_set"
print("\nExperiment for pins = "+str(pins)+"\n")
image_retrieval(queryset_path, dataset_path, pins,threshold)
