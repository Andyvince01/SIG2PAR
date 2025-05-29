import cv2, csv, argparse, random
import numpy as np
from PIL import Image
from tqdm import tqdm

import os
import shutil
import numpy as np

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--data", type=str, default='foo_test.csv', help="Dataset labels")
    parser.add_argument("--images", type=str, default='foo_test/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results.csv', help="CSV file of the results")
    args = parser.parse_args()
    return args

args = init_parameter()

# Here you should initialize your neural network

import torch
from src.models import SIG2PAR

model = SIG2PAR(load_weights=True)
checkpoint = torch.load('/user/avitale/PAR2025/SIG2PAR/runs/2025-05-28_07-25-18/best_model.pth', map_location='cuda', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Reading CSV test file
with open(args.data, mode='r') as csv_file:
    gt = csv.reader(csv_file, delimiter=',')
    gt_num = 0
    gt_dict = {}
    for row in gt:
        gt_dict.update({row[0]: [int(round(float(row[1]))), int(round(float(row[2]))), int(round(float(row[3]))), int(round(float(row[4]))), int(round(float(row[5]))) ]})
        gt_num += 1
print(gt_num)
np.save('gt_dict.npy', gt_dict)

logits_images = {}

# Opening CSV results file
with open(args.results, 'w', newline='') as res_file:
    writer = csv.writer(res_file)
    # Processing all the images
    for image in tqdm(gt_dict.keys(), desc="Processing images...", unit="image"):
        # Reading the image
        img = Image.open(args.images + image).convert('RGB')
        if img.size == 0:
            print("Error")
        # Here you should add your code for applying your method
        results, logits = model.generate(img)
        logits_images[image] = logits
        p_u = results['upper_color']
        p_l = results['lower_color']
        p_g = results['gender']
        p_b = results['bag']
        p_h = results['hat']
        ########################################################
        # Writing a row in the CSV file
        writer.writerow([image, p_u, p_l, p_g, p_b, p_h])
        
# #--- Calculating the accuracy between data and results ---#
with open(args.results, mode='r') as res_file:
    res = csv.reader(res_file, delimiter=',')
    res_num = 0
    res_dict = {}
    for row in res:
        res_dict.update({row[0]: [int(round(float(row[1]))), int(round(float(row[2]))), int(round(float(row[3]))), int(round(float(row[4]))), int(round(float(row[5]))) ]})
        res_num += 1

np.save("res_dict.npy", res_dict)

with open("logits.csv", "w", newline='') as file:
    writer = csv.writer(file)
    for image in gt_dict.keys():
        gt_u, gt_l = gt_dict[image][0], gt_dict[image][1]
        pred_u, pred_l = res_dict[image][0], res_dict[image][1]

        if gt_u != pred_u or gt_l != pred_l:
            logit = logits_images[image]

            # Write to CSV for logging
            writer.writerow([image, pred_u, logit['upper_color'], pred_l, logit['lower_color']])
print(res_num)

#--- Calculating the accuracy ---#
# correct_upper = 0
# correct_lower = 0
# correct_hat = 0
# correct_bag = 0
# correct_gender = 0

# for image in res_dict.keys():
#     if image in gt_dict.keys():
#         if res_dict[image][0] == gt_dict[image][0]:
#             correct_upper += 1
#         if res_dict[image][1] == gt_dict[image][1]:
#             correct_lower += 1
#         if res_dict[image][2] == gt_dict[image][2]:
#             correct_gender += 1
#         if res_dict[image][3] == gt_dict[image][3]:
#             correct_bag += 1
#         if res_dict[image][4] == gt_dict[image][4]:
#             correct_hat += 1

# print(f"Upper color accuracy: {correct_upper / gt_num * 100:.2f}%")
# print(f"Lower color accuracy: {correct_lower / gt_num * 100:.2f}%")
# print(f"Gender accuracy: {correct_gender / gt_num * 100:.2f}%")
# print(f"Bag accuracy: {correct_bag / gt_num * 100:.2f}%")
# print(f"Hat accuracy: {correct_hat / gt_num * 100:.2f}%")
# print(f"Total accuracy: {(correct_upper + correct_lower + correct_gender + correct_bag + correct_hat) / (5 * gt_num) * 100:.2f}%")
#--- End of the script ---#