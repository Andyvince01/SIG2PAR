# >>> __init__.py
# Original author: Andrea Vincenzo Ricciardi
from src.data.par_dataset import PARDataset

__all__ = ["PARDataset"]
        
# def read_annotation(file):
#     # Reading Training Annotations
#     with open(file, "r") as file:
#             annotations = file.readlines()
#     return annotations

# def dataset_bar_chart(file_path):
#     upper_values = {key: 0 for key in color_to_int_map.keys()}
#     lower_values = {key: 0 for key in color_to_int_map.keys()}
#     gender_values = {key: 0 for key in gender_to_int_map.keys()}
#     bag_values = {key: 0 for key in bag_to_int_map.keys()}
#     hat_values = {key: 0 for key in hat_to_int_map.keys()}

#     annotations = read_annotation(file_path)

#     for annotation in annotations:
#         parts = annotation.strip().split(',')

#         upper_values[str(int_to_color_map[int(parts[1])])] += 1
#         lower_values[str(int_to_color_map[int(parts[2])])] += 1
#         gender_values[str(int_to_gender_map[int(parts[3])])] += 1
#         bag_values[str(int_to_bag_map[int(parts[4])])] += 1
#         hat_values[str(int_to_hat_map[int(parts[5])])] += 1

#     # Creazione e visualizzazione di tutti e cinque i grafici
#     create_bar_chart('Upper Body Clothing Colors', upper_values, 'navy')
#     create_bar_chart('Lower Body Clothing Colors', lower_values, 'green')
#     create_bar_chart('Gender Distribution', gender_values, 'purple')
#     create_bar_chart('Bag Presence', bag_values, 'red')
#     create_bar_chart('Hat Presence', hat_values, 'blue')

# def create_bar_chart(title, data, color):
#     plt.figure(figsize=(10, 5))
#     bars = plt.bar(data.keys(), data.values(), color=color)
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, yval, yval, ha='center', va='bottom')
#     plt.xlabel('Categories')
#     plt.ylabel('Frequency')
#     plt.title(title)
#     plt.show()
    
# import matplotlib.pyplot as plt

# dataset_bar_chart("data/mivia_par/training_set.txt")