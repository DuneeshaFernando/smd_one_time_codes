# Read all files in ServerMachineDataset/test and rename columns

import csv
from os import listdir
from os.path import isfile, join
import pandas as pd

test_dir_path = 'ServerMachineDataset/test/'
output_dir_path = 'smd/test/'
test_label_dir_path = 'ServerMachineDataset/test_label/'
label_output_dir_path = 'smd/test_label/'
final_test_dir_path = 'smd/test_with_labels/'

machine_file_list = [f for f in listdir(test_dir_path) if isfile(join(test_dir_path, f))]

# Comment section by section appropriately

# Section 1
# for machine_file in machine_file_list:
#     with open(test_dir_path + machine_file, 'r') as in_file:
#         stripped = (line.strip() for line in in_file)
#         lines = (line.split(",") for line in stripped if line)
#         output_file = output_dir_path + machine_file.strip('.txt') + ".csv"
#         column_names = []
#         for i in range(1, 39):
#             column_names.append("col" + str(i))
#         with open(output_file, 'w') as out_file:
#             writer = csv.writer(out_file)
#             writer.writerow(column_names)
#             writer.writerows(lines)

# Section 2
# for machine_file in machine_file_list:
#     with open(test_label_dir_path + machine_file, 'r') as in_file:
#         stripped = (line.strip() for line in in_file)
#         lines = (line.split(",") for line in stripped if line)
#         output_file = label_output_dir_path + machine_file.strip('.txt') + ".csv"
#         column_names = ['Normal/Attack']
#         with open(output_file, 'w') as out_file:
#             writer = csv.writer(out_file)
#             writer.writerow(column_names)
#             writer.writerows(lines)

# Section 3
machine_file_list = [f for f in listdir(output_dir_path) if isfile(join(output_dir_path, f))]
for machine_file in machine_file_list:
    df = pd.read_csv(output_dir_path + machine_file)
    label_df = pd.read_csv(label_output_dir_path + machine_file)
    new_df = pd.concat([df, label_df], axis=1)
    new_df.to_csv(final_test_dir_path + machine_file, index=False)
