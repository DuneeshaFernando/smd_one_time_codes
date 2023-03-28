# Read all files in ServerMachineDataset/train and rename columns

import csv
from os import listdir
from os.path import isfile, join

train_dir_path = 'ServerMachineDataset/train/'
output_dir_path = 'smd/train/'

machine_file_list = [f for f in listdir(train_dir_path) if isfile(join(train_dir_path, f))]

for machine_file in machine_file_list:
    with open(train_dir_path + machine_file, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        output_file = output_dir_path + machine_file.strip('.txt') + ".csv"
        column_names = []
        for i in range(1, 39):
            column_names.append("col" + str(i))
        with open(output_file, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(column_names)
            writer.writerows(lines)
