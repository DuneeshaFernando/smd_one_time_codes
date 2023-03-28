import pandas as pd

train_dir_path = 'smd/train/'

# Rather than picking all machine files, only pick the files w/o concept drift
machine_file_list = ['machine-1-1.csv', 'machine-1-2.csv', 'machine-1-5.csv', 'machine-1-6.csv', 'machine-1-7.csv', 'machine-2-5.csv', 'machine-2-7.csv', 'machine-2-8.csv', 'machine-3-1.csv', 'machine-3-4.csv', 'machine-3-6.csv', 'machine-3-9.csv', 'machine-3-10.csv', 'machine-3-11.csv']

# This 1st section is to count all zero occurrences

for machine_file in machine_file_list:
    df = pd.read_csv(train_dir_path+machine_file)
    describe_df = df.describe()
    describe_df.to_csv('EDA_results/df_describe/' + machine_file)

# Counting no.of all zero columns in df
all_zero_count = {'Column 1':0,'Column 2':0,'Column 3':0,'Column 4':0,'Column 5':0,'Column 6':0,'Column 7':0,'Column 8':0,'Column 9':0,'Column 10':0,'Column 11':0,'Column 12':0,'Column 13':0,'Column 14':0,'Column 15':0,'Column 16':0,'Column 17':0,'Column 18':0,'Column 19':0,'Column 20':0,'Column 21':0,'Column 22':0,'Column 23':0,'Column 24':0,'Column 25':0,'Column 26':0,'Column 27':0,'Column 28':0,'Column 29':0,'Column 30':0,'Column 31':0,'Column 32':0,'Column 33':0,'Column 34':0,'Column 35':0,'Column 36':0,'Column 37':0,'Column 38':0}
for i in range(38):
    for machine_file in machine_file_list:
        df = pd.read_csv('EDA_results/df_describe/'+machine_file)
        min_val = df.iloc[3]['col'+str(i+1)]
        max_val = df.iloc[7]['col'+str(i+1)]
        if (max_val-min_val)==0:
            all_zero_count['Column '+str(i+1)]+=1
print(all_zero_count)

# This 2nd section is to visualize box plots

final_df = {'1-1':None,'1-2':None,'1-5':None,'1-6':None,'1-7':None,'2-5':None,'2-7':None,'2-8':None,'3-1':None,'3-4':None,'3-6':None,'3-9':None,'3-10':None,'3-11':None}
# Rather than drawing individual boxplots, form a dataframe per column
for i in range(38):
    print("column"+str(i+1))
    for machine_file in machine_file_list:
        df = pd.read_csv(train_dir_path + machine_file)
        final_df[machine_file.strip('.csv').strip('machine-')] = df['col'+str(i+1)]
    result = pd.DataFrame(final_df)
    result.to_csv('EDA_results/14mach_col_wise_dfs/' + 'column' + str(i+1) + '.csv', index=False)

# This 3rd section is to obtain collinearity plots

df_list = []
for machine_file in machine_file_list:
    df = pd.read_csv('EDA_results/df_describe/' + machine_file)
    df_list.append(df.iloc[[1], 1:])
final_df = pd.concat(df_list, ignore_index=True)
final_df.insert(0,"Machine",[f.strip('.csv') for f in machine_file_list])
final_df.to_csv('EDA_results/mean_analysis.csv', index=False)

df = pd.read_csv('EDA_results/mean_analysis.csv')
selected_column_list = ['Machine','col6','col7','col23','col24','col26']
sub_df = df[selected_column_list]
sub_df.to_csv('EDA_results/5_selected_columns_for_14_machines.csv', index=False)
