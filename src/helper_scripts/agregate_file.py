import pandas as pd
import os
import numpy as np


def get_metadata(metadata, strain_number):
    for index, row in metadata.iterrows():
        if strain_number in row["Strain"]:
            return row["Otype"], row["Htype"], row["Host"]

def strip_zeros(plate):
    if plate[2] == '0':
        return 'PM{}'.format(plate[3])

def strip_comas(x, number):
    with open("data/trash/complete_{0}_{1}.csv".format(x, number), 'r') as r, open("data/omnilog_data/complete_{0}_{1}.csv".format(x, number), 'w') as w:
        for num, line in enumerate(r):
            if num < 12:
                newline = line[:-96] + "\n" if "\n" in line else line[:-1]
            else:
                newline = line
            w.write(newline)

for x, filename in enumerate(os.listdir('data/biolog_csv')):
    path = os.path.abspath('data/biolog_csv/{0}'.format(filename))
    
    metadata_path = os.path.abspath('data/final_onmilog_metadata.csv')
    metadata = pd.read_csv(metadata_path)

    new_data = pd.read_csv(path)

    data_file = new_data["Data File"]
    file_name = data_file[1]

    setup_time = new_data["Setup Time"]
    time = setup_time[1]

    position = new_data["Position"]
    pos = position[1]

    plate_type = new_data["Plate Type"]
    plate = plate_type[1]
    plate = strip_zeros(plate)

    strain_type = new_data["Strain Type"]
    strain = strain_type[1]

    sample_number = new_data["Sample Number"]
    sample = sample_number[1]

    strain_name = new_data["Strain Name"]
    name = strain_name[1]

    strain_number = new_data["Strain Number"]
    number = strain_number[1]

    other_info = new_data["Other"]
    other = other_info[1]

    # get otype, htype and host from the metadata
    o_type, h_type, host = get_metadata(metadata, number)

    df = pd.DataFrame(index = list(range(0,13)), columns = ['Unnamed 0', 'Unnamed 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44', 'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 47', 'Unnamed: 48', 'Unnamed: 49', 'Unnamed: 50', 'Unnamed: 51', 'Unnamed: 52', 'Unnamed: 53', 'Unnamed: 54', 'Unnamed: 55', 'Unnamed: 56', 'Unnamed: 57', 'Unnamed: 58', 'Unnamed: 59', 'Unnamed: 60', 'Unnamed: 61', 'Unnamed: 62', 'Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65', 'Unnamed: 66', 'Unnamed: 67', 'Unnamed: 68', 'Unnamed: 69', 'Unnamed: 70', 'Unnamed: 71', 'Unnamed: 72', 'Unnamed: 73', 'Unnamed: 74', 'Unnamed: 75', 'Unnamed: 76', 'Unnamed: 77', 'Unnamed: 78', 'Unnamed: 79', 'Unnamed: 80', 'Unnamed: 81', 'Unnamed: 82', 'Unnamed: 83', 'Unnamed: 84', 'Unnamed: 85', 'Unnamed: 86', 'Unnamed: 87', 'Unnamed: 88', 'Unnamed: 89', 'Unnamed: 90', 'Unnamed: 91', 'Unnamed: 92', 'Unnamed: 93', 'Unnamed: 94', 'Unnamed: 95', 'Unnamed: 96'])

    df.at[0, 'Unnamed 0'] = 'Data File'
    df.at[0, 'Unnamed 1'] = file_name

    # set time
    df.at[1, 'Unnamed 0'] = 'Set up Time'
    df.at[1, 'Unnamed 1'] = time

    # set Position
    df.at[2, 'Unnamed 0'] = 'Position'
    df.at[2, 'Unnamed 1'] = pos

    # set plate type
    df.at[3, 'Unnamed 0'] = 'Plate Type'
    df.at[3, 'Unnamed 1'] = plate

    # set strain type
    df.at[4, 'Unnamed 0'] = 'Strain Type'
    df.at[4, 'Unnamed 1'] = strain

    # set sample number
    df.at[5, 'Unnamed 0'] = 'Sample Number'
    df.at[5, 'Unnamed 1'] = sample

    # set strain name
    df.at[6, 'Unnamed 0'] = 'Strain Name'
    df.at[6, 'Unnamed 1'] = number

    df.at[7, 'Unnamed 0'] = 'O-type'
    df.at[7, 'Unnamed 1'] = o_type

    df.at[8, 'Unnamed 0'] = 'H-type'
    df.at[8, 'Unnamed 1'] = h_type

    df.at[9, 'Unnamed 0'] = 'Host'
    df.at[9, 'Unnamed 1'] = host

    # set strain number
    df.at[10, 'Unnamed 0'] = 'Strain Number'
    df.at[10, 'Unnamed 1'] = number

    # set other
    df.at[11, 'Unnamed 0'] = 'Other'
    df.at[11, 'Unnamed 1'] = other

    hour = new_data.loc[:3000, 'Hour':]
    df0 = pd.DataFrame(hour)

    header_list = df0.columns.values
    values_list = df0.values.tolist()
    s = pd.Series(header_list, index=df.columns)

    df2 = pd.DataFrame(columns = ['Unnamed 0', 'Unnamed 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44', 'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 47', 'Unnamed: 48', 'Unnamed: 49', 'Unnamed: 50', 'Unnamed: 51', 'Unnamed: 52', 'Unnamed: 53', 'Unnamed: 54', 'Unnamed: 55', 'Unnamed: 56', 'Unnamed: 57', 'Unnamed: 58', 'Unnamed: 59', 'Unnamed: 60', 'Unnamed: 61', 'Unnamed: 62', 'Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65', 'Unnamed: 66', 'Unnamed: 67', 'Unnamed: 68', 'Unnamed: 69', 'Unnamed: 70', 'Unnamed: 71', 'Unnamed: 72', 'Unnamed: 73', 'Unnamed: 74', 'Unnamed: 75', 'Unnamed: 76', 'Unnamed: 77', 'Unnamed: 78', 'Unnamed: 79', 'Unnamed: 80', 'Unnamed: 81', 'Unnamed: 82', 'Unnamed: 83', 'Unnamed: 84', 'Unnamed: 85', 'Unnamed: 86', 'Unnamed: 87', 'Unnamed: 88', 'Unnamed: 89', 'Unnamed: 90', 'Unnamed: 91', 'Unnamed: 92', 'Unnamed: 93', 'Unnamed: 94', 'Unnamed: 95', 'Unnamed: 96'])

    s = pd.Series(header_list, index=df2.columns)
    df2 = df2.append(s, ignore_index=True)

    df3 = pd.DataFrame(data = values_list, columns = ['Unnamed 0', 'Unnamed 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44', 'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 47', 'Unnamed: 48', 'Unnamed: 49', 'Unnamed: 50', 'Unnamed: 51', 'Unnamed: 52', 'Unnamed: 53', 'Unnamed: 54', 'Unnamed: 55', 'Unnamed: 56', 'Unnamed: 57', 'Unnamed: 58', 'Unnamed: 59', 'Unnamed: 60', 'Unnamed: 61', 'Unnamed: 62', 'Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65', 'Unnamed: 66', 'Unnamed: 67', 'Unnamed: 68', 'Unnamed: 69', 'Unnamed: 70', 'Unnamed: 71', 'Unnamed: 72', 'Unnamed: 73', 'Unnamed: 74', 'Unnamed: 75', 'Unnamed: 76', 'Unnamed: 77', 'Unnamed: 78', 'Unnamed: 79', 'Unnamed: 80', 'Unnamed: 81', 'Unnamed: 82', 'Unnamed: 83', 'Unnamed: 84', 'Unnamed: 85', 'Unnamed: 86', 'Unnamed: 87', 'Unnamed: 88', 'Unnamed: 89', 'Unnamed: 90', 'Unnamed: 91', 'Unnamed: 92', 'Unnamed: 93', 'Unnamed: 94', 'Unnamed: 95', 'Unnamed: 96'])


    df4 = df.append(df2)
    df5 = df4.append(df3)

    df5.to_csv("data/trash/complete_{0}_{1}.csv".format(x, number), index = False, header = False)
    print("Done file {0}".format(file_name))

    strip_comas(x, number)
