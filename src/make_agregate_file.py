import pandas as pd
import os

path = os.path.abspath('../../Desktop/ecoli/biolog_csv/_07-2890_Escherichia coli__2_37_PMX_438_2#19#2019_A_ 1B_1.csv')
new_data = pd.read_csv(path)

data_file = new_data["Data File"]
setup_time = new_data["Setup Time"]
position = new_data["Position"]
plate_type = new_data["Plate Type"]
strain_type = new_data["Strain Type"]
sample_number = new_data["Sample Number"]
strain_name = new_data["Strain Name"]
strain_number = new_data["Strain Number"]
other = new_data["Other"]

print(data_file)
