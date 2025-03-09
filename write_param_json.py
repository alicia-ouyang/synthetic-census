import json

state_initials_input = input("Enter the state initals in uppercase for parameter file in a list separated by spaces: ")
data_path = input("Enter the path of your data folder: ")

states_list = state_initials_input.split()
for state_initials in states_list:
    param_dict = {
        "state": state_initials,
        "micro_file": f"{data_path}/{state_initials}/{state_initials.lower()}.2010.pums.01.txt",
        "person_micro_file": f"{data_path}/output/{state_initials}/person_micro.csv",
        "block_file": f"{data_path}/{state_initials}/block_data.csv",
        "block_clean_file": f"{data_path}/{state_initials}/block_data_cleaned.csv", 
        "synthetic_output_dir": f"{data_path}/output/{state_initials}/", 
        "include_probs": False,
        "num_sols":5000
    }
    with open(f"{state_initials}_param.json", 'w') as f:  
        json.dump(param_dict, f)

