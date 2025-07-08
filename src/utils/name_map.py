file = 'name_map.csv'
# with open(file) as f:
#     data = f.readlines()
# print(data)

import pandas as pd

df = pd.read_csv(file)
print(df.head())
# js = df.to_json('temp.json', orient='records', lines=True)

data = {}
for name, roll, email in zip(df["Name"], df["roll_number"], df["email"]):
    d = {
            'name':name.lower().replace(" ","_"), 
            'roll': roll.lower(),
            'email': email.lower(),    
        }
    data[roll.lower()] = d

import json
file_op = "students_details.json"

with open(file_op, 'w') as f:
    json.dump(data, f, indent=4)

print(file_op, len(data))