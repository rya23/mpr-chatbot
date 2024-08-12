import json
import ast

# Load the JSON data
with open("newest_data.json", "r") as file:
    data = json.load(file)

# Convert the images field to proper JSON format
for key, value in data.items():
    # Use ast.literal_eval to safely evaluate the string as a Python list
    images_list = ast.literal_eval(value["images"])
    # Replace the images field with the proper JSON array
    value["images"] = images_list

# Save the updated data back to a JSON file
with open("data_corrected.json", "w") as file:
    json.dump(data, file, indent=4)

print("Conversion complete. Corrected data saved to 'data_corrected.json'.")
