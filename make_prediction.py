import os
from PIL import Image
from app import predict_step

# Assuming your dataset is in a directory, replace 'your_dataset_directory' with the actual directory path
dataset_directory = '../../archive/Images'

# List all image files in the dataset directory
image_files = [f for f in os.listdir(dataset_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create a dictionary to store predictions
all_predictions = {}

# Make predictions for each image in the dataset
count = 0

for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(dataset_directory, image_file)

    # Load the image
    image = Image.open(image_path)

    # Make predictions for the current image
    predictions = predict_step([image])
    print(count)
    print(predictions) 

    count += 1

    # Store the predictions in the dictionary
    all_predictions[image_file] = predictions[0]

# Save predictions to a text file
output_file_path = 'predictions.txt'    

with open(output_file_path, 'w') as file:
    for image_file, prediction in all_predictions.items():
        file.write(f"{image_file}: {prediction}\n")

print(f"Predictions saved to {output_file_path}")
