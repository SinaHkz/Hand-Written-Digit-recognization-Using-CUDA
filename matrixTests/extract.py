import numpy as np # type: ignore
import struct
import csv

def read_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (first 8 bytes)
        magic_number, num_labels = struct.unpack('>II', f.read(8))
        
        # Ensure the magic number is correct (0x00000801 for MNIST labels)
        if magic_number != 2049:
            raise ValueError("Invalid magic number in the label file.")
        
        # Read the label data (remaining bytes)
        labels = np.fromfile(f, dtype=np.uint8)
        
    return labels

def save_labels_to_csv(labels, csv_file_path):
    # Write the labels to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header (optional)
        writer.writerow(["Label"])
        
        # Write each label as a new row
        for label in labels:
            writer.writerow([label])

# Path to the MNIST training labels file (adjust the path if necessary)
label_file_path = 't10k-labels.idx1-ubyte'

# Read the labels
labels = read_labels(label_file_path)

# Path to save the CSV file
csv_file_path = 'labels_test.csv'

# Save the labels to CSV
save_labels_to_csv(labels, csv_file_path)

print(f"Labels have been saved to {csv_file_path}")
