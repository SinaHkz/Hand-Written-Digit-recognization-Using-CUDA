import os
import urllib.request
import gzip
import shutil

# MNIST dataset URLs (Updated mirror)
mnist_files = {
    "train-images.idx3-ubyte": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels.idx1-ubyte": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images.idx3-ubyte": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels.idx1-ubyte": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
}

# Function to download and extract MNIST files
def download_mnist():
    for filename, url in mnist_files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, filename + ".gz")

            # Extract the .gz file
            with gzip.open(filename + ".gz", "rb") as in_f, open(filename, "wb") as out_f:
                shutil.copyfileobj(in_f, out_f)

            os.remove(filename + ".gz")  # Delete compressed file after extraction
            print(f"Saved {filename}")

# Download MNIST dataset
download_mnist()
print("MNIST dataset downloaded and saved in the current directory.")
