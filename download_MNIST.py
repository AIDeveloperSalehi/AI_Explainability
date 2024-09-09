import requests
import gzip
import os

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    for file in files:
        print(f"Downloading {file}...")
        response = requests.get(base_url + file, verify=False)
        if response.status_code == 200:
            with open(file, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {file}")
        else:
            print(f"Failed to download {file}")

if __name__ == "__main__":
    download_mnist()

