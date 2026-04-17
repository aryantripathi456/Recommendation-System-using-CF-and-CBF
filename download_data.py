import os
import shutil
import kagglehub
import requests
import zipfile
import io

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def download_tmdb():
    print("Downloading TMDB 5k dataset...")
    try:
        # Download latest version
        path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
        print(f"Downloaded TMDB dataset to: {path}")
        
        # Copy to local data directory
        for file in os.listdir(path):
            if file.endswith('.csv'):
                shutil.copy(os.path.join(path, file), os.path.join(DATA_DIR, file))
                print(f"Copied {file} to {DATA_DIR}/")
    except Exception as e:
        print(f"Failed to download TMDB: {e}")
        print("You might need to authenticate with Kaggle if it fails. For public datasets kagglehub usually works directly if it doesn't hit a rate limit.")

def download_movielens():
    print("Downloading MovieLens Small dataset...")
    try:
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(DATA_DIR)
        print(f"Extracted MovieLens data to {DATA_DIR}/ml-latest-small/")
    except Exception as e:
        print(f"Failed to download MovieLens: {e}")

if __name__ == "__main__":
    download_tmdb()
    download_movielens()
    print("Data download process finished.")
