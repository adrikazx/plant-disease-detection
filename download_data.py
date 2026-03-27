import os
import shutil
import argparse
import sys
import subprocess

def install_kagglehub():
    """Auto-install kagglehub if it's not present."""
    try:
        import kagglehub
    except ImportError:
        print("Required package 'kagglehub' not found. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])

def download_and_extract(dataset_name="emmarex/plantdisease", download_path="./data"):
    install_kagglehub()
    import kagglehub
    
    print(f"Downloading {dataset_name} using KaggleHub...")
    print("No API Key is required for this public dataset!")
    
    # --- AUTO-FIX CORRUPTED CACHE ---
    cache_path = os.path.expanduser(f"~/.cache/kagglehub/datasets/{dataset_name}")
    if os.path.exists(cache_path):
        print(f"Clearing old cache at {cache_path} to prevent 'Bad magic number' resume errors...")
        shutil.rmtree(cache_path, ignore_errors=True)
    # --------------------------------
    
    try:
        # kagglehub securely downloads and caches public datasets automatically without auth
        dataset_cache_path = kagglehub.dataset_download(dataset_name)
        print(f"Downloaded securely to cache: {dataset_cache_path}")
        
        # Ensure our target data directory exists
        os.makedirs(download_path, exist_ok=True)
        
        print(f"Copying files to your {download_path} directory...")
        
        # Copy all items from the cache to the target directory
        for item in os.listdir(dataset_cache_path):
            s = os.path.join(dataset_cache_path, item)
            d = os.path.join(download_path, item)
            
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
                
        print("\n✅ Dataset successfully downloaded and prepared in the 'data' folder.")
        print("You are now ready to run `python train.py`!")
            
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the plant disease dataset")
    parser.add_argument("--dataset", type=str, default="emmarex/plantdisease", help="Kaggle Dataset string")
    parser.add_argument("--path", type=str, default="./data", help="Download path")
    args = parser.parse_args()
    
    # Resolve absolute path for the data directory to avoid confusion
    abs_path = os.path.abspath(args.path)
    download_and_extract(args.dataset, abs_path)
