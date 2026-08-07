"""
Official INCLUDE Dataset Downloader

Features:
- Connect to Zenodo
- Display dataset information
- Download complete dataset
- Download selected categories
- Skip existing downloads
- Show download progress
"""

import requests
from pathlib import Path
from tqdm import tqdm
from src.config.settings import DOWNLOAD_DIR,ZENODO_API


def get_dataset_information():
    """
    Connect to the Zenodo API and return dataset information.
    """

    print("Connecting to Zenodo...")

    try:
        response = requests.get(ZENODO_API, timeout=60)

        response.raise_for_status()

        print("Connected Successfully.\n")

        return response.json()

    except requests.exceptions.Timeout:
        print("Connection timed out. Please try again.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Network Error : {e}")
        return None


def get_available_files(dataset):
    """
    Returns all downloadable files sorted alphabetically.
    """

    files = dataset["files"]

    return sorted(files, key=lambda x: x["key"])


def get_categories(files):
    """
    Returns all dataset categories.
    """

    categories = set()

    for file in files:

        name = file["key"]

        if name.endswith(".zip"):
            category = name.rsplit("_", 1)[0]
            categories.add(category)

    return sorted(categories)


def print_dataset_summary(files, categories):
    """
    Prints dataset summary.
    """

    total_size = sum(file["size"] for file in files if file["key"].endswith(".zip"))

    total_size_gb = total_size / (1024 ** 3)

    print("\n" + "=" * 45)
    print("OFFICIAL INCLUDE DATASET")
    print("=" * 45)

    print(f"Total ZIP Files : {len([f for f in files if f['key'].endswith('.zip')])}")
    print(f"Categories      : {len(categories)}")
    print(f"Dataset Size    : {total_size_gb:.2f} GB")


def download_file(file):
    """
    Downloads a single file from Zenodo.
    """

    filename = file["key"]
    url = file["links"]["self"]

    save_path = DOWNLOAD_DIR / filename

    if save_path.exists():
        print(f"✓ {filename} already exists. Skipping.")
        return

    print(f"\nDownloading: {filename}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(save_path, "wb") as f:

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=filename
        ) as progress:

            for chunk in response.iter_content(chunk_size=1024 * 1024):

                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

    print("\nDownload Completed.")


def download_all(files):
    """
    Downloads all ZIP files.
    """

    zip_files = [file for file in files if file["key"].endswith(".zip")]

    print(f"\nDownloading {len(zip_files)} ZIP files...\n")

    for file in zip_files:
        download_file(file)

    print("\nAll downloads completed.")


def download_selected_categories(files, categories):
    """
    Downloads selected categories.
    """

    print("\nAvailable Categories:\n")

    for i, category in enumerate(categories, start=1):
        print(f"{i}. {category}")

    selection = input(
        "\nEnter category numbers (comma separated): "
    )

    selected_numbers = {
        int(x.strip()) for x in selection.split(",")
    }

    selected_categories = {
        categories[i - 1] for i in selected_numbers
    }

    print("\nSelected Categories:")

    for category in selected_categories:
        print(f"• {category}")

    print()

    for file in files:

        filename = file["key"]

        if not filename.endswith(".zip"):
            continue

        category = filename.rsplit("_", 1)[0]

        if category in selected_categories:
            download_file(file)


def show_menu():
    """
    Displays the download menu.
    """

    print("\n" + "=" * 40)
    print("INCLUDE DATASET DOWNLOADER")
    print("=" * 40)

    print("1. Download Complete Dataset")
    print("2. Download Selected Categories")
    print("3. Exit")

    return input("\nEnter your choice: ").strip()


def main():

    dataset = get_dataset_information()

    if dataset:

        files = get_available_files(dataset)

        categories = get_categories(files)

        print(f"\nTotal Files : {len(files)}\n")
        
        for file in files:
            print(file["key"])

        print_dataset_summary(files, categories)

        print("\nAvailable Categories:\n")

        for i, category in enumerate(categories, start=1):
            print(f"{i}. {category}")

    ''' Downloads all zip files'''

    choice = show_menu()

    if choice == "1":
        download_all(files)

    elif choice == "2":
        download_selected_categories(files, categories)

    else:
        print("Goodbye!")
  

if __name__ == "__main__":
    main()