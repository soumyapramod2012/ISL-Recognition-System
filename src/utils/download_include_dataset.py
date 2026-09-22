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
    Prints dataset summary and category-wise ZIP sizes.
    """

    zip_files = [
        file for file in files
        if file["key"].endswith(".zip")
    ]

    total_size = sum(file["size"] for file in zip_files)
    total_size_gb = total_size / (1024 ** 3)

    print("\n" + "=" * 55)
    print("OFFICIAL INCLUDE DATASET")
    print("=" * 55)

    print(f"Total ZIP Files : {len(zip_files)}")
    print(f"Categories      : {len(categories)}")
    print(f"Dataset Size    : {total_size_gb:.2f} GB")

    print("\nCategory-wise size:")
    print("-" * 55)

    category_sizes = {}

    for file in zip_files:

        filename = file["key"]

        category = filename.rsplit("_", 1)[0]

        category_sizes[category] = (
            category_sizes.get(category, 0)
            + file["size"]
        )

    for category in categories:

        size_gb = category_sizes.get(category, 0) / (1024 ** 3)

        print(f"{category:<30} {size_gb:>8.2f} GB")

    print("-" * 55)
    print(f"{'TOTAL':<30} {total_size_gb:>8.2f} GB")


def download_file(file):
    """
    Downloads a file from Zenodo with resume and retry support.
    """

    filename = file["key"]
    url = file["links"]["self"]
    save_path = DOWNLOAD_DIR / filename

    expected_size = file["size"]

    # Create download directory if needed
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Check existing file
    existing_size = save_path.stat().st_size if save_path.exists() else 0

    if existing_size == expected_size:
        print(f"✓ {filename} already exists and is complete. Skipping.")
        return True

    if existing_size > expected_size:
        print(f"⚠ {filename} is larger than expected.")
        print("  Removing corrupted file and restarting.")
        save_path.unlink()
        existing_size = 0

    if existing_size > 0:
        print(
            f"\nResuming: {filename}"
            f"\nAlready downloaded: "
            f"{existing_size / (1024 ** 2):.2f} MB"
        )
    else:
        print(f"\nDownloading: {filename}")

    max_retries = 5
    downloaded = existing_size

    for attempt in range(1, max_retries + 1):

        try:

            headers = {}

            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"

            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=60
            )

            # If server ignores Range, restart safely
            if downloaded > 0 and response.status_code == 200:
                print(
                    "\nServer did not honor resume request."
                    "\nRestarting download from beginning."
                )

                downloaded = 0
                response.close()

                response = requests.get(
                    url,
                    stream=True,
                    timeout=60
                )

            response.raise_for_status()

            # Determine progress total
            if response.status_code == 206:
                total_size = expected_size
                mode = "ab"
            else:
                total_size = expected_size
                mode = "wb"

            with open(save_path, mode) as f:

                with tqdm(
                    total=total_size,
                    initial=downloaded,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=filename
                ) as progress:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):

                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))
                            downloaded += len(chunk)

            response.close()

            # Verify final size
            actual_size = save_path.stat().st_size

            if actual_size == expected_size:
                print(
                    f"\n✓ Download completed and verified: "
                    f"{filename}"
                )
                return True

            print(
                f"\n⚠ Incomplete download."
                f"\nExpected : {expected_size:,} bytes"
                f"\nActual   : {actual_size:,} bytes"
            )

            downloaded = actual_size

        except (
            requests.exceptions.RequestException,
            requests.exceptions.ChunkedEncodingError,
            ConnectionError
        ) as e:

            print(
                f"\n⚠ Connection interrupted "
                f"(attempt {attempt}/{max_retries})"
            )
            print(f"  {e}")

            if save_path.exists():
                downloaded = save_path.stat().st_size
                print(
                    f"  Saved so far: "
                    f"{downloaded / (1024 ** 2):.2f} MB"
                )

            if attempt < max_retries:
                print("  Retrying...")
            else:
                print("  Maximum retries reached.")

    print(f"\n✗ Failed to download: {filename}")
    return False


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
    """
    Main program.
    """

    dataset = get_dataset_information()

    # Stop safely if Zenodo API is unavailable
    if not dataset:
        print("\nUnable to retrieve INCLUDE dataset information.")
        print("Please check your internet connection or try again later.")
        return

    files = get_available_files(dataset)

    categories = get_categories(files)

    print(f"\nTotal Files : {len(files)}\n")

    for file in files:
        print(file["key"])

    print_dataset_summary(files, categories)

    print("\nAvailable Categories:\n")

    for i, category in enumerate(categories, start=1):
        print(f"{i}. {category}")

    choice = show_menu()

    if choice == "1":
        download_all(files)

    elif choice == "2":
        download_selected_categories(files, categories)

    elif choice == "3":
        print("Goodbye!")

    else:
        print("Invalid choice.")
  

if __name__ == "__main__":
    main()