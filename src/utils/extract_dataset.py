"""
Extract INCLUDE Dataset ZIP files
"""

from pathlib import Path
import zipfile

from src.config.settings import DOWNLOAD_DIR, RAW_DIR
def get_zip_files():
    """
    Returns all downloaded ZIP files.
    """

    return sorted(DOWNLOAD_DIR.glob("*.zip"))


def create_output_folder():
    """
    Creates the extraction folder.
    """

    output_folder = RAW_DIR / "INCLUDE"
    output_folder.mkdir(parents=True, exist_ok=True)

    marker_folder = output_folder / ".extracted"
    marker_folder.mkdir(exist_ok=True)

    return output_folder


def extract_zip(zip_file, output_folder):
    """
    Extracts a single ZIP file.
    """

    marker_file = (
    output_folder
    / ".extracted"
    / f"{zip_file.stem}.done"
    )

    if marker_file.exists():
        print(f"✓ {zip_file.name} already extracted. Skipping.")
        return

    print(f"Extracting {zip_file.name}...")

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_folder)

    marker_file.touch()

    print("Done.")


def main():

    zip_files = get_zip_files()

    print(f"Found {len(zip_files)} ZIP files.")

    output_folder = create_output_folder()

    for zip_file in zip_files:
        extract_zip(zip_file, output_folder)

    print("\nExtraction Completed.")


if __name__ == "__main__":
    main()