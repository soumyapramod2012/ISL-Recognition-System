from src.preprocessing.dataset_processor import DatasetProcessor

processor = DatasetProcessor(
    "dataset/raw/INCLUDE"
)

videos = processor.get_video_files()

print("Total videos :", len(videos))

print()

for video in videos[:10]:

    print(video["label"])

    print(video["path"])

    print()