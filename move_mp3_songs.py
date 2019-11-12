import os

ROOT_DIR = "/mnt/DATA/SKK"
DEST_DIR = "/mnt/DATA/SEAS/Semester 3/SS/presto-chango/MP3 Songs"

for root, dirs, files in os.walk(ROOT_DIR):
    # print("Root", root)
    # print("Dirs", dirs)
    # print(files)
    for file in files:
        print(file)
        with open(os.path.join(root, file), "rb") as f:
            file_content = f.read()
        with open(os.path.join(DEST_DIR, file), "wb") as f:
            f.write(file_content)
