import os
import shutil

def organize_images_into_folders(root_dir):
    for filename in os.listdir(root_dir):
        if filename.startswith('.'):
            continue  # skip hidden files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # skip non-image files

        class_name = filename.split('-')[0]  # e.g., bladder from bladder-0001.png
        class_dir = os.path.join(root_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        src = os.path.join(root_dir, filename)
        dst = os.path.join(class_dir, filename)

        shutil.move(src, dst)
        print(f"Moved {filename} -> {class_dir}/")

if __name__ == "__main__":
    organize_images_into_folders('./datasets/img/train')
    organize_images_into_folders('./datasets/img/test')