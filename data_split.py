from sklearn.model_selection import train_test_split
from glob import glob
import shutil
import os



if __name__ == "__main__":
    

    # 1. 이미지 파일 경로 (뒤에 확장자 포함)
    image_files = glob("./data/*.bmp")

    # 2. 이미지 파일명 가져오기
    images = [name.replace(".bmp", "") for name in image_files]

    # splitting the dataset
    # train:test = 3:1
    train_names, test_names = train_test_split(images, test_size=0.25, shuffle=True)

    def batch_move_files(file_list, source_path, destination_path):
        for file in file_list:
            image = file.split('/')[-1] + '.bmp'
            txt = file.split('/')[-1] + '.json'
            shutil.copy(os.path.join(source_path, image), destination_path)
            shutil.copy(os.path.join(source_path, txt), destination_path)
        return

    # 3. 이미지 파일 경로
    source_dir = "./data/"

    # 4. 분리된 데이터 셋들을 저장할 새로운 경로
    train_dir = "./data/train/"
    test_dir = "./data/test/"
    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(test_dir, exist_ok = True)
    batch_move_files(train_names, source_dir, train_dir)
    batch_move_files(test_names, source_dir, test_dir)

