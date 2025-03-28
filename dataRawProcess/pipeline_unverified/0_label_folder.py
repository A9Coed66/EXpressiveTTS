import os
import re
import argparse

parser = argparse.ArgumentParser(description="Label audio in folder")
parser.add_argument("--data_path", type=str, default='/home4/tuanlha/DataTest', help="Path to the data directory")
args = parser.parse_args()


def rename_files_in_directory(directory):
    # Lấy danh sách các file trong thư mục
    files = os.listdir(directory)

    # Lọc ra chỉ các file (bỏ qua thư mục con)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]

    # Sắp xếp các file theo thứ tự alphabet
    files.sort()

    pattern = re.compile(r'\[.*?\]')

    # Đánh số và đổi tên các file
    for index, filename in enumerate(files, start=1):
        # Tạo tên mới với số đánh số ở đầu
        new_name = f"{index} {filename}"

        # Xóa phần [*] của tên file
        new_name = re.sub(pattern, '', new_name).strip()

        # Đường dẫn đầy đủ của file cũ và file mới
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)

        # Đổi tên file
        os.rename(old_file, new_file)
        print(f"Đã đổi tên {filename} thành {new_name}")

# Thay đổi đường dẫn thư mục của bạn tại đây
directory_path = args.data_path
rename_files_in_directory(directory_path)