import os
import re
import shutil

RAW_DATA = 'rawdata'
CLEAR_DATA = 'data'


def is_target(name):
    pattern = "^[0-9]{11}-[0-9]{2}-[0-9]{2}\.(wav|dat)$"
    if re.match(pattern, name) is not None:
        return True
    return False


def make_stu_dirs(stu_id):
    stu_root = os.path.join(CLEAR_DATA, stu_id)
    if not os.path.exists(stu_root):
        os.makedirs(stu_root)

    for word in range(0, 20):
        word_dir = os.path.join(stu_root, str(word))
        if not os.path.exists(word_dir):
            os.makedirs(word_dir)


def move_data():
    students = os.listdir(RAW_DATA)
    # print('STUDENTS NUMBER: {}'.format(len(students)))
    for stu in students:
        stu_dir = os.path.join(RAW_DATA, stu)
        if not os.path.isdir(stu_dir):
            continue

        make_stu_dirs(stu)

        clear_stu = os.path.join(CLEAR_DATA, stu)
        files = os.listdir(stu_dir)
        for file in files:
            # file_path = os.path.join(stu_dir, file)
            # if not os.path.isfile(file_path):
            #     continue

            if not is_target(file):
                continue

            word_id = os.path.splitext(file)[0].split('-')[1]
            word_id = int(word_id)
            file_path = os.path.join(stu_dir, file)
            dst_path = os.path.join(clear_stu, str(word_id), file)
            # copy file
            shutil.copyfile(file_path, dst_path)
            # print('Move {} to {}...'.format(file_path, dst_path))


def check_clear_data():
    err_list = []
    for stu in os.listdir(CLEAR_DATA):
        stu_path = os.path.join(CLEAR_DATA, stu)
        if not os.path.isdir(stu_path):
            continue

        words = os.listdir(stu_path)
        is_wrong = False
        for word in words:
            word_path = os.path.join(stu_path, word)
            if not os.path.isdir(word_path):
                continue

            files = os.listdir(word_path)
            num = len(files)
            if num != 20:
                is_wrong = True
                print('{} only have {} files for word {}'.format(stu, num, word))
        if is_wrong:
            err_list.append(stu)
    print()
    print('ERROR: ')
    print(err_list)


if __name__ == "__main__":
    move_data()
    print()
    check_clear_data()
