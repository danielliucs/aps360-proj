import os

def renamefiles(directory):

    # if not os.path.exists(dest):
    #     os.makedirs(dest)
    word_to_first_dot = {}

    for file in os.listdir(directory):
        idx = file.find('.')
        word_to_first_dot[file] = idx

    for file in os.listdir(directory):
        idx = word_to_first_dot[file]
        new_name = file[:idx] + file[idx+1:]
        print(file, new_name)
        old_file = f'{directory}/{file}'
        new_file = f'{directory}/{new_name}'
        os.rename(old_file, new_file)
        

if __name__ == '__main__':
    dir = "C:/Users/User/Desktop/aps360-proj/testing_dataset/testing_yawning"
    renamefiles(dir)