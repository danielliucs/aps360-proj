import os

def removeDot(directory):

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
        if not os.path.exists(new_file): 
            os.rename(old_file, new_file)
    

def assignYawningOrNormal(directory):
    word_to_first_dot = {}
    yawned = {}
    for file in os.listdir(directory):
        idx = file.find('noyawn')
        if idx == -1:
            new_idx = file.find('yawn')
            if new_idx != -1:
                word_to_first_dot[file] = (new_idx, new_idx+len('yawn'))
                yawned[file] = True
            else:
                new_idx = file.find('.jpg')
                #print(f'file is this {file} with new_idx being this {new_idx} in MOV')
                word_to_first_dot[file] = (new_idx, new_idx)
                yawned[file] = False
        else:
            #print(idx)
            word_to_first_dot[file] = (idx, idx+len('noyawn'))
            yawned[file] = False

    for file in os.listdir(directory):
        idx = word_to_first_dot[file][0]
        length = word_to_first_dot[file][1]
        new_name = ""
        if yawned[file]:
            new_name = file[:idx] + "Yawning" + file[length:]
        else:
            new_name = file[:idx] + "Normal" + file[length:]
        print(file, new_name)
        old_file = f'{directory}/{file}'
        new_file = f'{directory}/{new_name}'
        if not os.path.exists(new_file): 
            os.rename(old_file, new_file)


if __name__ == '__main__':
    dir = "C:/Users/User/Desktop/aps360-proj/dataset/allPhotos"
    removeDot(dir)
    #assignYawningOrNormal(dir)