#Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image

# class CustomDataset(Dataset):
#     def __init__(self) -> None:
#         super().__init__()

def get_relevant_indicies(dataset):
    """Returns the indicies of the classes in the dataset"""
    indicies = []
    for i in range(len(dataset)):
        idx = dataset[i][1]
        indicies.append(idx)
    return indicies

def convert_imgs_to_tensor(path, transform, video_to_frames, Yawning, testing):

    video_as_tensors = []
    if testing:
        for name, frames in video_to_frames.items():
            list_of_imgs_to_tensors = []
            for jpg in frames:
                #print(f'{path}/{jpg}')
                img = Image.open(f'{path}/{jpg}')
                tensor = transform(img)
                list_of_imgs_to_tensors.append(tensor)
            #For varying length videos   
            if len(list_of_imgs_to_tensors) <= 32 and len(list_of_imgs_to_tensors) > 20:
                list_of_imgs_to_tensors = list_of_imgs_to_tensors * 2
            if len(list_of_imgs_to_tensors) <= 20 and len(list_of_imgs_to_tensors) > 15:
                list_of_imgs_to_tensors = list_of_imgs_to_tensors * 3
            if len(list_of_imgs_to_tensors) <= 15 and len(list_of_imgs_to_tensors) > 10:
                list_of_imgs_to_tensors = list_of_imgs_to_tensors * 5
            if len(list_of_imgs_to_tensors) <= 10 and len(list_of_imgs_to_tensors) > 5:
                list_of_imgs_to_tensors = list_of_imgs_to_tensors * 7
            if len(list_of_imgs_to_tensors) <= 5:
                list_of_imgs_to_tensors = list_of_imgs_to_tensors * 15
        
            stacked_frames = torch.stack(list_of_imgs_to_tensors) #Stacking the frames on top of one another
            #I put this here to make sure we aren't infinitely looping
            print(name, len(list_of_imgs_to_tensors), stacked_frames.shape, len(frames))
            if Yawning == False:
                video_as_tensors.append((stacked_frames, 0)) #0 for no yawning
            else:
                video_as_tensors.append((stacked_frames, 1)) #1 for yawning
    else:
        for name, frames in video_to_frames.items():
            if Yawning == False:
                #If yawning in name we want to stop and return we have everything for normal
                if "Yawning" in name:
                    break
            else:
                #If normal in name and yawning is true we skip the first couple until we get to yawning
                if "Normal" in name:
                    continue
            list_of_imgs_to_tensors = []
            for jpg in frames:
                #print(f'{path}/{jpg}')
                img = Image.open(f'{path}/{jpg}')
                tensor = transform(img)
                list_of_imgs_to_tensors.append(tensor)
            #For varying length videos   
            if len(list_of_imgs_to_tensors) <= 32 and len(list_of_imgs_to_tensors) > 20:
                list_of_imgs_to_tensors = list_of_imgs_to_tensors * 2
            if len(list_of_imgs_to_tensors) <= 20 and len(list_of_imgs_to_tensors) > 15:
                list_of_imgs_to_tensors = list_of_imgs_to_tensors * 3
            if len(list_of_imgs_to_tensors) <= 15 and len(list_of_imgs_to_tensors) > 10:
                list_of_imgs_to_tensors = list_of_imgs_to_tensors * 5
            if len(list_of_imgs_to_tensors) <= 10 and len(list_of_imgs_to_tensors) > 5:
                list_of_imgs_to_tensors = list_of_imgs_to_tensors * 7
            if len(list_of_imgs_to_tensors) <= 5:
                list_of_imgs_to_tensors = list_of_imgs_to_tensors * 15

            stacked_frames = torch.stack(list_of_imgs_to_tensors) #Stacking the frames on top of one another

            #I put this here to make sure we aren't infinitely looping
            print(name, len(list_of_imgs_to_tensors), stacked_frames.shape, len(frames))
            if Yawning == False:
                video_as_tensors.append((stacked_frames, 0)) #0 for no yawning
            else:
                video_as_tensors.append((stacked_frames, 1)) #1 for yawning
    
    return video_as_tensors

def get_data(batch_size, folder, video_to_frames):#, video_to_frames):
    """Takes a batch_size and the name of the folder (name of folder most likely called dataset)
    Example:
    get_data(1, "~/aps360-proj/dataset")
    """
    transform = transforms.Compose(
        [transforms.Resize((224,224)), transforms.transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    #Load images and convert all to tensors
    normal = convert_imgs_to_tensor(f'{folder}/Normal', transform, video_to_frames, False, False)
    yawning = convert_imgs_to_tensor(f'{folder}/Yawning', transform, video_to_frames, True, False)

    #Concat both datasets
    trainset = torch.utils.data.ConcatDataset([normal, yawning])

    train_size = int(0.9*len(trainset)) #90% split for training
    val_size = len(trainset) - train_size #10% for validation
    train_split, val_split = torch.utils.data.random_split(trainset, [train_size, val_size])

    #To test if we split correctly
    # for i in val_split:
    #     print(i)

   # print("We have actually split it", train_split)
    #SAMPLER DOES NOT WORK FOR SOME REASON LOL so I split and then do this
    #https://stackoverflow.com/questions/74291946/wrong-labels-when-using-dataloader-pytorch
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, num_workers=1)
    #print("Our dataloader", train_loader)
    return train_loader, val_loader

def get_testing_data(batch_size, folder, yawning_frames, normal_frames):
    transform = transforms.Compose(
        [transforms.Resize((224,224)), transforms.transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    #Load images and convert all to tensors
    normal_testing = convert_imgs_to_tensor(f'{folder}/testing_normal', transform, normal_frames, False, True)
    print("NOW FOR YAWNING!")
    yawning_testing = convert_imgs_to_tensor(f'{folder}/testing_yawning', transform, yawning_frames, True, True)

    #Concat both datasets
    testset = torch.utils.data.ConcatDataset([normal_testing, yawning_testing])

    #To test if it's alright
    # for i in testset:
    #     print(i)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=1, shuffle=True)
    #print("Our dataloader", test_loader)
    return test_loader
    
def visualize(loader):
    """Visualize the data we parse, mostly for a sanity check"""
    classes = ("Normal", "Yawning")
    k = 0
    copy = loader
    for images, labels in copy:
        image = images[0].squeeze(0) #Only want the RGB stuff!

        img = np.transpose(image, [1,2,0])
        #print(labels[0])
        # normalize pixel intensity values to [0, 1]
        img = img / 2 + 0.5
        plt.subplot(3, 5, k+1, title=classes[labels])
        plt.axis('off')
        plt.imshow(img) #Creates the image
        k += 1
        if k > 14:
            break
    plt.show() #Actually shows the image
def save_features(loader, train, dir):
    """Save tensors of loaders into folders depending on which dataset it is"""
    classes=('Normal', 'Yawning')

    if train == "train":
        dir = f'{dir}/train_frames'
    elif train == "val":
        dir = f'{dir}/val_frames'
    else:
        dir = f'{dir}/test_frames'

    img_count = 0
    for img, label in loader:
        if not os.path.exists(f'{dir}/{classes[label]}'):
            os.makedirs(f'{dir}/{classes[label]}')
        print(f'We are currently saving: {dir}/{classes[label]}/{img_count}')
        torch.save(img, f'{dir}/{classes[label]}/{img_count}.tensorsaved')
        img_count += 1

all_file_names = []

def scanFiles(directory):
    for filename in os.scandir(directory):
        if filename.name == "README.md":
            continue
        if filename.is_dir():
            scanFiles(filename.path)
        if filename.is_file():
            all_file_names.append(filename.name)
            
def map_frames_to_vid(testing, file_names):
    if testing:
        img_arr = []
        idx = file_names[0].find('MOV')
        if idx == -1:
            idx = file_names[0].find('mov') 

        if idx == -1:
            vid_name = file_names[0]
        else:
            vid_name = file_names[0][:idx]

        video_to_frames = {}
        for img in file_names:
            curr_idx = img.find('MOV')
            if curr_idx == -1:
                curr_idx = img.find('mov')

            if curr_idx == -1:
                curr_name = img
            else:
                curr_name = img[:curr_idx]

            if curr_name != vid_name: #The next video is here
                video_to_frames[vid_name] = img_arr #Save current images of that video into hash table
                img_arr = [] #Reset image array
                vid_name = curr_name #New vid_name
            img_arr.append(img)
        return video_to_frames
    else:
        img_arr = []
        idx = all_file_names[0].find('.')
        vid_name = all_file_names[0][:idx]
        video_to_frames = {}
        for img in all_file_names:
            curr_idx = img.find('.')
            curr_name = img[:curr_idx]
            if curr_name != vid_name: #The next video is here
                video_to_frames[vid_name] = img_arr #Save current images of that video into hash table
                img_arr = [] #Reset image array
                vid_name = curr_name #New vid_name
            img_arr.append(img)
        return video_to_frames

def get_yawn_or_normal_files(list, yawning):

    for name in all_file_names:
        if yawning:
            idx = name.find("Yawning")
        else:
            idx = name.find("Normal")
        if idx != -1:
            list.append(name)

    return list
            

def main():
    
    print("Why are you not importing this file idiot")
    #Change the paths to your own paths
    #TODO maybe make it through user input
    #TODO wrap things into a class for more organization maybe
    scanFiles("C:/Users/leonz/github/aps360-proj/testing_dataset")
    yawn_files = []
    normal_files = []
    get_yawn_or_normal_files(yawn_files, True)
    get_yawn_or_normal_files(normal_files, False)
    yawn_set = map_frames_to_vid(True, yawn_files)
    normal_set = map_frames_to_vid(True, normal_files)
    #print(normal_set)
    # video_to_frames = map_frames_to_vid(False, [])
    # get_data(1, "C:/Users/User/Desktop/aps360-proj/dataset", video_to_frames)
    # #print(video_to_frames)
    # train_loader, data_loader = get_data(1, "C:/Users/User/Desktop/aps360-proj/dataset", video_to_frames)
    # save_features(train_loader, "train", "C:/Users/User/Desktop/aps360-proj/dataset")
    # save_features(data_loader, "val", "C:/Users/User/Desktop/aps360-proj/dataset")
    
    test_loader = get_testing_data(1, "C:/Users/leonz/github/aps360-proj/testing_dataset", yawn_set, normal_set)
    save_features(test_loader, "test", "C:/Users/leonz/github/aps360-proj/testing_dataset")
    

if __name__ == '__main__':
    main()