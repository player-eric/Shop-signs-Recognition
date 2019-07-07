import csv
from PIL import Image
import numpy as np
import random
import h5py


def cropping():
    images = []
    csvfile = open("train.csv", "r")
    reader = csv.reader(csvfile)
    for item in reader:
        image = {}
        image["path"] = item[0]
        image["label"] = item[1]
        image["start_index_x"] = item[2]
        image["start_index_y"] = item[3]
        image["end_index_x"] = item[4]
        image["end_index_y"] = item[5]
        images.append(image)

    max_x = 0
    max_y = 0
    for image in images:
        if image["path"] != "filename":
            img = Image.open("train/" + image["path"])
            print(images.index(image), "/", len(images))
            x = int(image["start_index_x"])
            y = int(image["start_index_y"])
            w = int(image["end_index_x"]) - int(image["start_index_x"])
            h = int(image["end_index_y"]) - int(image["start_index_y"])
            region = img.crop((x, y, x + w, y + h))
            region.save("cropped/" + image["path"])
            if w > max_x:
                max_x = w
            if h > max_y:
                max_y = h
    # max_x=456 max_y=172


def padding():
    paths = []
    csvfile = open("train.csv", "r")
    reader = csv.reader(csvfile)
    for item in reader:
        paths.append(item[0])

    desired_size = 456
    for path in paths:
        if path != "filename":
            im = Image.open("cropped/" + path)
            old_size = im.size
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            im = im.resize(new_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(
                im,
                ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2),
            )
            new_im.save("padding/" + path)
            print(paths.index(path), "/", len(paths))


def compress():
    paths = []
    csvfile = open("train.csv", "r")
    reader = csv.reader(csvfile)
    for item in reader:
        paths.append(item[0])
    target_size = 128
    for path in paths:
        if path != "filename":
            im = Image.open("ai_final/cropped/" + path)
            im = im.resize((target_size, target_size), Image.ANTIALIAS)
            im.save("ai_final/compressed/" + path)
            print(paths.index(path), "/", len(paths))


def store_dataset():
    images = []
    csvfile = open("train.csv", "r")
    reader = csv.reader(csvfile)
    for item in reader:
        image = {}
        image["path"] = item[0]
        image["label"] = item[1]
        image["start_index_x"] = item[2]
        image["start_index_y"] = item[3]
        image["end_index_x"] = item[4]
        image["end_index_y"] = item[5]
        images.append(image)

    random.shuffle(images)

    X_whole = []
    Y_whole = []
    classes = []
    for image in images:
        if image["path"] != "filename":
            img = Image.open("compressed/" + image["path"])
            img_arr = np.array(img)
            X_whole.append(img_arr)
            print(images.index(image), "/", len(images), " ", image["path"])
            Y_whole.append(int(image["label"]))
    print("images stored successfully")

    data_file = h5py.File("data_compressed.hdf5", "w")
    X_whole = np.array(X_whole)
    data_file["X_whole"] = X_whole
    Y_whole = np.array(Y_whole)
    data_file["Y_whole"] = Y_whole


def load_dataset():
    data = h5py.File("data_compressed.hdf5", "r")
    X_whole = np.array(data["X_whole"][:])
    Y_whole = np.array(data["Y_whole"][:])
    Y_whole = Y_whole.reshape((1, Y_whole.shape[0]))
    print("data loaded successfully")
    return X_whole, Y_whole


def divide_dataset(training_ratio):
    X_whole, Y_whole = load_dataset()
    Y_whole = Y_whole - 1
    training_size = int(Y_whole.shape[1] * training_ratio / 10)

    X_train_orig = X_whole[0:training_size]
    X_test_orig = X_whole[training_size:]
    Y_train_orig = Y_whole[0:, 0:training_size]
    Y_test_orig = Y_whole[0:, training_size:]
    return X_train_orig, X_test_orig, Y_train_orig, Y_test_orig


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
