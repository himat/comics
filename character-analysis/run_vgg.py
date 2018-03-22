######
# python run_vgg.py --gpu 0
# This file runs a folder of images through a VGG-16 network and prints the top
#   5 classes predicted.
######

import keras
import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
import numpy as np
import os
import json
import argparse

vgg_weights_file = "../vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
vgg_classes_file = "../vgg16/imagenet_class_index.json"
images_folder = "../data/raw_panel_images/12/"

dirname = os.path.dirname(__file__)
vgg_weights_file = os.path.join(dirname, vgg_weights_file)
vgg_classes_file = os.path.join(dirname, vgg_classes_file)
images_folder = os.path.join(dirname, images_folder)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    return args

def get_vgg_classes():
    with open(vgg_classes_file) as f:
        class_dict = json.load(f)
    classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    return classes

def print_top_k(class_probs, k=5):
    classes_num_to_name = get_vgg_classes()

    class_probs = class_probs.squeeze(axis=0)
    assert(k <= len(class_probs))

    highest_classes = np.argpartition(class_probs, -k)[-k:]
    highest_classes = highest_classes[np.argsort(class_probs[highest_classes])[::-1]]

    for high_class in highest_classes:
        print(f"{classes_num_to_name[high_class]} with prob: {class_probs[high_class]}")

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    # VGG 16 input dims
    input_img_dims = (224, 224)

    # Load the model
    model_vgg16 = VGG16()
    model_vgg16.load_weights(vgg_weights_file)

    # Load and classify images
    for _im_file in os.listdir(images_folder):
        image_file = images_folder + _im_file
        print(f"\n\nim file: {image_file}")

        img = cv2.imread(image_file)
        img = cv2.resize(img, input_img_dims)

        # Run through model
        model_input = np.expand_dims(img, axis=0)
        # assert(maodel_input.ndim == 4)
        class_probs = model_vgg16.predict(model_input)
        assert(class_probs.shape == (1, 1000))

        print_top_k(class_probs)

        cv2.imshow("image", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
