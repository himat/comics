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
import random
import os
import json
import argparse

vgg_weights_file = "../vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
vgg_classes_file = "../vgg16/imagenet_class_index.json"
images_base_folder = "../data/raw_panel_images/"
    
# VGG 16 input dims
input_img_dims = (224, 224)

bad_classes = ["comic_book", "book_jacket"]

dirname = os.path.dirname(__file__)
vgg_weights_file = os.path.join(dirname, vgg_weights_file)
vgg_classes_file = os.path.join(dirname, vgg_classes_file)
images_base_folder = os.path.join(dirname, images_base_folder)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="Which gpu to use")
    parser.add_argument("--all-folders", action="store_true", help="If flag set, then will classify images in all folders")
    parser.add_argument("--bw", action="store_true", help="Convert images to black and white before passing through classifier")
    parser.add_argument("--show-images", action="store_true", help="Render images that are not badly classified")
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

    highest_classes = [classes_num_to_name[high_class] for high_class in highest_classes]
    return highest_classes


# images_folder: folder that directly contains the images (no subfolders inside it)
def classify_in_folder(model_vgg16, images_folder, image_color, show_good_images):

    # stats
    num_total_images = 0
    num_bad_classified = 0

    # Load and classify images
    for _im_file in os.listdir(images_folder):
        image_file = images_folder + _im_file
        print(f"\n\nim file: {image_file}")
        num_total_images += 1

        img = cv2.imread(image_file, image_color)
        img = cv2.resize(img, input_img_dims) # Reduce size to be appropriate for VGG input

        # Run through model
        model_input_img = np.expand_dims(img, axis=0).astype("float64") # For batch size
        if image_color == cv2.IMREAD_GRAYSCALE: 
            # Doesn't load the single channel by default, so add extra dim
            model_input_img = np.expand_dims(model_input_img, axis=3)
            # Repeating the b/w image along the 3 channels
            model_input_img = np.repeat(model_input_img, 3, axis=3) 
        model_input_img = keras.applications.vgg16.preprocess_input(model_input_img)
        assert(model_input_img.ndim == 4)

        class_probs = model_vgg16.predict(model_input_img)
        assert(class_probs.shape == (1, 1000))

        highest_classes = print_top_k(class_probs)

        if highest_classes[0] in bad_classes:
            num_bad_classified += 1
        elif show_good_images:
            cv2.imshow("image", img)
            while cv2.getWindowProperty("image", 0) >= 0:
                key = cv2.waitKey(50)
                if key > -1:
                    cv2.destroyAllWindows()
                    break
    
    # Done
    error_rate = (num_bad_classified / num_total_images) * 100
    print("\n\n{}/{} ({}%) images classified as one of {}".format(num_bad_classified, num_total_images, error_rate, bad_classes))

    return num_bad_classified, num_total_images

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    image_color = cv2.IMREAD_GRAYSCALE if args.bw else cv2.IMREAD_UNCHANGED
    
    # Load the model
    model_vgg16 = VGG16()
    model_vgg16.load_weights(vgg_weights_file)

    if args.all_folders:
        num_bad_classified = 0
        num_total_images = 0
        for images_folder_num in sorted(os.listdir(images_base_folder)):
            images_folder = images_base_folder + images_folder_num + "/"
            num_bad_classified_i, num_total_images_i = classify_in_folder(model_vgg16, images_folder, image_color, args.show_images)
            num_bad_classified += num_bad_classified_i
            num_total_images += num_total_images_i

        error_rate = (num_bad_classified / num_total_images) * 100
        print("\n\n{}/{} ({}%) images classified as one of {}".format(num_bad_classified, num_total_images, error_rate, bad_classes))
    else:
        # images_folder_num = random.choice(os.listdir(images_base_folder))
        images_folder_num = "2726"
        images_folder = images_base_folder + images_folder_num + "/"
        classify_in_folder(model_vgg16, images_folder, image_color, args.show_images)



if __name__ == "__main__":
    main()
