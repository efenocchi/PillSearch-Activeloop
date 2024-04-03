from typing import Optional, Union

import yaml
from FastSAM.fastsam import FastSAM, FastSAMPrompt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os
from PIL import Image
import cv2
import torch

IMAGES_TO_DISCARD = "images_to_discard.yaml"
IMAGES_TO_NOT_MASK = "images_to_not_mask.yaml"


"""
Mask the images using FastSAM model and save the masked images in the output folder.
There are some images that are not pills images, so they are discarded.
After the first masking process, if an image has not been masked well, it is added to the not_to_mask list. These images will be saved to the output folder without being masked.
"""


class FastSAM_segmentation:
    """
    FastSAM model interface.
    process: process the image and return the masked image
    """

    def __init__(self, device: Union[str, int] = "cpu"):
        import os

        local_path = os.path.dirname(os.path.realpath(__file__))
        self.model = FastSAM(local_path + "/FastSAM/weights/FastSAM.pt")
        self.device = device

    def process(self, image_path: str, output_name: str, test=False) -> Image:
        everything_results = self.model(
            image_path,
            device=self.device,
            retina_masks=True,
            # imgsz=1024,
            conf=0.4,
            iou=0.9,
        )
        prompt_process = FastSAMPrompt(
            image_path, everything_results, device=self.device
        )
        ann = prompt_process.everything_prompt()

        # text prompt
        ann = prompt_process.text_prompt(text="only the first or above pill")
        # original_im
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        try:
            mask_array = np.array(ann[0])
            # plt.imshow(original_im)

            # Ensure the mask is a binary mask (True/False)
            # binary_mask = mask_array.astype(np.uint8) * 255

            masked_complete = img.copy()
            masked_complete = img * np.expand_dims(mask_array, axis=-1)
            if test:
                os.makedirs("./test", exist_ok=True)
                cv2.imwrite(f"./test/{output_name}_masked.png", masked_complete)
            else:
                os.makedirs("./output", exist_ok=True)
                cv2.imwrite(f"./output/{output_name}_masked.png", masked_complete)

            # original_im.save("./output/original_im.jpg")
        except:
            if test:
                os.makedirs("./test", exist_ok=True)
                cv2.imwrite(f"./test/{output_name}_masked.png", img)
                masked_complete = img.copy()
            else:
                os.makedirs("./output", exist_ok=True)
                cv2.imwrite(f"./output/{output_name}_masked.png", img)
                masked_complete = img.copy()

        return masked_complete


def segment_image(image_paths: Optional[list[str]], test=False):
    """
    Segment the images and return the masked images.
    Used to compute the embeddings for the images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSAM_segmentation(device)
    masked_images = []
    for image_path in image_paths:
        image_name = image_path.split("/")[-1].split(".")[0]
        masked_images.append(
            model.process(image_path=image_path, output_name=image_name, test=test)
        )

    return masked_images


def save_not_masked_images(not_to_mask: Optional[list]):
    """
    Save and return the not masked images.
    """
    masked_images = []
    for image_path in not_to_mask:
        image_name = image_path.split("/")[-1].split(".")[0]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cv2.imwrite(f"./output/{image_name}_masked.png", img)
        masked_images.append(img)

    return masked_images


if __name__ == "__main__":
    # not pills images, to discard
    with open(IMAGES_TO_DISCARD, "r") as file:
        discard = yaml.safe_load(file)["pills_images_to_discard"]["name"]
    with open(IMAGES_TO_NOT_MASK, "r") as file:
        not_mask = yaml.safe_load(file)["pills_images_to_not_mask"]["name"]

    already_masked = os.listdir("./output")
    already_masked = [
        filename.split("_masked")[0]
        for filename in already_masked
        if filename.endswith(".png")
    ]

    image_folder = "./images"

    not_mask = [os.path.join(image_folder, filename) for filename in not_mask]

    image_paths = [
        os.path.join(image_folder, filename)
        for filename in os.listdir(image_folder)
        if filename.endswith(".jpg")
        and filename.split("/")[-1].split(".")[0]
        not in discard + already_masked + not_mask
    ]

    image_paths.sort()
    masked_images = segment_image(image_paths)
    not_masked_images = save_not_masked_images(not_mask)

    # print(f"masked {len(masked_images)} images")
