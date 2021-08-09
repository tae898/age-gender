"""
This is just a simple client example. Hack it as much as you want. 
"""
import argparse
import requests
import jsonpickle
import logging
from PIL import Image, ImageDraw, ImageFont
import pickle
import numpy as np
import io

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main(url_face: str, url_age_gender: str, image_path: str):

    logging.debug(f"loading image ...")
    if isinstance(image_path, str):
        with open(image_path, 'rb') as stream:
            binary_image = stream.read()
    elif isinstance(image_path, Image.Image):
        binary_image = io.BytesIO()
        image_path.save(binary_image, format='JPEG')
        binary_image = binary_image.getvalue()

    data = {'image': binary_image}
    logging.info(f"image loaded!")

    logging.debug(f"sending image to server...")
    data = jsonpickle.encode(data)
    response = requests.post(url_face, json=data)
    logging.info(f"got {response} from server!...")
    response = jsonpickle.decode(response.text)

    face_detection_recognition = response['face_detection_recognition']
    logging.info(f"{len(face_detection_recognition)} faces deteced!")

    bboxes = [fdr['bbox'] for fdr in face_detection_recognition]
    det_scores = [fdr['det_score'] for fdr in face_detection_recognition]
    landmarks = [fdr['landmark'] for fdr in face_detection_recognition]

    logging.debug(f"sending embeddings to server ...")
    data = [fdr['normed_embedding'] for fdr in face_detection_recognition]

    # -1 accounts for the batch size.
    data = np.array(data).reshape(-1, 512).astype(np.float32)

    # I wanna get rid of this pickling part but dunno how.
    data = pickle.dumps(data)

    data = {'embeddings': data}
    data = jsonpickle.encode(data)
    response = requests.post(url_age_gender, json=data)
    logging.info(f"got {response} from server!...")

    response = jsonpickle.decode(response.text)
    ages = response['ages']
    genders = response['genders']

    logging.debug(f"annotating image ...")
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("fonts/arial.ttf", 25)

    for gender, age, bbox in zip(genders, ages, bboxes):
        draw.rectangle(bbox.tolist(), outline=(0, 0, 0))
        draw.text((bbox[0], bbox[1]), "AGE: " +
                  str(age), fill=(255, 0, 0), font=font)
        draw.text((bbox[0], bbox[3]), 'MALE ' + str(round(gender['m']*100)) + str("%") +
                  ', ' 'FEMALE ' + str(round(gender['f']*100)) + str("%"), fill=(0, 255, 0), font=font)
        image.save(image_path + '.ANNOTATED.jpg')
    logging.debug(
        f"image annotated and saved at {image_path + '.ANNOTATED.jpg'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract face, gender, and age.')
    parser.add_argument('--url-face', type=str,
                        default='http://127.0.0.1:10002/')
    parser.add_argument('--url-age-gender', type=str,
                        default='http://127.0.0.1:10003/')
    parser.add_argument('--image-path', type=str)

    args = vars(parser.parse_args())

    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)
