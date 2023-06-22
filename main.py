# -------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------
import cv2
import numpy as np
from picamera import PiCamera
import picamera.array
from PIL import Image
from time import sleep
from pathlib import Path
from datetime import datetime, timedelta
from logzero import logger, logfile
from orbit import ISS, ephemeris
from skyfield.api import load
import os
import csv
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
BASE_FOLDER = Path(__file__).parent.resolve()

ORIGINAL_IMAGE_WIDTH = 2592  # resolution of camera
ORIGINAL_IMAGE_HEIGHT = 1952

RESIZED_IMAGE_WIDTH = 224  # to resize for the machine learning model
RESIZED_IMAGE_HEIGHT = 224

DARKNESS_THRESHOLD = 100  # for making image square

CLOUD_THRESHOLDS = (170, 170, 170)  # for removing clouds / water
WATER_THRESHOLDS = (65, 100, 140)

USEFUL_FRAC = 0.05  # fraction of image that needs to be useful to keep the image

LOOP_REPEAT_DELAY = 10  # the minimum number of seconds the main loop takes to repeat
ERROR_DELAY = 2  # so that the program does not create huge numbers of error messages

MODEL_PATH = f"{BASE_FOLDER}/model.tflite"  # files for machine learning
LABEL_PATH = f"{BASE_FOLDER}/labels.txt"

DATA_FILE = f"{BASE_FOLDER}/data.csv"  # file for data storage

# -------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------
camera = None

dengue_level = ""
probability = ""

#Used for image naming
img_count = 1

#Total storage taken up in bytes
storage = 0

#Counts how many times the loop has run
loop_counter = 1

#For machine learning
interpreter = None
labels = dict()

#To make sure the storage limit is not overrun
out_of_storage = False

# -------------------------------------------------------------------
# Subprograms
# -------------------------------------------------------------------


def is_daytime():
  #Returns boolean true if the ISS is in day time or false for night
  timescale = load.timescale()
  t = timescale.now()

  if ISS.at(t).is_sunlit(ephemeris):
    return True
  else:
    return False


## Image capturing
def camera_init():
  #Sets up camera
  global camera
  camera = PiCamera()
  camera.rotation = 180
  camera.resolution = (ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_HEIGHT)


def take_array_photo(
) -> np.array:  # modified from https://projects.raspberrypi.org/en/projects/astropi-ndvi/6
  # Take photo as a numpy array
  stream = picamera.array.PiRGBArray(camera)
  #adding array of image values to stream
  camera.capture(stream, format="bgr", use_video_port=True)
  image = stream.array

  return image


# NDVI calculations


def convert_to_ndvi(original: np.array) -> np.array:
  #Converts image array to ndvi data array,
  #keeping original size
  contrasted = contrast_stretch(original)
  ndvi = calc_ndvi(contrasted)
  ndvi_contrasted = contrast_stretch(ndvi)
  ndvi_contrasted = np.array(ndvi_contrasted, dtype=np.uint8)
  return ndvi_contrasted


def calc_ndvi(image):
  #Calculates ndvi index on array
  b, g, r = cv2.split(image)
  bottom = (r.astype(float) + b.astype(float))
  bottom[bottom == 0] = 0.01
  ndvi = (b.astype(float) - r) / bottom
  return ndvi


def contrast_stretch(
    im):  # from https://projects.raspberrypi.org/en/projects/astropi-ndvi/3
  #Increasing the image contrast before / after the ndvi calculation
  in_min = np.percentile(im, 5)
  in_max = np.percentile(im, 95)

  out_min = 0.0
  out_max = 255.0

  out = im - in_min
  try:
    out *= ((out_min - out_max) / (in_min - in_max))
  except ZeroDivisionError:
    logger.error(
      f"Zero division error in contrast_stretch on image {img_count}")

  out += in_min
  return out


#Image Processing


def resize(image: np.array) -> np.array:
  # Resizes image to make it
  # suitable for the machine learning model
  image = cv2.resize(image,
                     dsize=(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT),
                     interpolation=cv2.INTER_CUBIC)
  return image


def get_brightness_array(
    image):  # returns the total value (R+G+B) of each pixel
  return image.sum(axis=2)


def make_square(image: np.array) -> np.array:
  # Makes an image square, and crops any dark pixels away from the edges
  # all coordinates are (y, x) from the top left corner

  brightness_array = get_brightness_array(
    image)  # calculate brightness of every pixel
  bright_pixels = np.stack(np.where(brightness_array >= DARKNESS_THRESHOLD),
                           axis=1)  # coords of pixels brighter than threshold

  if len(
      bright_pixels) < 10:  # return blank image if no bright pixels are found
    return np.zeros((RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, 3),
                    dtype=np.uint8)

  total_height = len(image)  # original size of image
  total_width = len(image[0])
  min_height = min(bright_pixels[:, 0])  # extremes of bright region
  max_height = max(bright_pixels[:, 0])
  min_width = min(bright_pixels[:, 1])
  max_width = max(bright_pixels[:, 1])
  height_diff = max_height - min_height  # height of bright region
  width_diff = max_width - min_width  # width of bright region

  centre = [height_diff // 2 + min_height,
            width_diff // 2 + min_width]  # center of bright region
  final_radius = max((
    centre[0] - min_height,  # largest distance to the centre 
    centre[1] - min_width))
  final_radius = min((
    final_radius,  # make final radius smaller if it does not fit on the original image
    centre[0],
    centre[1],
    total_height - centre[0],
    total_width - centre[1]))

  final_image = np.split(
    image, [centre[0] - final_radius, centre[0] + final_radius],
    axis=0)[1].copy()  # - Cut image into three parts - top region of dark
  #   pixels, central region containing image, and
  #   bottom region of dark pixels
  # - Take second part and make a copy, so the original
  #   image can be discarded
  final_image = np.split(final_image,
                         [centre[1] - final_radius, centre[1] + final_radius],
                         axis=1)[1].copy()  # Same as above, but for x axis

  return final_image


def make_image(array):
  #Convert array of BGR values to Image object

  # Convert from BGR (used by cv2) to RGB (used by PIL)
  if len(array.shape) == 3:
    array = np.copy(np.flip(array, axis=2))

  image = Image.fromarray(array)
  return image


def process(photo_array: np.array) -> tuple[bool, np.array]:
  # Processes a photo.
  # 1) Photo is made square, so the ISS's window fills the photo
  # 2) Pixels with BGR lower than water threshold or higher than cloud threshold are replaced with black
  # 3) If the fraction of the image that is not black is too low, it is discarded
  # 4) Returns boolean of whether the image is useful, and the processed image if it is useful
  photo_array = make_square(photo_array)

  blue_array, green_array, red_array = cv2.split(photo_array)

  # Every pixel in the image is labelled either True (to be replaced with black) or False (to keep)
  # Logical statement and values of constants determined through trial and error
  truth_array = np.expand_dims(np.logical_or(
    np.logical_and(
      np.logical_and(np.where(blue_array < WATER_THRESHOLDS[0], True, False),
                     np.where(green_array < WATER_THRESHOLDS[1], True, False)),
      np.where(red_array < WATER_THRESHOLDS[2], True, False)),
    np.logical_and(
      np.logical_and(np.where(blue_array > CLOUD_THRESHOLDS[0], True, False),
                     np.where(green_array > CLOUD_THRESHOLDS[1], True, False)),
      np.where(red_array > CLOUD_THRESHOLDS[2], True, False))),
                               axis=2)

  amount_black = np.count_nonzero(
    truth_array)  # Calculate the fraction of the image that has been kept
  image_size = np.prod(photo_array.shape[:2])
  amount_useful = 1 - (amount_black / image_size)

  if amount_useful > USEFUL_FRAC:  # If the image is useful
    black_array = np.zeros(photo_array.shape, dtype=np.uint8)
    photo_array_black = np.where(
      truth_array, black_array,
      photo_array)  # combine black and original photo depending on truth array
    return (True, photo_array_black)

  else:
    return (
      False, photo_array
    )  # Return the original image to keep the function's output consistent


# CSV data table


def create_csv_file(DATA_FILE):
  # Create a new CSV file and add the header row
  with open(DATA_FILE, "w") as f:
    writer = csv.writer(f)
    header = ("Image number", "Dengue fever risk", "Probability", "Date/Time",
              "Latitude", "Longitude")
    writer.writerow(header)


def add_csv_data(DATA_FILE, data):
  # Add a row of data to the data_file CSV
  with open(DATA_FILE, "a") as f:
    writer = csv.writer(f)
    writer.writerow(data)


# Machine learning


def classify_image(
    interpreter,
    image):  # from https://teachablemachine.withgoogle.com/train/image
  # Classifies an image using the machine learning model into either
  # No risk, Low risk or High risk; returns probability of each category
  size = common.input_size(interpreter)
  common.set_input(
    interpreter,
    cv2.resize(image, size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC))
  interpreter.invoke()
  return classify.get_classes(interpreter)


def machine_learning_init(
):  # modified from https://teachablemachine.withgoogle.com/train/image
  # set up machine learning variables
  global interpreter, labels
  interpreter = make_interpreter(MODEL_PATH)
  interpreter.allocate_tensors()
  labels = read_label_file(LABEL_PATH)


# -------------------------------------------------------------------
# Main program
# -------------------------------------------------------------------

#Set a logfile name
logfile(BASE_FOLDER / "events.log")

#Set up CSV file for Dengue fever risk levels
create_csv_file(DATA_FILE)

#Initialise camera
camera_init()

#Initialise machine learning
machine_learning_init()

#Record the start and current time
start_time = datetime.now()
now_time = datetime.now()

# Run a loop for time minutes (must finish in 180-time taken for 1 loop mins)
while (now_time < start_time + timedelta(minutes=178)):

  logger.info(f"Iteration {loop_counter}")

  # Update the current time
  now_time = datetime.now()
  daytime = is_daytime()

  try:
    # Main program

    # Keeping under the storage limit of 3GB with a safe buffer (leaving 200 MB for text data)
    if storage > 2_800_000_000 and not out_of_storage:
      logger.error("Exceeded maximum storage limit")
      out_of_storage = True

    if daytime:

      path = f"{BASE_FOLDER}/Images/image_{img_count:04d}.jpg"

      original_photo = take_array_photo()
      # don't save the photo if there is no storage space left
      if not out_of_storage:
        make_image(original_photo).save(path)
        # immediately save the file
        with open(path, "r") as f:
          f.flush()
          os.fsync(f.fileno())

      # Crop image, remove clouds / water and check it contains enough data to be useful
      useful, processed_photo = process(original_photo)

      if not useful:
        #If image is not useful the next loop will start and the image will be overwritten as img_count hasn't been increased
        logger.info("Image not useful")
        loop_counter += 1
        sleep(
          max(0,
              LOOP_REPEAT_DELAY - (datetime.now() - now_time).total_seconds()))
        continue

      logger.info("Image useful")

      #ndvi conversion
      resized_photo = resize(processed_photo)
      ndvi_array = convert_to_ndvi(resized_photo)
      # Duplicate each pixel brightness value so the image can be read in RGB format
      ndvi_array_rgb = np.repeat(ndvi_array[:, :, np.newaxis], 3, axis=2)

      #Machine learning
      results = classify_image(interpreter, ndvi_array_rgb)
      dengue_level = f"{labels[results[0].id]}"
      probability = f"{results[0].score}"

      #Add data to CSV file
      location = ISS.coordinates()
      data = (f"{img_count:04d}", dengue_level, probability, datetime.now(),
              location.latitude.degrees, location.longitude.degrees)
      add_csv_data(DATA_FILE, data)
      #Add image size in bytes to total storage - text data is ignored as it takes up so little space
      if not out_of_storage:
        storage += os.path.getsize(path)
      img_count += 1

    else:
      # Wait 10 seconds before checking if the ISS is sunlit again
      sleep(
        max(0,
            LOOP_REPEAT_DELAY - (datetime.now() - now_time).total_seconds()))
      continue

    #Wait 10 seconds to run loop again
    loop_counter += 1
    sleep(
      max(0, LOOP_REPEAT_DELAY - (datetime.now() - now_time).total_seconds()))

  except Exception as e:
    #Handles any errors logging them with the error type
    logger.error(f"{e.__class__.__name__}: {e}")
    img_count += 1
    # Wait 2 seconds to avoid generating excessive numbers of errors
    sleep(max(0, ERROR_DELAY - (datetime.now() - now_time).total_seconds()))

#Closing files
camera.close()

logger.info("Execution complete.")
