import csv
import sys
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
# Turn off all Warnings and Information messages
#   0 = all messages are logged (default behavior)
#   1 = INFO messages are not printed
#   2 = INFO and WARNING messages are not printed
#   3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print("Using Python version: " + sys.version)
#print("Is GPU avaliable: ", tf.test.is_gpu_available())

VERBOSE = False
print("********************************")
print(sys.platform)
# Set pictures base directory
prefix = "./photos/"

# people.csv has lines like this:
#   Key,Mother,Father,Sex,MonBorn,DayBorn,YrBorn
#   AdolphWeiss,,,M,0,0,1878
#   ArthurOHara,MaryOHara,HarryOHaraI,M,2,25,1925

def loadNames(fname):
  '''Collect info on all people'''
  try:
    with open(fname, mode='r') as csv_file:
      csv_reader = csv.DictReader(csv_file)
      people = {}
      for person in csv_reader:
        if VERBOSE and len(people) == 0:
          print(f'First person = {person}')
        people[person["Key"]] = person
  except Exception as ex:
    print(f'Trouble reading {fname} {ex}')
    return []
  print(f'Found {len(people)} people in {fname}.')
  return people

# .\data\ArthurOHara.csv has lines like this:
#   Directory,Photo,Suffix,Year,Month,Day,Left,Right,Top,Bottom,Width,Height
#   Jeanne_Photos/ohara,OHara_children_collage,.png,0,0,0,1254,9329,4806,18241,836,600
#   years-1980-1990/1985-1986,54_Art_in_Ystad_Sweden,.jpg,1986,6,0,29578,37312,14199,22828,483,600

def loadPerson(key, withYear):
  '''Load all the picture info for one person'''
  try:
    fname = f'./OneFamilyData/{key}.csv'
    with open(fname, mode='r') as csv_file:
      csv_reader = csv.DictReader(csv_file)
      pix = []
      for pic in csv_reader:
        if VERBOSE and len(pix) == 0:
          print(f'First picture = {pic}')
        if withYear == (pic["Year"] != '0'):   # Careful, comparing booleans
          pix.append(pic)
  except Exception as ex:
    print(f'Trouble reading {fname} {ex}')
    return []
  numPix = len(pix)
  which = ("with date" if withYear else "without date")
  print(f'==== {key} photos {which} = {numPix}')
  return pix

def getScaledImage(pic):
  fname = pic["Photo"] + pic["Suffix"]
  imageLoc = prefix + pic["Directory"] + "/" + fname

  try:
    data = tf.io.read_file(imageLoc)
    #data.to('cuda')
    image = tf.image.decode_jpeg(data, channels=3)   # 1 means grayscale, 3 for RGB
  except Exception as ex:
    print(f"**** Trouble reading {imageLoc} {ex}")
    return None

  width = int(pic["Width"])
  height = int(pic["Height"])
  x = int(int(pic["Left"]) / 65535.0 * width)
  y = int(int(pic["Top"]) / 65535.0 * height)
  w = int(int(pic["Right"]) / 65535.0 * width) - x
  h = int(int(pic["Bottom"]) / 65535.0 * height) - y
  if VERBOSE:
    print(f"         Orig W,H={width},{height}  Face x,y={x},{y} w,h={w},{h}")

  try:
    cropped = tf.image.crop_to_bounding_box(image, y, x, h, w)
  except Exception as ex:
    print(f"**** Error on {fname}, Orig W,H={width},{height}  Face x,y={x},{y} w,h={w},{h}")
    return None

  # resize the image to the desired size
  return tf.image.resize(cropped, [100, 60])

# Returns two equal sized lists, one with cropped input images and
# the other with output ages in each corresponding image.
# This is for one person at a time. E.g., ArthurOHara has 385 images.

def getScaledImagesWithDate(personAge, pix):
  '''Collect all the images and scale them'''
  imgs = []
  lbls = []
  for pic in pix:
    # Calcuate photo date, in fractional years
    photoAge = int(pic["Year"])
    photoAge += (int(pic["Month"]) if pic["Month"] != '0' else 6) / 12.0
    photoAge += (int(pic["Day"]) if pic["Day"] != '0' else 15) / 365.0
    ageYears = photoAge - personAge
    if VERBOSE:
      imageLoc = pic["Photo"]
      print(f'     Age is {ageYears:.2f} in {imageLoc}')

    scaled = getScaledImage(pic)
    imgs.append(scaled)
    lbls.append(ageYears)
  return (imgs,lbls)

def getScaledImagesNoDate(pix):
  '''Collect all the images and scale them'''
  imgs = []
  for pic in pix:
    scaled = getScaledImage(pic)
    imgs.append(scaled)
  return imgs

#
# Do all the processing for one person
#

def doPerson(ppl, key):
  '''Do all the processing for one person'''
  if not (key in ppl):
    print(f'Person {key} not found')
    return None

  pix = loadPerson(key, True)
  if len(pix) == 0:
    print(f'No dated photos found for {key}')
    return None

  # Calcuate person birthdate, in fractional years
  person = ppl[key]
  age = int(person["YrBorn"])
  age += (int(person["MonBorn"]) if person["MonBorn"] != '0' else 6) / 12.0
  age += (int(person["DayBorn"]) if person["DayBorn"] != '0' else 15) / 365.0
  if age == 0:
    print(f'Age of {key} is unknown')
    return None

  # Process each photo separately
  (images,labels) = getScaledImagesWithDate(age, pix)
  if len(images) == 0:
    print(f'Error processing images for {key}')
    return None

  return (images,labels)

#
# Show some faces for one person, but limit to what fits on the screen
#

def plotPerson(name, images, labels):
  '''Visualize faces and save them to image folders'''
  savePath = rf'./persons/{name}'
  if not os.path.exists(savePath):
    os.makedirs(savePath)

  count = len(images)
  for i in range(count):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    images[i] = images[i] / np.amax(images[i])
    plt.imshow(images[i])
    plt.axis('off')
    #plt.xlabel(f'{ttl} {labels[i]:.0f}')
    plt.savefig(f'./persons/{name}/{labels[i]:.0f}_{i}_{arg}.png', bbox_inches='tight', pad_inches=0)
  plt.show()

#
# Start of main
#
print(device_lib.list_local_devices())

if len(sys.argv) < 2:
  print("Usage: TrainData.py [-verbose] keynames")
  quit(1)

people = loadNames("people.csv")

for arg in sys.argv[1:]:
  if arg.lower() == "-verbose":
    VERBOSE = True
    continue

  result = doPerson(people, arg)
  if result == None: continue
  (images,labels) = result

# model = AgeModel.TrainModel(images, labels)
  plotPerson(arg, images, labels)

  # Now, let's run our model on pix with no dates
  pix = loadPerson(arg, False)
  scaled = getScaledImagesNoDate(pix)
  # ages = AgeModel.RunModel(model, scaled)
  # plotPerson(arg, scaled, ages, "Age ~")
