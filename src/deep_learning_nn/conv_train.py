import os
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = False
TRAINING_DATA_FILE = "training_data.npy"

class DogsVCats():
  imgDir = '/home/paul/repos/pytorch_weather/resources/cats_dogs/PetImages/'
  IMG_SIZE = 50
  LABELS = {'Cat': 0, 'Dog': 1}
  training_data = []
  cat_count = 0
  dog_count = 0

  def make_training_data(self):
    for label in self.LABELS:
      print(label)
      print('----')
      p = self.imgDir + label
      for f in tqdm(os.listdir(p)):
        try:
            img = cv2.imread(os.path.join(p, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50,50))
            self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

            if label == 'Cat':
              self.cat_count += 1
            elif label == 'Dog':
              self.dog_count += 1
        except Exception as e:
          continue
    np.random.shuffle(self.training_data)
    print(len(self.training_data), self.dog_count, self.cat_count)
    np.save(TRAINING_DATA_FILE, self.training_data)

d = DogsVCats()
if REBUILD_DATA:
  d.make_training_data()
else:
  training_data = np.load(TRAINING_DATA_FILE, allow_pickle=True)
  print(len(training_data))

