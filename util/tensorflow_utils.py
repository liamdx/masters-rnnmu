import tensorflow as tf
import cv2 
import sys
import numpy as np

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value = [value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def load_image(addr):
    img = cv2.imread(addr)
    if img is None:
        print("Failed to load image: " + addr)
        return None
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.resize(img, (128,960))
    img = np.array(img)
    img = np.divide(img, 255.0)
    return img

def createDataset(filename, img_paths, labels):
    # writer = tf.python_io.TFRecordWriter(filename)
    _data = []
    _labels = []
    
    for k, v in img_paths.items():
        img = load_image(v["Piano"])
        label = labels[k]
        if img is None:
            continue
        
        _data.append(img)
        _labels.append(label)
        # create a feature (?)
        
        
#        _feature = {
#            'image_raw': bytes_feature(img.tostring()),
#            'label': float_feature(label)
#        }
#        
        # example = tf.train.Example(features=tf.train.Features(feature=_feature))
        #writer.write(example.SerializeToString())
    return _data, _labels
    # writer.close()
    # sys.stdout.flush()