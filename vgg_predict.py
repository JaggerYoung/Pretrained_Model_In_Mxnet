import find_mxnet
import mxnet as mx
import logging
import time
import cv2
import random
import glob

BATCH_SIZE = 15
NUM_SAMPLES = 1

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
	self.label = label
	self.data_names = data_names
	self.label_names = label_names

	self.pad = 0
	self.index = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n,x in zip(self.label_names, self.label)]

def readData(FileName):
    data_1 = []
    data_2 = []
    data_3 = []
    f = open(FileName,'r')
    total = f.readlines()

    for eachLine in range(len(total)):
        tmp = total[eachLine].split('\n')
	tmp_1, tmp_2, tmp_3 = tmp[0].split(' ',2)
	tmp_1 = '/home/yzg/UCF-101'+tmp_1
	data_1.append(tmp_1)
	data_2.append(tmp_2)
	data_3.append(tmp_3)
    f.close()
    return (data_1, data_2, data_3)

def ImageSeqToMatrix(dirName, num, data_shape):
    pic = []
    for filename in glob.glob(dirName+'*.jpg'):
        pic.append(filename)
    
    #ret = []
    #len_pic = len(pic)
    #tmp = len_pic/num
    #for i in range(num):
    #    ret.append(pic[i*tmp])
    r_1 = []
    g_1 = []
    b_1 = []
    mat = []
    #for i in range(len(ret)):
    img = cv2.imread(pic[0])
    b,g,r = cv2.split(img)
    r = cv2.resize(r, (data_shape[2], data_shape[1]))
    g = cv2.resize(g, (data_shape[2], data_shape[1]))
    b = cv2.resize(b, (data_shape[2], data_shape[1]))
    #r_1.append(r)
    #g_1.append(g)
    #b_1.append(b)

    mat.append(r)
    mat.append(g)
    mat.append(b)

    return mat

class VGGIter(mx.io.DataIter):
    def __init__(self, fname, num, batch_size, data_shape):
         self.batch_size = batch_size
	 self.fname = fname
	 self.data_shape = data_shape
	 self.count = num/batch_size
         (self.data_1, self.data_2, self.data_3) = readData(fname)

	 self.provide_data = [('data', (batch_size,) + data_shape)]
	 self.provide_label = [('label',(batch_size,))]

    def __iter__(self):
         for k in range(self.count):
	     data = []
	     label = []
	     for i in range(self.batch_size):
	         idx = k * batch_size + i
		 pic = ImageSeqToMatrix(self.data_1[idx], NUM_SAMPLES, self.data_shape)
		 data.append(pic)
		 label.append(int(self.data_3[idx]))
	
	     data_all = [mx.nd.array(data)]
	     label_all = [mx.nd.array(label)]
	     data_names = ['data']
	     label_names = ['label']

	     data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
	     yield data_batch
    
    def reset(self):
        pass

if __name__ == '__main__':
    
    train_num = 107258
    #train_num = 100
    test_num = 41822

    batch_size = BATCH_SIZE
    data_shape = (3, 224, 224)
    num_label = 101

    train_file = '/home/yzg/mxnet/example/C3D_UCF101/data/train.lst'
    test_file = '/home/yzg/mxnet/example/C3D_UCF101/data/test.lst'

    data_train = VGGIter(train_file, train_num, batch_size, data_shape)
    data_val = VGGIter(test_file, test_num, batch_size, data_shape)
    
    print data_train.provide_data
    devs = [mx.context.gpu(0)]

    model = mx.model.FeedForward.load("./vgg_model/vgg16", epoch=00, ctx=devs, numpy_batch_size=BATCH_SIZE)

    internals = model.symbol.get_internals()
    print internals.list_outputs()
    fea_symbol = internals['relu7_output']
    feature_extractor = mx.model.FeedForward(ctx=devs, symbol=fea_symbol, numpy_batch_size=1,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
					     allow_extra_params=True)
    vgg_result = feature_extractor.predict(data_train)

    print vgg_result
