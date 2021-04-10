import numpy as np
import cv2
import os
import torch 
from torch import nn
from tqdm import trange
X = []
Y = []
width,height = 256,256


classes = os.listdir("dataset")
print(classes)

for i in range(len(classes)):
  c = classes[i]
  files = os.listdir("dataset/"+c)
  path = "dataset/"
  path = path + c+"/"
  print("current class: ",c)
  for f in range(len(files)):
    #print(files[f])
    
    src = cv2.imread(path+files[f])
    
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)
    #output = cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
    output = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
    #output = cv2.cvtColor(resize)
  
    X.append(output)
    Y.append(i)



X = np.array(X)
Y = np.array(Y,dtype=float)
#X = X.reshape(X.shape[0],X.shape[-1],X.shape[2],X.shape[2])

#Y = np.array(Y)
#print(X.shape)
#print(X[0][0][0])
fac = 0.99 / 255
#print(np.array(X).reshape(241,3,256,256).shape)
#train_imgs = np.asfarray(X[:, 1:],dtype="float32") * fac + 0.01
np.save('/Users/loic/toast_classifier/train_imgs',np.array(X))
np.save('/Users/loic/toast_classifier/train_labels',np.array(X))


path = "/Users/loic/toast_classifier/"
imgs = np.load(os.path.join(path,'train_imgs.npy'))
labels = np.load(os.path.join(path,'train_labels.npy'))
print(imgs[9].shape)

import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid
def show_grid():
  fig = plt.figure(figsize=(8., 8.))

  grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(6, 6),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

  import random 
  x = random.randint(0,len(X)-1)
  for ax, im in zip(grid, X[x:]):
    # Iterating over the grid returns the Axes.
    #print(im.shape)
    #im = im.reshape(im.shape[1],im.shape[1],im.shape[0])
    ax.imshow(im)


  plt.show()
#show_grid()
"""
class the_net(torch.nn.Module):
    def __init__(self):
        super(the_net,self).__init__()
        self.l1 = nn.Linear(784,128,bias=False)
        #self.l2 = nn.Linear(128,10,bias=True)
        #self.l3 = nn.Linear(32,10,bias=False)
        #self.l2 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128,10,bias=False)
        #self.l3 = nn.Linear(64,10,bias=False)
        #self.l4 = nn.Bilinear(64,10,10,bias=False)
        
        ############### softmin suuuuuuucks ##################
        self.soft_max = nn.Softmax(dim=1)
    def forward(self,x):
        x = F.relu(self.l1(x))
        
        #x = F.relu(self.l2(x))
        x = self.l2(x)
        #x = self.l3(x)
       # x = self.l3(x)
        x = self.soft_max(x)
        return x
"""
class neural_net(torch.nn.Module):
  def __init__(self):
    super(neural_net,self).__init__()
    self.layer1 = nn.Linear(256*256,128*128,bias=True)
    self.layer2 = nn.Linear(128*128,64*64,bias=False)
    #self.layer3 = nn.Linear(64*64,24*24,bias=False)
    #self.layer4 = nn.Linear(24*24,256,bias=False)
    #self.layer5 = nn.Linear(256,64,bias=False)
    #self.layer6 = nn.Linear(64,6,bias=False)
    #self.layer3 = nn.Linear(64*64,32*32,bias=False)
    #self.layer4 = nn.Linear(32*32,10*10,bias=False)
    self.layer3 = nn.Linear(64*64,10*10,bias=False)
    self.layer4 = nn.Linear(10*10,6,bias=False)


    self.soft_max = nn.Softmax(dim=1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
  def forward(self,x):
    x = self.relu(self.layer1(x))
    #x = self.relu(self.layer2(x))
    #x = self.relu(self.layer3(x))
    #x = self.relu(self.layer4(x))
    #x = self.relu(self.layer5(x))
    #x = self.layer5(self.layer4(self.layer3(self.layer2(x))))
    #x = self.soft_max(self.layer6(x))
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.sigmoid(x)
    return x

model = neural_net()


loss_function = nn.NLLLoss(reduction="none")
#loss_function = nn.CrossEntropyLoss(reduction="mean")
#qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
#torch.backends.quantized.engine = 'qnnpack'
optim = torch.optim.Adam(model.parameters(),lr=0.0001)
iters = 400
bs = 64
losses, accuracies = [], []

for i in (t := trange(iters)):
    samp = np.random.randint(0,X.shape[0],size=(bs))
    X2 = torch.tensor(X[samp].reshape((-1,256*256))).float()
    Y2 = torch.tensor(Y[samp]).long()
    #print(Y2.size(),Y2)
    #exit()
    model.zero_grad()
    out = model(X2)
    cat = torch.argmax(out,dim=1)
    accuracy = (cat ==Y2).float().mean()
    loss = loss_function(out,Y2)
    loss = loss.mean()
    loss.backward()
    optim.step()
    loss,accuracy = loss.item(), accuracy.item()
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f"%(loss,accuracy))
plot(losses)
plot(accuracies)
model.eval()