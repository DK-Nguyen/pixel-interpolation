#%%
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams.update({'font.size': 8})
from PIL import Image
from nnInterpolation import nearestNeighbor
from bilinearInterpolation import bi
from ppgInterpolation import ppg
import time
from qualityMeasurement import mse, mae

#%% Data Preprocessing
def readimagefile(filename,imSize,imType):
    rawData = open(filename,'rb').read()
    rawImage = Image.frombytes('F',imSize,rawData,'raw',imType)
    rawImage = np.asarray(rawImage)
    # normalizing
#    rawImage = rawImage / (2**10 - 1)
    if filename == 'raw_image2' or filename == 'raw_image5':
        rawImage = rawImage / (2**10 - 1)
    else: 
        rawImage = rawImage / (2**8 - 1)
        
    R = np.zeros(rawImage.shape)  
    G = np.zeros(rawImage.shape)
    B = np.zeros(rawImage.shape)
    
    R[1::2,1::2] = rawImage[1::2,1::2]
    G[0::2, 1::2] = rawImage[0::2, 1::2]
    G[1::2, 0::2] = rawImage[1::2, 0::2]
    B[0::2,0::2] = rawImage[0::2,0::2]
    
    return [R, G, B]

#%% Read the images
[R2, G2, B2] = readimagefile('raw_image2', (1008, 1018), 'F;16')
raw2 = R2+G2+B2
[R5, G5, B5] = readimagefile('raw_image5', (1008, 1018), 'F;16')
raw5 = R5+G5+B5
[Rtest, Gtest, Btest] = readimagefile('testikuva.raw', (512, 512), 'F;8')
rawTest = Rtest + Gtest + Btest
imTest = np.array(Image.open('testikuva.tiff'))
#%% Function to plot stuff
def plotting(images, names, method):
    figure = plt.figure()
    figure.suptitle(method)
    plt.subplot(3,4,1); plt.imshow(images[0]); plt.axis('off'); plt.title("Red Channel")
    plt.subplot(3,4,2); plt.imshow(images[1]); plt.axis('off'); plt.title("Green Channel")
    plt.subplot(3,4,3); plt.imshow(images[2]); plt.axis('off'); plt.title("Blue Channel")
    plt.subplot(3,4,4); plt.imshow(images[3]); plt.axis('off'); plt.title(names[0])

    plt.subplot(3,4,5); plt.imshow(images[4]); plt.axis('off'); plt.title("Red Channel")
    plt.subplot(3,4,6); plt.imshow(images[5]); plt.axis('off'); plt.title("Green Channel")
    plt.subplot(3,4,7); plt.imshow(images[6]); plt.axis('off'); plt.title("Blue Channel")
    plt.subplot(3,4,8); plt.imshow(images[7]); plt.axis('off'); plt.title(names[1])
     
    plt.subplot(3,4,9); plt.imshow(images[8]); plt.axis('off'); plt.title("Red Channel")
    plt.subplot(3,4,10); plt.imshow(images[9]); plt.axis('off'); plt.title("Green Channel")
    plt.subplot(3,4,11); plt.imshow(images[10]); plt.axis('off'); plt.title("Blue Channel")
    plt.subplot(3,4,12); plt.imshow(images[11]); plt.axis('off'); plt.title(names[2])
    figure.show()

def plottingPerformance(method, x_values, y_values):
    figure = plt.figure()
    figure.suptitle(method)
    x_pos = np.arange(len(x_values))
    plt.xticks(x_pos, x_values)
    plt.ylabel(method)
    plt.bar(x_pos, y_values, align='center', alpha=0.5)

#%% Plotting the raw images and their channels
images = [R2, G2, B2, raw2, R5, G5, B5, raw5, Rtest, Gtest, Btest, rawTest]
names = ["raw_image2", "raw_image5", "testikuva.raw"]
plotting(images, names, "Raw Images")

#%% METHOD 1: NEAREST NEIGHBOR INTERPOLATION
# pros: simple, computationally fast. cons: lower quality output than other methods
start = time. time()
[newR2_nn, newG2_nn, newB2_nn, interpolatedIm2_nn] = nearestNeighbor(R2, G2, B2)
end = time. time()
[newR5_nn, newG5_nn, newB5_nn, interpolatedIm5_nn] = nearestNeighbor(R5, G5, B5)
[newRtest_nn, newGtest_nn, newBtest_nn, interpolatedImTest_nn] = nearestNeighbor(Rtest, Gtest, Btest)
nnTime = end - start
print('Nearest neighbor interpolation: ', nnTime)
nnMSE = mse(imTest, interpolatedImTest_nn)
nnMAE = mae(imTest, interpolatedImTest_nn)

#%% Plot the results for interpolation using Nearest Neighbor
images_nn = [newR2_nn, newG2_nn, newB2_nn, interpolatedIm2_nn, \
           newR5_nn, newG5_nn, newB5_nn, interpolatedIm5_nn, \
           newRtest_nn, newGtest_nn, newBtest_nn, interpolatedImTest_nn]
names_nn = ["image2", "image5", "test image"]
plotting(images_nn, names_nn, "Nearest Neighbor Interpolation")
 
#%% METHOD 2: BILINEAR INTERPOLATION
start = time. time()
[newR2_bl, newG2_bl, newB2_bl, interpolatedIm2_bl] = bi(R2, G2, B2)
end = time. time()
[newR5_bl, newG5_bl, newB5_bl, interpolatedIm5_bl] = bi(R5, G5, B5)
[newRtest_bl, newGtest_bl, newBtest_bl, interpolatedImTest_bl] = bi(Rtest, Gtest, Btest)
bilinearTime = end - start
print('Bilinear interpolation: ', bilinearTime)    
bilinearMSE = mse(imTest, interpolatedImTest_bl)
bilinearMAE = mae(imTest, interpolatedImTest_bl)

#%% Plot the results for Bilinear Interpolation
images_bl = [newR2_bl, newG2_bl, newB2_bl, interpolatedIm2_bl, \
             newR5_bl, newG5_bl, newB5_bl, interpolatedIm5_bl,\
             newRtest_bl, newGtest_bl, newBtest_bl, interpolatedImTest_bl]
names_bl = ["image2", "image5", "test image"]
plotting(images_bl, names_bl, "Bilinear Interpolation")

#%% METHOD 3: PATTERNED PIXEL GROUPING INTERPOLATION
start = time. time()
[newR2_ppg, newG2_ppg, newB2_ppg, interpolatedIm2_ppg] = ppg(R2, G2, B2)
end = time. time()
[newR5_ppg, newG5_ppg, newB5_ppg, interpolatedIm5_ppg] = ppg(R5, G5, B5)
[interpolatedRtest_ppg, interpolatedGtest_ppg, interpolatedBtest_ppg, \
                         interpolatedImTest_ppg] = ppg(Rtest, Gtest, Btest)
ppgTime = end - start
print('PPG interpolation: ', ppgTime)
ppgMSE = mse(imTest, interpolatedImTest_ppg)
ppgMAE = mae(imTest, interpolatedImTest_ppg)
#%% Plot the results for PPG Interpolation
images_ppg = [newR2_ppg, newG2_ppg, newB2_ppg, interpolatedIm2_ppg, \
           newR5_ppg, newG5_ppg, newB5_ppg, interpolatedIm5_ppg,\
           interpolatedRtest_ppg, interpolatedGtest_ppg, interpolatedBtest_ppg, \
           interpolatedImTest_ppg]
names_ppg = ["image2", "image5", "test image"]
plotting(images_ppg, names_ppg, "PPG Interpolation")

#%% Performance of our Methods
plottingPerformance('MSE', ('Nearest Neighbor MSE','Bilinear MSE', 'PPG MSE'), \
                    [nnMSE, bilinearMSE, ppgMSE])
plottingPerformance('MAE', ('Nearest Neighbor MAE','Bilinear MAE', 'PPG MAE'), \
                    [nnMAE, bilinearMAE, ppgMAE])
plottingPerformance('Time', ('Nearest Neighbor Time','Bilinear Time', 'PPG Time'), \
                    [nnTime, bilinearTime, ppgTime])

#%% Plot the final results for visual comparison
plt.figure();
ax11 = plt.subplot(331)
plt.imshow(interpolatedIm2_nn); plt.title('Nearest Neighbor')
ax12 = plt.subplot(332,sharex=ax11,sharey=ax11)
plt.imshow(interpolatedIm2_bl); plt.title('Bilinear')
ax13 = plt.subplot(333,sharex=ax12,sharey=ax12)
plt.imshow(interpolatedIm2_ppg); plt.title('PPG')
ax21 = plt.subplot(334)
plt.imshow(interpolatedIm5_nn); 
ax22 = plt.subplot(335,sharex=ax21,sharey=ax21)
plt.imshow(interpolatedIm5_bl); 
ax23 = plt.subplot(336,sharex=ax22,sharey=ax22)
plt.imshow(interpolatedIm5_ppg); 
ax31 = plt.subplot(337)
plt.imshow(interpolatedImTest_nn); 
ax32 = plt.subplot(338,sharex=ax31,sharey=ax31)
plt.imshow(interpolatedImTest_bl); 
ax33 = plt.subplot(339,sharex=ax32,sharey=ax32)
plt.imshow(interpolatedImTest_ppg);