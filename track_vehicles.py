## Import
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage.measurements import label
import time
import glob
from sklearn import svm, grid_search
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid 
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from moviepy.editor import VideoFileClip

# Class to identify, store, and draw tracked vehicles
class VehTracker():
    def __init__(self,img_RGB_in):
        
        # FRAME
        frame_height = img_RGB_in.shape[0]
        frame_width = img_RGB_in.shape[1]
        self.Frame = img_RGB_in
        # Binary image for current frame
        self.img_BIN_in = None
        # RGB image for output of current frame
        self.img_RGB_out = img_RGB_in
        # Current number of consecutive failed frames
        #self.num_failed_frame_curr = 0
        # Number of frames processed
        self.frame_num = 0
        
        # HEATMAP
        self.heatmap = np.zeros_like(img_RGB_in[:,:,0]).astype(np.float)
        self.heatmap_list = []
        
        # SVM
        self.SVM = None
        self.X_scaler = None
        
        # HYPERPARAMETERS
        self.ystart = 400
        self.ystop_list = (450, 490, 500, 525, 550, 600, 650)
        #self.scale_list = (0.8, 1.0, 1.5, 3.0)
        self.scale_list = (1.0, 1.0, 1.5, 1.5, 2.0, 3.0, 3.5)
        self.orient = 10
        self.pix_per_cell = 16
        self.cell_per_block = 2
        self.cells_per_step = 1  # Instead of overlap, define how many cells to step         
        self.window = 64 # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        self.spatial_size = (16, 16)
        self.hist_bins = 32
        self.heat_threshold = 25
        self.heatmap_flt_size = 5
        #self.svm_params = {'kernel':['linear'], 'C':[0.1, 1, 10, 100]}
        self.svm_params = {'kernel':['linear'], 'C':[0.1]}
        
        return
    
    def convert_color(self, img, conv='RGB2YUV'):
        '''
        Convert given image from RGB to desired color space.
        '''
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        if conv == 'RGB2YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        if conv == 'RGB2HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, 
                            vis=False, feature_vec=True):
        '''
        Extract HOG features from given image.
        '''
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, 
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      block_norm= 'L2-Hys',
                                      transform_sqrt=False, 
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(img, orientations=orient, 
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           block_norm= 'L2-Hys',
                           transform_sqrt=False, 
                           visualise=vis, feature_vector=feature_vec)
            return features

    def bin_spatial(self, img, size=(32, 32)):
        '''
        Bin the given image spatially.
        '''
        color1 = cv2.resize(img[:,:,0], size).ravel()
        color2 = cv2.resize(img[:,:,1], size).ravel()
        color3 = cv2.resize(img[:,:,2], size).ravel()
        out = np.hstack((color1, color2, color3))     
        return out
                        
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):    
        '''
        Return a feature vector of a color histogram on given image.
        '''        
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        #hist_features = np.concatenate((channel1_hist[0], channel2_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
            
        return hist_features
    
    def extract_features(self, imgs, cspace='YUV', spatial_size=(32, 32),
                            hist_bins=32, hist_range=(0, 256)):
        '''
        Define a function to extract features from a list of images. Used for SVM training.
        '''
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            else: feature_image = np.copy(image)      
            # Apply bin_spatial() to get spatial color features
            #spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            #print("Shape of spatial histogram features: " + str(spatial_features.shape))
            # Apply color_hist() also with a color space option now
            #hist_features = self.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            #print("Shape of color histogram features: " + str(hist_features.shape))
            # Calculate HOG
            hog1 = self.get_hog_features(feature_image[:,:,0], self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=True)
            hog2 = self.get_hog_features(feature_image[:,:,1], self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=True)
            hog3 = self.get_hog_features(feature_image[:,:,2], self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=True)
            hog_features = np.hstack((hog1, hog2, hog3))
            #hog_features = np.hstack((hog1, hog2))
            #print("Shape of HOG features: " + str(hog_features.shape))
            # Append the new feature vector to the features list
            #features.append(np.hstack((spatial_features, hist_features, hog_features)))
            #features.append(np.hstack((hist_features, hog_features)))
            features.append(hog_features)
            #print("Shape of total features: " + str(features[0].shape))
        # Return list of feature vectors
        return features
        
    def train_svm(self):
        '''
        Train an SVM with the given paths for training data
        '''
        # Read in car and non-car images
        images = glob.glob('training_data_non_vehicle\*.png')
        notcars = []
        cnt = 0
        for image in images:
            notcars.append(image)
#             cnt = cnt + 1
#             if (cnt == 5000):
#                 break;
        
        images = glob.glob('training_data_vehicle\*.png')
        cars = []
        cnt = 0
        for image in images:
            cars.append(image)
#             cnt = cnt + 1
#             if (cnt == 5000):
#                 break;
        
        print('Number of car images: ' + str(len(cars)))
        print('Number of non-car images: ' + str(len(notcars)))
        
        car_features = self.extract_features(cars, cspace='YUV', spatial_size=self.spatial_size,
                                hist_bins=self.hist_bins, hist_range=(0, 256))
        notcar_features = self.extract_features(notcars, cspace='YUV', spatial_size=self.spatial_size,
                                hist_bins=self.hist_bins, hist_range=(0, 256))
        
        print('Features extracted.')
        
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)
        
        print('Data randomized.')
            
        # Fit a per-column scaler only on the training data
        self.X_scaler = StandardScaler().fit(X_train)
        pickle.dump( self.X_scaler, open( "X_scaler.p", "wb" ))
        # Apply the scaler to X_train and X_test
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test) 
        
        print('Data normalized.')
        
        # Create and train SVM
        # Check the training time for the SVM
        t=time.time()
        svr = svm.SVC()
        print("Starting grid search...")
        self.SVM = grid_search.GridSearchCV(svr, self.svm_params)
        print("Finished gird search. Starting fit...")
        self.SVM.fit(X_train, y_train)
        t2 = time.time()
        pickle.dump( self.SVM, open( "SVM.p", "wb" ))
        print(round(t2-t, 2), 'Seconds to train SVM...')
        print('Best parameters: ' + str(self.SVM.best_params_))
        # Check the score of the SVM
        print('Test Accuracy of SVM = ', round(self.SVM.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        print('My SVM predicts: ', self.SVM.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVM')

        return
    
    def load_SVM(self):
        '''
        Load SVM from pickled data.
        '''
        
        print("Loading pickled SVM")
        self.SVM = pickle.load( open( "SVM.p", "rb" ) )
        self.X_scaler = pickle.load( open( "X_scaler.p", "rb"))
                
        return

    def find_cars(self, img):
        '''
        Extract features using spatial binning, color histogram, and hog sub-sampling and make predictions.
        '''
        bbox_list = []
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
                
        it = 0
        for scale in self.scale_list:
            ystop = self.ystop_list[it]            
            it = it + 1
            
            img_tosearch = img[self.ystart:ystop,:,:]
            ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2YUV')
            #print('Scale: ' + str(scale))
            if scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
                #img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
                
            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]
        
            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
            nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
            nfeat_per_block = self.orient*self.cell_per_block**2
            
            nblocks_per_window = (self.window // self.pix_per_cell) - self.cell_per_block + 1
            nxsteps = (nxblocks - nblocks_per_window) // self.cells_per_step + 1
            nysteps = (nyblocks - nblocks_per_window) // self.cells_per_step + 1
            
            # Compute individual channel HOG features for the entire image
            hog1 = self.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog2 = self.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog3 = self.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*self.cells_per_step
                    xpos = xb*self.cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    #hog_features = np.hstack((hog_feat1, hog_feat2))
        
                    xleft = xpos*self.pix_per_cell
                    ytop = ypos*self.pix_per_cell
        
                    # Extract the image patch
                    #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+self.window, xleft:xleft+self.window], (64,64))
                    #subimg = cv2.resize(img_tosearch[ytop:ytop+self.window, xleft:xleft+self.window], (64,64))
                                      
                    # Get color features
                    #spatial_features = self.bin_spatial(subimg, size=self.spatial_size)
                    #hist_features = self.color_hist(subimg, nbins=self.hist_bins)
        
                    # Scale features and make a prediction
                    #test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)) 
                    #test_features = self.X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))  
                    test_features = self.X_scaler.transform(hog_features.reshape(1, -1))  
                    test_prediction = self.SVM.predict(test_features)
                    
                    if test_prediction == 1:
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(self.window*scale)
                        bbox = ((xbox_left, ytop_draw+self.ystart),(xbox_left+win_draw,ytop_draw+win_draw+self.ystart))
                        bbox_list.append(bbox)
                        cv2.rectangle(draw_img,(xbox_left, ytop_draw+self.ystart),(xbox_left+win_draw,ytop_draw+win_draw+self.ystart),(0,0,255),6) 
               
            #print('bbox size: ' + str(len(bbox_list)))
                        
        return draw_img, bbox_list
    
    def add_heat(self, bbox_list):
        '''
        Loop through bounding box list and add "heat" to heatmap.
        '''
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        return
    
    def apply_threshold(self, thresh):
        '''
        Update heat with 0 for every location with insufficient heat.
        '''
        # Zero out pixels below the threshold
        self.heatmap[self.heatmap < thresh] = 0
        return
    
    def flt_heatmap(self):
        '''
        Perform rolling average filter on heatmaps
        '''
        # Init output
        out_heatmap = np.zeros_like(self.Frame[:,:,0]).astype(np.float)
        
        # Remove first element of list if reached filter size
        if (len(self.heatmap_list) == self.heatmap_flt_size):
            self.heatmap_list.pop(0)
         
        # Add newest heatmap   
        self.heatmap_list.append(self.heatmap)            
            
        # Calculate sum of heatmap list
        for heatmap in self.heatmap_list:
            out_heatmap = np.add(out_heatmap, heatmap)
            
        #out_heatmap = np.divide(out_heatmap,len(self.heatmap_list))
        
        return out_heatmap
    
    def draw_labeled_bboxes(self, img, labels):
        '''
        Identify and draw bounding box around each car that has been identified
        '''
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img
    
    def process_frame(self, new_frame):
        '''
        Process given frame.
        '''
        
        # Store new frame
        self.Frame = new_frame
        
        # Reset heat map
        self.heatmap = np.zeros_like(self.Frame[:,:,0]).astype(np.float)
        
        out_img, bbox_list = self.find_cars(new_frame)
        
        #plt.imshow(out_img)
        
        # Add heat to each box in box list
        self.add_heat(bbox_list)
            
        # Apply threshold to help remove false positives
        #self.apply_threshold(self.heat_threshold)
        
        # Clip heat map to [0,255]    
        #self.heatmap = np.clip(self.heatmap, 0, 255) 
        
        # Filter list of heatmaps
        self.heatmap = self.flt_heatmap()
        
        # Apply threshold for filtered heatmap
        self.apply_threshold(self.heat_threshold)
        
        # Clip heat map to [0,255]    
        self.heatmap = np.clip(self.heatmap, 0, 255)        
        # Find final boxes from heatmap using label function
        labels = label(self.heatmap)
        out_frame = self.draw_labeled_bboxes(np.copy(new_frame), labels)
        
        return out_frame
    
    def viz_pipeline(self):
        '''
        Visualize pipeline on sample images
        '''
        # Retreive sample images to plot
        car_file = 'training_data_vehicle/7.png'
        not_car_file = 'training_data_non_vehicle/extra40.png'
        img_car = mpimg.imread(car_file)
        img_not_car = mpimg.imread(not_car_file)
        img_frame_1 = mpimg.imread('test_images/test1.jpg')
        img_frame_2 = mpimg.imread('test_images/test2.jpg')
        img_frame_3 = mpimg.imread('test_images/test3.jpg')
        img_frame_4 = mpimg.imread('test_images/test4.jpg')
        img_frame_5 = mpimg.imread('test_images/test5.jpg')
        img_frame_6 = mpimg.imread('test_images/test6.jpg')
        
        # Plot original images
        fig = plt.figure(0)
        plt.suptitle('Original Images')
        plt.subplot(1,2,1)
        plt.imshow(img_car)
        plt.title('Car')
        plt.subplot(1,2,2)
        plt.imshow(img_not_car)
        plt.title('Not Car')
        fig.tight_layout()
        plt.savefig('output_images/originals.png')        
        #plt.show()
        
        # Plot color transformed images
        fig = plt.figure(1)
        plt.suptitle('YUV Transformed Images')
        plt.subplot(1,2,1)
        img_car_color_xfrm = self.convert_color(img_car)
        plt.imshow(img_car_color_xfrm)
        plt.title('Car')
        plt.subplot(1,2,2)
        img_not_car_color_xfrm = self.convert_color(img_not_car)
        plt.imshow(img_not_car_color_xfrm)
        plt.title('Not Car')
        fig.tight_layout()
        plt.savefig('output_images/YUV_xfrm.png')
        #plt.show()
        
        # Plot spatial histograms
        fig = plt.figure(2)
        plt.suptitle('Spatial Histogram Features')
        plt.subplot(1,2,1)
        plt.plot(self.bin_spatial(img_car_color_xfrm))
        plt.title('Car')
        plt.subplot(1,2,2)
        plt.plot(self.bin_spatial(img_not_car_color_xfrm))
        plt.title('Not Car')
        fig.tight_layout()
        plt.savefig('output_images/spatial_histogram.png')
        #plt.show() 
        
        # Plot color histograms
        fig = plt.figure(3)
        plt.suptitle('Color Histogram Features')
        plt.subplot(1,2,1)
        plt.plot(self.color_hist(img_car_color_xfrm))
        plt.title('Car')
        plt.subplot(1,2,2)
        plt.plot(self.color_hist(img_not_car_color_xfrm))
        plt.title('Not Car')
        fig.tight_layout()
        plt.savefig('output_images/color_histogram.png')
        #plt.show()   
        
        # Plot HOG visualization
        fig = plt.figure(4)
        plt.suptitle('HOG')
        hog1, img_hog_car_1 = self.get_hog_features(img_car_color_xfrm[:,:,0], self.orient, self.pix_per_cell, self.cell_per_block, vis=True, feature_vec=True)
        hog2, img_hog_car_2 = self.get_hog_features(img_car_color_xfrm[:,:,1], self.orient, self.pix_per_cell, self.cell_per_block, vis=True, feature_vec=True)
        hog3, img_hog_car_3 = self.get_hog_features(img_car_color_xfrm[:,:,2], self.orient, self.pix_per_cell, self.cell_per_block, vis=True, feature_vec=True)
        hog_car_features = np.hstack((hog1, hog2, hog3))
        hog1, img_hog_not_car_1 = self.get_hog_features(img_not_car_color_xfrm[:,:,0], self.orient, self.pix_per_cell, self.cell_per_block, vis=True, feature_vec=True)
        hog2, img_hog_not_car_2 = self.get_hog_features(img_not_car_color_xfrm[:,:,1], self.orient, self.pix_per_cell, self.cell_per_block, vis=True, feature_vec=True)
        hog3, img_hog_not_car_3 = self.get_hog_features(img_not_car_color_xfrm[:,:,2], self.orient, self.pix_per_cell, self.cell_per_block, vis=True, feature_vec=True)
        hog_not_car_features = np.hstack((hog1, hog2, hog3))
        plt.subplot(3,2,1)
        plt.imshow(img_hog_car_1)
        plt.title('Car - Ch1')
        plt.subplot(3,2,3)
        plt.imshow(img_hog_car_2)
        plt.title('Car - Ch2')
        plt.subplot(3,2,5)
        plt.imshow(img_hog_car_3)
        plt.title('Car - Ch3')
        plt.subplot(3,2,2)
        plt.imshow(img_hog_not_car_1)
        plt.title('Not Car - Ch1')
        plt.subplot(3,2,4)
        plt.imshow(img_hog_not_car_2)
        plt.title('Not Car - Ch2')
        plt.subplot(3,2,6)
        plt.imshow(img_hog_not_car_3)
        plt.title('Not Car - Ch3')
        fig.tight_layout()
        plt.savefig('output_images/hog_viz.png')
        #plt.show()
        
        # Plot HOG features
        fig = plt.figure(5)
        plt.suptitle('HOG Features')
        plt.subplot(1,2,1)
        plt.plot(hog_car_features)
        plt.title('Car')
        plt.subplot(1,2,2)
        plt.plot(hog_not_car_features)
        plt.title('Not Car')
        fig.tight_layout()
        plt.savefig('output_images/hog_features.png')
        #plt.show()
        
        # Plot total raw features
        files = []
        files.append(car_file)
        car_features = self.extract_features(files, cspace='YUV', spatial_size=self.spatial_size,
                                hist_bins=self.hist_bins, hist_range=(0, 256))
        files = []
        files.append(not_car_file)
        notcar_features = self.extract_features(files, cspace='YUV', spatial_size=self.spatial_size,
                                hist_bins=self.hist_bins, hist_range=(0, 256))
        fig = plt.figure(6)
        plt.suptitle('Total Raw Features')
        plt.subplot(1,2,1)
        plot_car_arr = np.array(car_features).flatten()
        plt.plot(plot_car_arr)
        plt.title('Car')
        plt.subplot(1,2,2)
        plot_not_car_arr = np.array(notcar_features).flatten()
        plt.plot(plot_not_car_arr)
        plt.title('Not Car')
        fig.tight_layout()
        plt.savefig('output_images/tot_raw_features.png')
        
        # Plot total features
        fig = plt.figure(7)
        plt.suptitle('Total Scaled Features')
        plt.subplot(1,2,1)
        scaled_car_feats = self.X_scaler.transform(car_features)
        plt.plot(np.array(scaled_car_feats).flatten())
        plt.title('Car')
        plt.subplot(1,2,2)
        scaled_not_car_feats = self.X_scaler.transform(notcar_features)
        plt.plot(np.array(scaled_not_car_feats).flatten())
        plt.title('Not Car')
        fig.tight_layout()
        plt.savefig('output_images/tot_scaled_features.png')
        
        # Test vehicle detection on single frame
        fig = plt.figure(8)
        plt.suptitle('Raw Detection')
        plt.subplot(3,2,1)
        veh_dtct_frame, bbox_list = self.find_cars(img_frame_1)
        plt.imshow(veh_dtct_frame)
        plt.subplot(3,2,2)
        veh_dtct_frame, bbox_list = self.find_cars(img_frame_2)
        plt.imshow(veh_dtct_frame)
        plt.subplot(3,2,3)
        veh_dtct_frame, bbox_list = self.find_cars(img_frame_3)
        plt.imshow(veh_dtct_frame)
        plt.subplot(3,2,4)
        veh_dtct_frame, bbox_list = self.find_cars(img_frame_4)
        plt.imshow(veh_dtct_frame)
        plt.subplot(3,2,5)
        veh_dtct_frame, bbox_list = self.find_cars(img_frame_5)
        plt.imshow(veh_dtct_frame)
        plt.subplot(3,2,6)
        veh_dtct_frame, bbox_list = self.find_cars(img_frame_6)
        plt.imshow(veh_dtct_frame)
        plt.savefig('output_images/raw_detection.png')
        
        # Visualize heatmap
        fig = plt.figure(9)
        plt.suptitle('Heat Map')
        plt.subplot(3,2,1)
        veh_dtct_frame_1 = self.process_frame(img_frame_1)
        self.heatmap_list = []
        plt.imshow(self.heatmap, cmap='hot')
        plt.subplot(3,2,2)
        veh_dtct_frame_2 = self.process_frame(img_frame_2)
        self.heatmap_list = []
        plt.imshow(self.heatmap, cmap='hot')
        plt.subplot(3,2,3)
        veh_dtct_frame_3 = self.process_frame(img_frame_3)
        self.heatmap_list = []
        plt.imshow(self.heatmap, cmap='hot')
        plt.subplot(3,2,4)
        veh_dtct_frame_4 = self.process_frame(img_frame_4)
        self.heatmap_list = []
        plt.imshow(self.heatmap, cmap='hot')
        plt.subplot(3,2,5)
        veh_dtct_frame_5 = self.process_frame(img_frame_5)
        self.heatmap_list = []
        plt.imshow(self.heatmap, cmap='hot')
        plt.subplot(3,2,6)
        veh_dtct_frame_6 = self.process_frame(img_frame_6)
        self.heatmap_list = []
        plt.imshow(self.heatmap, cmap='hot')
        plt.savefig('output_images/heat_maps.png')
        
        # Visualize final detection
        fig = plt.figure(10)
        plt.suptitle('Final Detection')
        plt.subplot(3,2,1)
        plt.imshow(veh_dtct_frame_1)
        plt.subplot(3,2,2)
        plt.imshow(veh_dtct_frame_2)
        plt.subplot(3,2,3)
        plt.imshow(veh_dtct_frame_3)
        plt.subplot(3,2,4)
        plt.imshow(veh_dtct_frame_4)
        plt.subplot(3,2,5)
        plt.imshow(veh_dtct_frame_5)
        plt.subplot(3,2,6)
        plt.imshow(veh_dtct_frame_6)
        plt.savefig('output_images/final_detection.png')
        
        # Show Plots
        plt.show()
    
## Create and Train SVM
# Instnatiate with test image
img = mpimg.imread('test_images/test5.jpg')
vehicle_tracker = VehTracker(img)
train_svm = False
if (train_svm):
    vehicle_tracker.train_svm()
else:
    vehicle_tracker.load_SVM()

viz = False
if (viz):
    vehicle_tracker.viz_pipeline()

## Process video
if (True):
    video_output  = 'output_videos/project_video_processed.mp4'
    #video_output  = 'output_videos/project_video_processed.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip = VideoFileClip("project_video.mp4")
    video_clip = clip.fl_image(vehicle_tracker.process_frame) #NOTE: this function expects color images!!
    video_clip.write_videofile(video_output, audio=False)
