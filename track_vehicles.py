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
        self.heat = np.zeros_like(img_RGB_in[:,:,0]).astype(np.float)
        self.heatmap = None
        
        # SVM
        self.SVM = None
        
        # HYPERPARAMETERS
        self.ystart = 400
        self.ystop = 656
        self.scale = 1.5
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.spatial_size = (32, 32)
        self.hist_bins = 32
        self.heat_threshold = 1
        self.svm_params = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        
        return
    
    def convert_color(self, img, conv='RGB2YCrCb'):
        '''
        Convert given image from RGB to desired color space.
        '''
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

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
        return np.hstack((color1, color2, color3))
                        
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
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
    
    def extract_features(self, imgs, cspace='RGB', spatial_size=(32, 32),
                            hist_bins=32, hist_range=(0, 256)):
        '''
        Define a function to extract features from a list of images
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
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            # Apply color_hist() also with a color space option now
            hist_features = self.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_features, hist_features)))
        # Return list of feature vectors
        return features

    def find_cars(self, img, svc, X_scaler):
        '''
        Extract features using hog sub-sampling and make predictions.
        '''
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        
        img_tosearch = img[self.ystart:self.ystop,:,:]
        ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2YCrCb')
        if self.scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/self.scale), np.int(imshape[0]/self.scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient*self.cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        
        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = self.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = self.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
    
                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell
    
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                spatial_features = self.bin_spatial(subimg, size=self.spatial_size)
                hist_features = self.color_hist(subimg, nbins=self.hist_bins)
    
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*self.scale)
                    ytop_draw = np.int(ytop*self.scale)
                    win_draw = np.int(window*self.scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+self.ystart),(xbox_left+win_draw,ytop_draw+win_draw+self.ystart),(0,0,255),6) 
                    
        return draw_img
    
    def train_svm(self):
        '''
        Train an SVM with the given paths for training data
        '''
        # Read in car and non-car images
        images = glob.glob('training_data_non_vehicle\*.png')
        notcars = []
        for image in images:
            notcars.append(image)
        
        images = glob.glob('training_data_vehicle\*.png')
        cars = []
        for image in images:
            cars.append(image)
        
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
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X_train and X_test
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test) 
        
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
        
        # Save in pickle
        pickle.dump( self.SVM, open( "SVM_YUV.p", "wb" ) )

        return
    
    def load_SVM(self):
        '''
        Load SVM from pickled data.
        '''
        
        print("Loading pickled SVM")
        self.SVM = pickle.load( open( "SVM_YUV.p", "rb" ) )
                
        return
    
    def add_heat(self, bbox_list):
        '''
        Loop through bounding box list and add "heat" for entire box.
        '''
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        return
    
    def apply_threshold(self):
        '''
        Update heat with 0 for every location with insufficient heat.
        '''
        # Zero out pixels below the threshold
        self.heat[self.heat <= self.heat_threshold] = 0
        # Return thresholded map
        return
    
    def draw_labeled_bboxes(self, img, labels):
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
        
        out_img = find_cars(img, svc, X_scaler)
        
        plt.imshow(out_img)
        
        # Add heat to each box in box list
        self.add_heat(box_list)
            
        # Apply threshold to help remove false positives
        self.apply_threshold()
        
        # Visualize the heatmap when displaying    
        self.heatmap = np.clip(self.heat, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(self.heatmap)
        draw_img = draw_labeled_bboxes(np.copy(new_frame), labels)
        
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        
        return
    
## Create and Train SVM
img = mpimg.imread('test_images/test6.jpg')
vehicle_tracker = VehTracker(img)
#vehicle_tracker.train_svm()
vehicle_tracker.load_SVM()


## Test Pipeline


## Process video
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    lane_lines = LaneLines(img,img)
    video_output  = 'output_videos/challenge_video_processed.mp4'
    #video_output  = 'output_videos/project_video_processed.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip = VideoFileClip("test_videos/challenge_video.mp4")
    video_clip = clip.fl_image(lane_lines.draw_frame) #NOTE: this function expects color images!!
    video_clip.write_videofile(video_output, audio=False)
