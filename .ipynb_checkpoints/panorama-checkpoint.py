import matplotlib.pyplot as plt
import numpy as np
from sys import exit
from pathlib import Path
from skimage.feature import ORB
from skimage.transform import warp
from skimage.measure import ransac
from skimage.color import rgb2gray
from skimage.io import ImageCollection
from skimage.feature import match_descriptors
from skimage.transform import ProjectiveTransform


class Panorama:
    '''Automatic panorama

    Parameters
    ----------
    in_dir : str
        Directory stores images to create panorama.
    out_dir : str
        Directory that save output panorama.
    '''
    
    def __init__(self, in_dir, out_dir):
        self.extension = ['jpg', 'png', 'JPG', 'PNG']
        
        if not Path(out_dir).is_dir():
            exit("[ERROR] Output directory doesn't exist.")

        self.out = out_dir
        self.load(in_dir)

    def load(self, in_dir):
        '''Load images in in_dir'''

        # Load images
        if not Path(in_dir).is_dir():
            exit("[ERROR] Input directory doesn't exist.")

        p = [f'{str(Path(in_dir))}/*.{ex}' for ex in self.extension]
        p = ':'.join(p)
        self.images = ImageCollection(p)
        self.num_imgs = len(self.images)

        if self.num_imgs < 2:
            exit("[ERROR] No images.")

        # convert images to gray scale
        self.grays = [rgb2gray(img) for img in self.images]

    def extract_features(self):
        '''Extract interest points and their feature vector (descriptors)'''

        self.keypoints, self.descriptors, self.corners = [], [], np.empty((self.num_imgs, 4, 2))
        orb = ORB(n_keypoints=1000, fast_threshold=0.05)

        for idx, img in enumerate(self.grays):
            orb.detect_and_extract(img)
            self.keypoints.append(orb.keypoints)
            self.descriptors.append(orb.descriptors)

            # get 4 corners of images
            r, c = img.shape
            self.corners[idx] = np.array([[0, 0], [0, r], [c, 0], [c, r]])


    def match_features(self):
        self.tforms = [ProjectiveTransform()]
        self.new_corners = np.copy(self.corners)

        for i in range(1, self.num_imgs):
            # Find correspondences between I(n) and I(n-1).
            matches = match_descriptors(self.descriptors[i-1], self.descriptors[i], cross_check=True)

            # Estimate the transformation between I(n) and I(n-1).
            src = self.keypoints[i][matches[:, 1]][:, ::-1]
            dst = self.keypoints[i-1][matches[:, 0]][:, ::-1]

            model, _ = ransac((src, dst), ProjectiveTransform, 4, residual_threshold=2, max_trials = 2000)
            self.tforms.append(ProjectiveTransform(model.params @ self.tforms[-1].params))

            # Inverting the transform for the center image and applying that transform to all the others to create a nicer panorama.
            self.new_corners[i] = self.tforms[-1](self.corners[i])

        print(self.new_corners)

        corner_min = np.min(self.new_corners, axis=1)
        corner_max = np.max(self.new_corners, axis=1)
        # xlim = np.array([corner_min[:, 0], corner_max[:, 0]])
        print("corners min", corner_min)
        print("corners max", corner_max)
        
        self.output_shape = np.max(corner_max, axis=0) - np.min(corner_min, axis=0)
        self.output_shape = np.ceil(self.output_shape[::-1]).astype(int)
        print("shape", self.output_shape)

    def stitch(self):
        pano_warped, pano_mask = [], []
        for i in range(self.num_imgs):
            pano_warped.append(warp(self.grays[i], self.tforms[i].inverse, order=0,
                                    output_shape=self.output_shape, cval=-1))

            pano_mask.append((pano_warped[-1] != -1))  # Mask == 1 inside image
            pano_warped[-1][~pano_mask[-1]] = 0      # Return background values to 0

        merged = sum(pano_warped)
        overlap = sum(pano_mask)*1.0
        self.panorama = merged / np.maximum(overlap, 1)



path = '/home/datami/Dev/UIT/CS231-ComputerVision/panorama/pano/'
a = Panorama(path, path)
a.extract_features()
a.match_features()
a.stitch()

fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(a.panorama, cmap='gray')
plt.tight_layout()
ax.axis('off')
plt.show()