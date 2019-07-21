import matplotlib.pyplot as plt
import numpy as np
import argparse
from sys import exit
from pathlib import Path
from skimage.feature import ORB
from skimage.transform import warp
from skimage.measure import ransac
from skimage.color import rgb2gray
from skimage.io import ImageCollection
from skimage.io import imsave
from skimage.color import gray2rgb
from skimage.feature import match_descriptors
from skimage.transform import SimilarityTransform
from skimage.transform import ProjectiveTransform

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
args = vars(ap.parse_args())


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

        self.keypoints, self.descriptors, self.corners = [
        ], [], np.empty((self.num_imgs, 4, 2))
        orb = ORB(n_keypoints=1000, fast_threshold=0.05)

        for idx, img in enumerate(self.grays):
            # Extract interest points and their features
            orb.detect_and_extract(img)
            self.keypoints.append(orb.keypoints)
            self.descriptors.append(orb.descriptors)

            # Get 4 corners of images
            r, c = img.shape
            self.corners[idx] = np.array([[0, 0], [0, r], [c, 0], [c, r]])

    def match_features(self):
        self.tforms = [ProjectiveTransform()]
        self.new_corners = np.copy(self.corners)

        for i in range(1, self.num_imgs):
            # Find correspondences between I(n) and I(n-1).
            matches = match_descriptors(
                self.descriptors[i-1], self.descriptors[i], cross_check=True)

            # Estimate the transformation between I(n) and I(n-1).
            src = self.keypoints[i][matches[:, 1]][:, ::-1]
            dst = self.keypoints[i-1][matches[:, 0]][:, ::-1]

            model, _ = ransac((src, dst), ProjectiveTransform,
                              4, residual_threshold=2, max_trials=2000)
            self.tforms.append(ProjectiveTransform(
                model.params @ self.tforms[-1].params))

            # Compute new corners transformed by models
            self.new_corners[i] = self.tforms[-1](self.corners[i])

        corners_min = np.min(self.new_corners, axis=1)
        corners_max = np.max(self.new_corners, axis=1)

        self.xLim = corners_max[:, 0] - corners_min[:, 0]
        self.yLim = corners_max[:, 1] - corners_min[:, 1]

    def adjust_center(self):
        '''Inverting the transform for the center image and applying that transform to all the others to create a nicer panorama.'''

        # Find center image, assume that the scene is always horizontal
        xCenterIdx = np.argsort(self.xLim)[self.num_imgs//2]
        centerTform = np.copy(self.tforms[xCenterIdx].params)
        invCenterTform = np.linalg.inv(centerTform)

        for i in range(self.num_imgs):
            self.tforms[i].params = self.tforms[i].params @ invCenterTform
            self.new_corners[i] = self.tforms[i](self.corners[i])

        # Recompute image corners after adjust center.
        corners_min = np.min(self.new_corners, axis=1)
        corners_max = np.max(self.new_corners, axis=1)

        self.corner_min = np.min(corners_min, axis=0)
        self.corner_max = np.max(corners_max, axis=0)

        self.output_shape = self.corner_max - self.corner_min
        self.output_shape = np.ceil(self.output_shape[::-1]).astype(int)

    def stitch(self):
        pano_warped, pano_mask = [], []
        # Translate images to right position so it can be display entirely.
        offset = SimilarityTransform(translation=-self.corner_min)

        for i in range(self.num_imgs):
            # Apply offset to all transformations
            self.tforms[i] += offset
            # Apply transformation on images
            pano_warped.append(warp(self.images[i], self.tforms[i].inverse, order=0,
                                    output_shape=self.output_shape, cval=-1))

            # Find mask to remove overlap regions
            # Mask == 1 inside image
            pano_mask.append((pano_warped[-1] != -1)*1)

            # Remove overlap region from previous mask
            if i > 0:
                overlap = pano_mask[-1] + pano_mask[-2]
                pano_mask[-2] = np.where(overlap > 1, 0, pano_mask[-2])

        # Stitch them together
        self.panorama = np.zeros_like(pano_warped[0])
        for i in range(self.num_imgs):
            self.panorama += pano_warped[i] * gray2rgb(pano_mask[i])
        
        # Normalize
        self.panorama[self.panorama > 1] = 1
        self.panorama[self.panorama < 0] = 0

    def save(self):
        imsave(Path(self.out).joinpath('panorama_out.png'), self.panorama)

    def auto_stitch(self):
        self.extract_features()
        self.match_features()
        self.adjust_center()
        self.stitch()
        self.save()


if __name__ == "__main__":
    p = Panorama(args['input'], args['output'])
    p.auto_stitch()