r"""Assorted function for use when computing metrics and evals."""
import collections
import os

import numpy as np
import scipy
from scipy import signal
from scipy.ndimage.filters import convolve
import tensorflow.compat.v1 as tf


def _FSpecialGauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  radius = size // 2
  offset = 0.0
  start, stop = -radius, radius + 1
  if size % 2 == 0:
    offset = 0.5
    stop -= 1
  x, y = np.mgrid[offset + start:stop, offset + start:stop]
  assert len(x) == size
  g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
  return g / g.sum()

def fspecial_gauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  radius = size // 2
  offset = 0.0
  start, stop = -radius, radius + 1
  if size % 2 == 0:
    offset = 0.5
    stop -= 1
  x, y = np.mgrid[offset + start:stop, offset + start:stop]
  assert len(x) == size
  g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
  return g / g.sum()

def ssim(img1, img2, max_val=255, filter_size=11,
         filter_sigma=1.5, k1=0.01, k2=0.03, mask=None):
  """Original code here: https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/compression/image_encoder/msssim.py
  Return the Structural Similarity Map between `img1` and `img2`.
  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
  Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError("Input images must have the same shape (%s vs. %s).",
                       img1.shape, img2.shape)
  if img1.ndim == 3:
    img1 = np.expand_dims(img1, 0)

  if img2.ndim == 3:
    img2 = np.expand_dims(img2, 0)

  if img1.ndim != 4:
    raise RuntimeError(
        "Input images must have four dimensions, not %d", img1.ndim)

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  _, height, width, _ = img1.shape

  # Filter size can't be larger than height or width of images.
  size = min(filter_size, height, width)

  # Scale down sigma if a smaller filter size is used.
  sigma = size * filter_sigma / filter_size if filter_size else 0

  if filter_size:
    window = np.reshape(fspecial_gauss(size, sigma), (1, size, size, 1))
    mu1 = signal.fftconvolve(img1, window, mode="same")
    mu2 = signal.fftconvolve(img2, window, mode="same")
    sigma11 = signal.fftconvolve(img1 * img1, window, mode="same")
    sigma22 = signal.fftconvolve(img2 * img2, window, mode="same")
    sigma12 = signal.fftconvolve(img1 * img2, window, mode="same")
  else:
    # Empty blur kernel so no need to convolve.
    mu1, mu2 = img1, img2
    sigma11 = img1 * img1
    sigma22 = img2 * img2
    sigma12 = img1 * img2

  mu11 = mu1 * mu1
  mu22 = mu2 * mu2
  mu12 = mu1 * mu2
  sigma11 -= mu11
  sigma22 -= mu22
  sigma12 -= mu12

  # Calculate intermediate values used by both ssim and cs_map.
  c1 = (k1 * max_val) ** 2
  c2 = (k2 * max_val) ** 2
  v1 = 2.0 * sigma12 + c2
  v2 = sigma11 + sigma22 + c2
  if mask is not None:
    score = (((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2))
    score = np.sum(mask * score) / (np.sum(mask*np.ones_like(score)))
  else:
    score = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
  # cs = np.mean(v1 / v2)
  return score



def load_lpips():
  """Return a function to compute the LPIPS distance between two images.

  Returns:
    distance: a function that takes two images [H, W, C] scaled from 0 to 1, and
    returns the LPIPS distance between them.
  """
  graph = tf.compat.v1.Graph()
  session = tf.compat.v1.Session(graph=graph)
  with graph.as_default():
    input1 = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
    input2 = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
    with tf.gfile.Open('alex_net.pb', 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      # Required order for network is [B, C, H, W].
      target = tf.transpose((input1[tf.newaxis] * 2.0) - 1.0, [0, 3, 1, 2])
      pred = tf.transpose((input2[tf.newaxis] * 2.0) - 1.0, [0, 3, 1, 2])
      tf.import_graph_def(
          graph_def, input_map={'0:0':target, '1:0':pred})
      distance = graph.get_operations()[-1].outputs[0]

  def lpips_distance(img1, img2):
    with graph.as_default():
      return session.run(distance, {input1:img1, input2:img2})[0, 0, 0, 0]
  return lpips_distance


