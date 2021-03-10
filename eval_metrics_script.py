"""
Called like so:
python eval_metrics_script.py --generated_views=/filepath/generated_views_dtu_6 --ground_truth_views=/filepath/gt_views_dtu_6
"""
import os
import glob
from absl import app
from absl import flags

import imageio
import numpy as np

import eval_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('generated_views', '', 'Directory to generated views.')
flags.DEFINE_string('ground_truth_views', '', 'Directory to ground truth views.')

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  lpips_fn = eval_utils.load_lpips()
  mse_fn = lambda x, y: np.mean((x - y)**2)
  psnr_fn = lambda x, y: -10 * np.log10(mse_fn(x, y))
  def ComputeMetrics(generated, gt):
    ssim_score = eval_utils.ssim(generated, gt)
    float32_gen = (generated / 255.).astype(np.float32)
    float32_gt = (gt / 255.).astype(np.float32)
    lpips_score = lpips_fn(float32_gen,
                          float32_gt)
    psnr_score = psnr_fn(float32_gen, float32_gt)
    return ssim_score, psnr_score, lpips_score

  images_to_eval = glob.glob(os.path.join(FLAGS.generated_views, "*.png"))
  files = [os.path.basename(s) for s in images_to_eval]
  ssim = []
  psnr = []
  lpips = []

  for k in files:
    try:
      gt_im = imageio.imread(os.path.join(FLAGS.ground_truth_views, k))
      gv_im = imageio.imread(os.path.join(FLAGS.generated_views, k))
    except Exception as e:
      print("I/O Error opening filename: %s" % k)

    ssim_score, psnr_score, lpips_score = ComputeMetrics(gt_im, gv_im)
    ssim.append(ssim_score)
    psnr.append(psnr_score)
    lpips.append(lpips_score)
  print("PSNR:")
  print("Mean: %04f" % np.mean(psnr))
  print("Stddev: %04f" % np.std(psnr))
  print()
  print("SSIM:")
  print("Mean: %04f" % np.mean(ssim))
  print("Stddev: %04f" % np.std(ssim))
  print()
  print("LPIPS:")
  print("Mean: %04f" % np.mean(lpips))
  print("Stddev: %04f" % np.std(lpips))

if __name__ == '__main__':
  app.run(main)
