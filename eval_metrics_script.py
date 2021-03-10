from absl import app
from  absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('generated_views', '', 'Directory to generated views.')
flags.DEFINE_string('ground_truth_views', '', 'Directory to ground truth views.')

def main():
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # recursively searches generated_veiws



if __name__ == '__main__':
  app.run(main)
