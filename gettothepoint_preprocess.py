import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
from os.path import basename
import config

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
ABSTRACT_TYPE_INCORRECT = '[INCORR]'
ABSTRACT_TYPE_CORRECT = '[CORR]'

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "data/gettothepoint"
chunks_dir = os.path.join(finished_files_dir, "chunked")

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


def chunk_file(in_file, chunks_dir, set_name):

  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all(file_summary, datatype):
    bin_file = file_summary+'.bin'
    filename = os.path.splitext(basename(file_summary))[0]
    chunks_dir = os.path.join("data/gettothepoint", "chunked_"+datatype+filename)
  # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
  # Chunk the data

    chunk_file(bin_file, chunks_dir, filename)

    print "Saved chunked data in %s" % chunks_dir


def tokenize_file(file_to_tokenize):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print "Preparing to tokenize %s to %s..." % (file_to_tokenize, file_to_tokenize+'.tok')

  # make IO list file
  print "Making list of files to tokenize..."
  with open("mapping.txt", "w") as f:
      #write input output filenames to mapping.txt
      f.write("%s \t %s\n" % ( file_to_tokenize, file_to_tokenize+'.tok' ))

  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print "Tokenizing %s" % (file_to_tokenize)
  subprocess.call(command)
  print "Stanford CoreNLP Tokenizer has finished."
  os.remove("mapping.txt")


  print "Successfully finished tokenizing %s to %s.\n" % (file_to_tokenize, file_to_tokenize+'.tok')


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]



  return lines


def write_to_bin(inp_file1, inp_file2, token):
    #process input sentences
    articles = get_art_abs(inp_file1+'.tok')
    abstracts = get_art_abs(inp_file2+'.tok')


    print('Number of articles: ', len(articles))
    print('Number of abstracts: ', len(abstracts))

    #also write padded data for modified GTP
    data_padded = open(inp_file2+'_padded.bin', 'wb')
    #write bin files for each article/abstract
    with open(inp_file2+'.bin', 'wb') as writer:
      # Get the strings to write to .bin file
      for article, abstract in zip(articles, abstracts):

          # Write to tf.Example
          tf_example = example_pb2.Example()
          tf_example.features.feature['article'].bytes_list.value.extend([article])
          tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
          tf_example_str = tf_example.SerializeToString()
          str_len = len(tf_example_str)
          #print(tf_example_str)
          writer.write(struct.pack('q', str_len))
          writer.write(struct.pack('%ds' % str_len, tf_example_str))

          #for padded
          tf_example = example_pb2.Example()
          tf_example.features.feature['article'].bytes_list.value.extend([article])
          tf_example.features.feature['abstract'].bytes_list.value.extend([token+" "+abstract])
          tf_example_str = tf_example.SerializeToString()
          str_len = len(tf_example_str)

          data_padded.write(struct.pack('q', str_len))
          data_padded.write(struct.pack('%ds' % str_len, tf_example_str))
    data_padded.close()

    print "Finished writing file %s\n" % inp_file2




def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
  #if len(sys.argv) != 3:
#    print "USAGE: python make_datafiles.py"
#    sys.exit()

  print('STARTING processing SMCs')
  file_source = config.sourcefile
  #can be article or abstract
  file_summary = config.smcfile

  print('Tokenizing files')
  # Run stanford tokenizer on the input file
  tokenize_file(file_source)
  tokenize_file(file_summary)

  print('Postprocess Files')
  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(file_source, file_summary, ABSTRACT_TYPE_CORRECT)

  print('Make chunked files')
  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all(file_summary, "")
  chunk_all(file_summary+"_padded", "padded_")

  print("STARTING processing SMICs")
  file_source = config.sourcefile_filtered_gtp
  #can be article or abstract
  file_summary = config.smicfile_filtered_gtp

  print('Tokenizing files')
  # Run stanford tokenizer on the input file
  tokenize_file(file_source)
  tokenize_file(file_summary)

  print('Postprocess Files')
  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(file_source, file_summary, ABSTRACT_TYPE_INCORRECT)

  print('Make chunked files')
  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all(file_summary, "")
  chunk_all(file_summary+"_padded", "padded_")
