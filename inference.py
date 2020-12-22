from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import modeling
import numpy as np
import tensorflow.compat.v1 as tf


def gather_indexes(sequence_tensor, positions):
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def get_mlm_logits(input_tensor, albert_config, mlm_positions, output_weights):
  input_tensor = gather_indexes(input_tensor, mlm_positions)
  with tf.variable_scope("cls/predictions"):
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=albert_config.embedding_size,
          activation=modeling.get_activation(albert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    output_bias = tf.get_variable(
        "output_bias",
        shape=[albert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(
        input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
  return logits


def get_sentence_order_logits(input_tensor, albert_config):
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, albert_config.hidden_size],
        initializer=modeling.create_initializer(
            albert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    return logits


def build_model(topk, albert_config_path, checkpoint_path):
  """Module function."""
  input_ids = tf.placeholder(tf.int32, [None, None], "input_ids")
  input_mask = tf.placeholder(tf.int32, [None, None], "input_mask")
  segment_ids = tf.placeholder(tf.int32, [None, None], "segment_ids")
  mlm_positions = tf.placeholder(tf.int32, [None, None], "mlm_positions")

  albert_config = modeling.AlbertConfig.from_json_file(albert_config_path)
  model = modeling.AlbertModel(
      config=albert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)

  mlm_logits = get_mlm_logits(model.get_sequence_output(), albert_config,
                 mlm_positions, model.get_embedding_table())
  nsp_logits = get_sentence_order_logits(model.get_pooled_output(), albert_config)

  mlm_scores = tf.nn.softmax(mlm_logits)
  nsp_scores = tf.nn.softmax(nsp_logits)

  mlm_topk_scores, mlm_topk_indices = tf.math.top_k(mlm_scores, k=topk)
  nsp_predictions = nsp_scores[:, 0]

  tvars = tf.trainable_variables()
  (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
          tvars, checkpoint_path)

  tf.logging.info("**** Trainable Variables ****")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)
  tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
  init = tf.global_variables_initializer()
  return init, (mlm_topk_scores, mlm_topk_indices), nsp_predictions


def create_single_example(tokenizer, sentence, sentence_next=None, max_seq_length=-1):
  mask_index = tokenizer.vocab["[MASK]"]
  if max_seq_length <= 0:
    max_seq_length = 2 + len(sentence)
    if sentence_next is not None:
      max_seq_length += 1 + len(sentence_next)

  input_ids = np.zeros([1, max_seq_length], dtype=np.int64)
  input_mask = np.zeros([1, max_seq_length], dtype=np.int64)
  segment_ids = np.ones([1, max_seq_length], dtype=np.int64)

  sentence = ["[CLS]"] + list(sentence) + ["[SEP]"]
  segment_ids[0, :len(sentence)] = 0

  if sentence_next is not None:
    sentence += (list(sentence_next) + ["[SEP]"])

  input_ids[0, :len(sentence)] = tokenizer.convert_tokens_to_ids(sentence)
  input_mask[0, :len(sentence)] = 1
  segment_ids[0, len(sentence):] = 0
  mlm_positions = np.nonzero(input_ids[0] == mask_index)[0][None]
  return {
    "input_ids:0": input_ids,
    "input_mask:0": input_mask,
    "segment_ids:0": segment_ids,
    "mlm_positions:0": mlm_positions
  }


if __name__ == "__main__":
  import os
  import sys
  from tokenization import FullTokenizer

  model_dir = sys.argv[1]
  top_k = int(sys.argv[2])
  
  tokenizer = FullTokenizer(os.path.join(model_dir, "vocab_chinese.txt"))

  init, (mlm_scores, mlm_indices), nsp_predictions = build_model(
    top_k, 
    os.path.join(model_dir, "albert_config.json"), 
    os.path.join(model_dir, "model.ckpt-best")
  )

  usage = """
  !exit 退出程序（注意以感叹号开头）
  !help 显示此帮助

  > 自然语#处#。&一种人##能技术。
  + 自然语#处#。&一种人##能技术。
  第一个句子： 自然语[MASK]处[MASK]。
  第二个句子： 一种人[MASK][MASK]能技术。
  输入句子相关联的概率: 0.233
  第1个MASK位置填充预测及概率：
  1       言      0.60
  2       的      0.11
  3       法      0.03
  第2个MASK位置填充预测及概率：
  1       理      0.91
  2       境      0.02
  3       学      0.00
  第3个MASK位置填充预测及概率：
  1       工      0.62
  2       体      0.11
  3       际      0.06
  第4个MASK位置填充预测及概率：
  1       智      0.85
  2       技      0.03
  3       全      0.02"""

  with tf.Session() as sess:
    sess.run(init)
    print(usage)
    while True:
      print("> ", end="")
      try:
        user_inputs = input()
      except KeyboardInterrupt:
        break
      except Exception as e:
        tf.logging.error(e)
        continue
      else:
        if user_inputs == "":
          continue
        print("+", user_inputs)

      if user_inputs.startswith("!"):
        cmd = user_inputs[1:].upper()
        if cmd == "EXIT":
          break
        elif cmd == "HELP":
          print(usage)
        else:
          with os.popen(cmd) as f:
            for line in f:
              print(line, end="")
        continue
      
      user_inputs = user_inputs.split("&")
      _replace_mask = lambda x: [token if token != "#" else "[MASK]" for token in tokenizer.tokenize(x)]

      sentence = _replace_mask(user_inputs[0])
      print("第一个句子：", "".join(sentence))
      if len(user_inputs) > 1:
        sentence_next = _replace_mask(user_inputs[1])
        print("第二个句子：", "".join(sentence_next))
      else:
        sentence_next = None
      example = create_single_example(tokenizer, sentence, sentence_next)
      scores, indices, nsp_pred = sess.run([mlm_scores, mlm_indices, nsp_predictions], feed_dict=example)

      if sentence_next is not None:
        print("输入句子相关联的概率: {:.3f}".format(nsp_pred[0]))
      for i, (_scores, _indices) in enumerate(zip(scores, indices)):
        print("第{}个MASK位置填充预测及概率：".format(i + 1))
        for j, (score, index) in enumerate(zip(_scores, _indices)):
            print("{}\t{}\t{:.2f}".format(j + 1, tokenizer.inv_vocab[index], score))
