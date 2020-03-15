import sys

# Find a way around this ugliness.
sys.path += ['C:/Users/sayka/Desktop/Course/Projects/sarcasm-detection/bert_repo']

import tensorflow as tf
import run_classifier
import run_classifier_with_tfhub
from preprocess import load_data, standard_loader
import datetime
import os

#   uncased_L-12_H-768_A-12: uncased BERT base model
#   uncased_L-24_H-1024_A-16: uncased BERT large model
#   cased_L-12_H-768_A-12: cased BERT large model
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500

cfg = {
    'model_ver': 'v1',
    'BERT_MODEL': 'uncased_L-12_H-768_A-12',
    'NUM_TRAIN_EPOCHS': 10.0,
}


class SarcasmBertBasic:
    def __init__(self, params):
        self.params = params
        self.estimator = None
        self.label_list = ['NOT_SARCASM', 'SARCASM']
        self.OUTPUT_DIR = 'bert_out_files/{}'.format(params['model_ver'])

        os.environ['TFHUB_CACHE_DIR'] = self.OUTPUT_DIR
        tf.gfile.MakeDirs(self.OUTPUT_DIR)
        print('***** Model output directory: {} *****'.format(self.OUTPUT_DIR))
        self.BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_' + params['BERT_MODEL'] + '/1'
        self.tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module(self.BERT_MODEL_HUB)

    def fit(self, train_data):
        params = self.params

        # Compute number of train and warmup steps from batch size
        num_train_steps = int(len(train_data) / TRAIN_BATCH_SIZE * params['NUM_TRAIN_EPOCHS'])
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

        model_fn = run_classifier_with_tfhub.model_fn_builder(
            num_labels=len(self.label_list),
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False,
            bert_hub_module_handle=self.BERT_MODEL_HUB
        )
        TPU_ADDRESS = 'grpc://10.114.88.226:8470'  # Arbitrary. We want a GPU
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
        NUM_TPU_CORES = 8
        ITERATIONS_PER_LOOP = 1000

        def get_run_config(output_dir):
            return tf.contrib.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                model_dir=output_dir,
                save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
                tpu_config=tf.contrib.tpu.TPUConfig(
                    iterations_per_loop=ITERATIONS_PER_LOOP,
                    num_shards=NUM_TPU_CORES,
                    per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

        # This should revert to GPU when TPU is set to false per bert doc.
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=get_run_config(self.OUTPUT_DIR),
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            predict_batch_size=PREDICT_BATCH_SIZE,
        )

        train_features = run_classifier.convert_examples_to_features(
            train_data, self.label_list, MAX_SEQ_LENGTH, self.tokenizer)
        print('***** Started training at {} *****'.format(datetime.datetime.now()))
        print('  Num examples = {}'.format(len(train_data)))
        print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = run_classifier.input_fn_builder(
            features=train_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=True)

        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print('***** Finished training at {} *****'.format(datetime.datetime.now()))

    def predict(self, pred_data):
        input_features = run_classifier.convert_examples_to_features(
            pred_data, self.label_list, MAX_SEQ_LENGTH, self.tokenizer)
        predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH,
                                                           is_training=False, drop_remainder=True)
        predictions = self.estimator.predict(predict_input_fn)

        return predictions


if __name__ == "__main__":
    # Example Training Run
    train, test = load_data(load_fn=standard_loader('both'), context_extent='all', split=0.2)
    model = SarcasmBertBasic(cfg)
    model.fit(train)
    preds = model.predict(test)
    print(preds)
