import sys

# Find a way around this ugliness.
if 'bert_repo' not in sys.path:
    sys.path += ['bert_repo']


import datetime
import os
import tensorflow as tf
import run_classifier
import run_classifier_with_tfhub
from preprocess import load_data, standard_loader
import optimization
import tensorflow_hub as hub

#   uncased_L-12_H-768_A-12: uncased BERT base model
#   uncased_L-24_H-1024_A-16: uncased BERT large model
#   cased_L-12_H-768_A-12: cased BERT large model
TRAIN_BATCH_SIZE = 6  # 32
EVAL_BATCH_SIZE = 6  # 8
PREDICT_BATCH_SIZE = 6  # 8
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512  # 128
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500

cfg = {
    'model_ver': 'v1',
    'BERT_MODEL': 'uncased_L-12_H-768_A-12',
    'NUM_TRAIN_EPOCHS': 1.7,
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

        run_config = tf.estimator.RunConfig(
            model_dir=self.OUTPUT_DIR,
            save_summary_steps=SAVE_SUMMARY_STEPS,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

        model_fn = self.model_fn_builder(
            num_labels=len(self.label_list),
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)

        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=self.OUTPUT_DIR,
            config=run_config,
            params={"batch_size": TRAIN_BATCH_SIZE})

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
            drop_remainder=False)

        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print('***** Finished training at {} *****'.format(datetime.datetime.now()))

    def predict(self, pred_data):
        input_features = run_classifier.convert_examples_to_features(
            pred_data, self.label_list, MAX_SEQ_LENGTH, self.tokenizer)
        predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH,
                                                           is_training=False, drop_remainder=False)
        predictions = self.estimator.predict(predict_input_fn)

        return predictions

    def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
        """Creates a classification model."""

        bert_module = hub.Module(
            self.BERT_MODEL_HUB,
            trainable=True)
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)
        bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        output_layer = bert_outputs["pooled_output"]

        hidden_size = output_layer.shape[-1].value

        # Create our own layer to tune for politeness data.
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            # Dropout helps prevent overfitting
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return (predicted_labels, log_probs)

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return loss, predicted_labels, log_probs

    def model_fn_builder(self, num_labels, learning_rate, num_train_steps, num_warmup_steps):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

            # TRAIN and EVAL
            if not is_predicting:

                (loss, predicted_labels, log_probs) = self.create_model(
                    is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                train_op = optimization.create_optimizer(
                    loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

                # Calculate evaluation metrics.
                def metric_fn(lab_ids, pred_labels):
                    accuracy = tf.metrics.accuracy(lab_ids, pred_labels)
                    f1_score = tf.contrib.metrics.f1_score(
                        lab_ids,
                        pred_labels)
                    recall = tf.metrics.recall(
                        lab_ids,
                        pred_labels)
                    precision = tf.metrics.precision(
                        lab_ids,
                        pred_labels)

                    return {
                        "eval_accuracy": accuracy,
                        "f1_score": f1_score,
                        "precision": precision,
                        "recall": recall,
                    }

                eval_metrics = metric_fn(label_ids, predicted_labels)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      train_op=train_op)
                else:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      eval_metric_ops=eval_metrics)
            else:
                (predicted_labels, log_probs) = self.create_model(
                    is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                predictions = {
                    'probabilities': log_probs,
                    'labels': predicted_labels
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Return the actual model function in the closure
        return model_fn


if __name__ == "__main__":
    # Example Training Run
    train, test = load_data(load_fn=standard_loader('both'), context_extent='all', split=0.2)
    model = SarcasmBertBasic(cfg)
    model.fit(train)
    preds = model.predict(test)
    print(preds)
