"""
Fine-tune a distilBERT model
on the Amazon Fine Foods dataset.
"""

import os

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import create_optimizer
from transformers.keras_callbacks import KerasMetricCallback


og_model_name = "distilbert-base-uncased"
model_name = "distilbert-uncased-finefoods"
batch_size = 12


def main():

    # get dataset
    dataset = load_dataset(
        "csv",
        data_files={"train": "train.csv", "eval": "test.csv"},
    )

    # load a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(og_model_name)
    # define a tokenizer that
    # - takes our desired text column (review/all_text)
    # - turncates the documents, so that maximal length is 512 tokens
    #   (which is the size of distilbert's input layer)
    def tokenize(data):
        return tokenizer(data["review/all_text"], truncation=True, padding="longest")

    # run tokenization
    dataset_tokenized = dataset.map(tokenize, batched=True)

    # convert encoded dataset to tf format
    tf_train = dataset_tokenized["train"].to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["label"],
        shuffle=True,
        batch_size=16,
    )

    tf_val = dataset_tokenized["eval"].to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["label"],
        shuffle=False,
        batch_size=16,
    )

    # laod a pretrained model
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        og_model_name, num_labels=2
    )

    # optimizer
    num_epochs = 5
    batches_per_epoch = len(dataset_tokenized["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)

    optimizer, schedule = create_optimizer(
        init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps
    )

    model.compile(optimizer=optimizer, loss=loss)

    # evaluation callback
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        preds, labs = eval_pred
        preds = preds[:, 0]
        score = metric.compute(predictions=preds, references=labs)
        return score

    metric_callback = KerasMetricCallback(
        metric_fn=compute_metrics, eval_dataset=tf_val
    )

    # fit model
    tensorboard_callback = TensorBoard(log_dir=f"./{model_name}/logs")

    callbacks = [metric_callback, tensorboard_callback]

    model.fit(
        tf_train,
        validation_data=tf_val,
        epochs=1,
        callbacks=callbacks,
    )

    model.save(f"./{model_name}/")


if __name__ == "__main__":
    main()
