
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# =========================
# CONFIG
# =========================
MODEL_NAME = "google/flan-t5-small"
DATA_PATH = "slm/dataset.csv"
SAVE_DIR = "models/slm_model"


# =========================
# LOAD CSV DATASET
# =========================
def load_dataset():

    print("Loading CSV dataset...")

    df = pd.read_csv(DATA_PATH)

    # remove empty rows
    df = df.dropna()

    # keep only required columns
    df = df[["input", "output"]]

    # convert to huggingface dataset
    dataset = Dataset.from_pandas(df)

    return dataset


# =========================
# TRAIN / VALID SPLIT
# =========================
def split_dataset(dataset):

    dataset = dataset.train_test_split(test_size=0.1)

    train_data = dataset["train"]
    val_data = dataset["test"]

    print("Train samples:", len(train_data))
    print("Validation samples:", len(val_data))

    return train_data, val_data


# =========================
# TOKENIZATION
# =========================
def tokenize_data(dataset, tokenizer):

    def tokenize(example):

        inputs = tokenizer(
            example["input"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

        labels = tokenizer(
            example["output"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

        inputs["labels"] = labels["input_ids"]

        return inputs

    return dataset.map(tokenize, remove_columns=dataset.column_names)


# =========================
# TRAIN FUNCTION
# =========================
def train():

    os.makedirs(SAVE_DIR, exist_ok=True)

    # load dataset
    dataset = load_dataset()

    # split dataset
    train_data, val_data = split_dataset(dataset)

    # load tokenizer + model
    print("Loading model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # tokenize
    print("Tokenizing dataset...")
    train_data = tokenize_data(train_data, tokenizer)
    val_data = tokenize_data(val_data, tokenizer)

    # data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # training arguments
    training_args = TrainingArguments(
    output_dir="slm_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2,
    save_steps=500,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50
    )
    # training_args = TrainingArguments(
    #     output_dir="slm_training",
    #     learning_rate=3e-5,
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=4,
    #     num_train_epochs=5,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=20,
    #     save_strategy="epoch",
    #     evaluation_strategy="epoch",
    #     save_total_limit=2,
    #     predict_with_generate=True,
    #     fp16=False,  # set True if GPU supports it
    #     report_to="none"
    # )


   




    # trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data
    )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator
    # )

    # start training
    print("\nStarting training...")
    trainer.train()

    # save final model
    print("\nSaving model...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("\nTraining complete!")
    print("Model saved at:", SAVE_DIR)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()




# import os
# import pandas as pd
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForSeq2Seq
# )

# # =========================
# # CONFIG
# # =========================
# MODEL_NAME = "google/flan-t5-small"
# DATA_PATH = "slm/dataset.csv"        # CSV file with 'input' and 'output' columns
# SAVE_DIR = "models/slm_model"

# # =========================
# # LOAD CSV DATASET
# # =========================
# def load_dataset():
#     print("Loading CSV dataset...")
#     df = pd.read_csv(DATA_PATH)
#     df = df.dropna()

#     if "input" not in df.columns or "output" not in df.columns:
#         raise ValueError("CSV must have columns 'input' and 'output'.")

#     dataset = Dataset.from_pandas(df[["input", "output"]])
#     return dataset

# # =========================
# # TRAIN / VALID SPLIT
# # =========================
# def split_dataset(dataset):
#     dataset = dataset.train_test_split(test_size=0.1)
#     train_data = dataset["train"]
#     val_data = dataset["test"]
#     print("Train samples:", len(train_data))
#     print("Validation samples:", len(val_data))
#     return train_data, val_data

# # =========================
# # TOKENIZATION
# # =========================
# def tokenize_data(dataset, tokenizer):
#     def tokenize(example):
#         inputs = tokenizer(
#             example["input"],
#             padding="max_length",
#             truncation=True,
#             max_length=128
#         )
#         labels = tokenizer(
#             example["output"],
#             padding="max_length",
#             truncation=True,
#             max_length=128
#         )
#         inputs["labels"] = labels["input_ids"]
#         return inputs
#     return dataset.map(tokenize, remove_columns=dataset.column_names)

# # =========================
# # TRAIN FUNCTION
# # =========================
# def train():
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     # Load dataset
#     dataset = load_dataset()
#     train_data, val_data = split_dataset(dataset)

#     # Load model + tokenizer
#     print("Loading model + tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

#     # Tokenize datasets
#     print("Tokenizing datasets...")
#     train_data = tokenize_data(train_data, tokenizer)
#     val_data = tokenize_data(val_data, tokenizer)

#     # Data collator
#     data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir=SAVE_DIR,
#         per_device_train_batch_size=4,
#         per_device_eval_batch_size=4,
#         num_train_epochs=3,
#         logging_dir="./logs",
#         logging_steps=50,
#         save_steps=500,
#         save_total_limit=2,
#         learning_rate=5e-5,
#         evaluation_strategy="steps",
#         eval_steps=50
#     )

#     # Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_data,
#         eval_dataset=val_data,
#         tokenizer=tokenizer,
#         data_collator=data_collator
#     )

#     # Train
#     print("\nStarting training...")
#     trainer.train()

#     # Save final model
#     print("\nSaving model...")
#     model.save_pretrained(SAVE_DIR)
#     tokenizer.save_pretrained(SAVE_DIR)
#     print("Training complete! Model saved at:", SAVE_DIR)

# # =========================
# # MAIN
# # =========================
# if __name__ == "__main__":
#     train()