import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

BATCH_SIZE = 16
BLOCK_SIZE = 256
#BATCH_SIZE = 16
#BLOCK_SIZE = 1024


def main():
    model_name = "facebook/opt-125m"
    data_file = "data/sample-1gb.txt"
    output_dir = "trained_model_bs_16_block_256"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    raw_datasets = load_dataset("text", data_files=data_file)
    raw_datasets["validation"] = load_dataset("text",
                                              data_files=data_file,
                                              split=f"train[:5%]",
                                              )
    raw_datasets["train"] = load_dataset("text",
                                         data_files=data_file,
                                         split=f"train[5%:]",
                                         )
    tokenized_dataset = raw_datasets.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length",
                                                             max_length=BLOCK_SIZE),
                                         batched=True,
                                         num_proc=4,
                                         remove_columns=["text"])
    training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=3, per_device_train_batch_size=BATCH_SIZE,
                                      per_device_eval_batch_size=BATCH_SIZE, evaluation_strategy="epoch", save_strategy="epoch",
                                      logging_strategy="epoch", learning_rate=2e-5, weight_decay=0.01,
                                      fp16=True
                                      )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"],
                      eval_dataset=tokenized_dataset["validation"], data_collator=data_collator)
    trainer.train()
    eval_results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
    print("Perplexidade = ", perplexity.item())
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    main()