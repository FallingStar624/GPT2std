from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, 
    Trainer, TrainingArguments, set_seed)
from datasets import load_dataset, Dataset
from torchinfo import summary
import glob
import argparse


if __name__ == '__main__':
    set_seed(22554)
    num_sentences = 500000  # 800만 이상 50*4 vs 200

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--idx", dest='idx', action='store') #
    parser.add_argument('-r', "--resume", dest='resume', action='store_true') # [Todo]shell #ckpt
    args = parser.parse_args()

    # argument check (model idx, resume)
    if not args.idx:
        print("-- check argument : [--idx] not defined")
        exit()
    current_model_idx = int(args.idx)
    print(f"-- current model idx : {str(current_model_idx)}")

    # load dataset
    start_point = current_model_idx*num_sentences
    end_point   = start_point + num_sentences
    dataset = load_dataset("openwebtext",
        split = f'train[{start_point}:{end_point}]') # number of sentences
    print(f"-- current dataset : from {start_point} to {end_point}")
    print(f"--check train data\n>>{dataset['text'][0][:200]}...\n")
    val_set = load_dataset("openwebtext", split='train[8000000:]')
    print(f"--check validation data\n>>{val_set['text'][0][:200]}...\n") 
   
    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = 'right'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # block size (max len) & batch size
    block_size = 512

    # tokenize text 
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=block_size)

    train_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    # train_dataset = tokenized_datasets["text"]

    valid_dataset = val_set.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    # valid_dataset = tokenized_val_datasets["text"]

    # model setting
    config = GPT2Config.from_pretrained("gpt2", output_hidden_states=True)
    
    config.n_embd  = 384 # small model experiment
    config.n_layer = 6
    config.n_head  = 12
    
    per_device_batch_size = 8
    
    model = GPT2LMHeadModel(config=config) # [Todo] change attn hidden

    # data collator and training arguments
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=f'./results_{current_model_idx}',  # output directory
        num_train_epochs=30,  # total number of training epochs 
        per_device_train_batch_size=per_device_batch_size,  # batch size per device during training
        save_steps=500,  # number of training steps between checkpoints
        save_total_limit=2,  # limit the total amount of checkpoints saved
        logging_steps=100,  # number of training steps between logging information to tensorboard
        logging_dir='./logs',  # directory for storing logs
        logging_first_step=True,  # log also the very first training step
        overwrite_output_dir=True,  # overwrite the content of the output directory
        learning_rate=5e-5,
        dataloader_num_workers=4, ###>>>>
        gradient_accumulation_steps=1, ###>>>>
    )
 
    # Trainer
    trainer = Trainer(
        model=model,  # the instantiated GPT-2 model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset, # validation dataset
        data_collator=data_collator,  # data collator
        tokenizer=tokenizer,  # tokenizer used for pre-processing
        # callbacks=[TensorBoardCallback()],  # TensorBoard callback for logging
    )

    # train
    if (args.resume):
        last_result_folder = sorted(glob.glob(f'results_{current_model_idx}/checkpoint*'))[0]
        print(f"-- resume training : {last_result_folder}")
        trainer.train(last_result_folder)
        trainer.save_model("./final_result_{current_model_idx}")
    else:
        trainer.train()