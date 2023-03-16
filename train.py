import os, json

import numpy as np

import transformers
from transformers import AutoTokenizer, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,Seq2SeqTrainer

from nltk.tokenize import sent_tokenize

import torch 

import evaluate

rouge_score = evaluate.load("rouge")


os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

from data.data import LECR_prepare_data
from model.model import give_model



params = {

    "load_tokenized_datasets":False,
    "save_dataset": True,

    "DEBUG" : False,
    "model_checkpoint":"google/flan-t5-large",
    "SEMANTIC_EMBEDING": "tools/semantic_embeding_v3_FlanEncoder_large_128.pkl",    # see tools/readme.txt
    "content_multiplier" : 10,
    "topic_multiplier" : 2,
    "topic_tree_multiplier":20,
    "topic_train_explode":False,
    "reduce_topics": False,
    
    "max_input_length": 128,
    "max_target_length":128,
    "generation_max_length":128,
    "output_dir": f"output/large_flan_v10_1_v2",

    "dropout_rate":0,
    "freeze_encoder":False,
    "num_decoder_layers":0,
    "freze_embed":False,
    "freze_layers": 0,  #["encoder.block.2","encoder.block.3","encoder.block.4","encoder.block.5","encoder.block.6","decoder.block.1","decoder.block.2","decoder.block.3","decoder.block.4"],   #0,    #["shared"]+["encoder.block."+x for x in map(str,list(range(8)))],  #["shared","encoder.block.0","encoder.block.1","block.3","block.4","block.5"]

    
    "eval_steps":0.05,
    "per_device_train_batch_size" : 16,  
    "per_device_eval_batch_size" : 64,
    "num_train_epochs" : 1,
    "gradient_accumulation_steps" : 8,
    "gradient_checkpointing":False,
    "learning_rate":2e-4,
    "sch" :"cosine",
    "cosine_warmup": 0.1,
    
    }



class paramss:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.params_dict = params 
            
params = paramss(**params)



def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}   

    return {k: round(v, 4) for k, v in result.items()}



def train(params):

    torch.backends.cuda.matmul.allow_tf32 = True
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(params.model_checkpoint)
    model = give_model(model_checkpoint=params.model_checkpoint, 
                       num_decoder_layers=params.num_decoder_layers,
                       dropout_rate=params.dropout_rate, 
                       freze_embed=params.freze_embed,
                       freeze_encoder=params.freeze_encoder, 
                       freze_layers=params.freze_layers) 
    
    dataset =  LECR_prepare_data(tokenizer = tokenizer, 
                             DEBUG = params.DEBUG ,
                             content_multiplier = params.content_multiplier, 
                             max_input_length = params.max_input_length,
                             max_target_length = params.max_target_length,
                             topic_multiplier = params.topic_multiplier, 
                             topic_tree_multiplier = params.topic_tree_multiplier, 
                             topic_train_explode = params.topic_train_explode,
                             load_tokenized_datasets=params.load_tokenized_datasets,
                             SEMANTIC_EMBEDING = params.SEMANTIC_EMBEDING,
                             save_dataset = params.save_dataset,
                             reduce_topics = params.reduce_topics
                             )
    tokenized_datasets = dataset.tokenized_Dataset
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    num_train_samples = len(tokenized_datasets["train"])

    number_of_batches_per_epoch = num_train_samples//params.per_device_train_batch_size
    number_of_steps_per_epoch = number_of_batches_per_epoch//params.gradient_accumulation_steps
    number_of_total_steps = number_of_steps_per_epoch*params.num_train_epochs

    optimizerr =  transformers.Adafactor(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=params.learning_rate,
                                         relative_step=False,
                                        #  weight_decay=0.01,
                                        #  beta1=0.9,
                                        #  clip_threshold=0.5
                                         )
    
    if params.sch == "one_cycle":
        lr_sch = torch.optim.lr_scheduler.OneCycleLR(optimizerr, params.learning_rate, total_steps=None, epochs=params.num_train_epochs, steps_per_epoch=number_of_steps_per_epoch, pct_start=0.3, 
                                        anneal_strategy='cos', cycle_momentum=False, base_momentum=0.85, max_momentum=0.95, div_factor=200.0, 
                                        final_div_factor=10000.0, three_phase=False, last_epoch=- 1, verbose=False)

    elif params.sch == "cosine":
        lr_sch = transformers.get_cosine_schedule_with_warmup(optimizer=optimizerr,
                                                          num_warmup_steps=int(params.cosine_warmup*number_of_total_steps),
                                                          num_training_steps=number_of_total_steps)
    
    optimizers = (optimizerr,lr_sch)

    model_name = params.model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(

        output_dir=params.output_dir,   
        evaluation_strategy="steps",
        eval_steps= int(params.eval_steps* number_of_steps_per_epoch) ,                   
        logging_steps=  int(params.eval_steps* number_of_steps_per_epoch) ,                
        save_steps=     int(params.eval_steps* number_of_steps_per_epoch)  ,               
        save_total_limit=1,
        per_device_train_batch_size=params.per_device_train_batch_size,  
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        # weight_decay=0.01,
        gradient_accumulation_steps=params.gradient_accumulation_steps,  
        num_train_epochs=params.num_train_epochs,
        predict_with_generate=True,
        gradient_checkpointing=params.gradient_checkpointing,
        # fp16=True, 
        bf16=True,
        tf32=True,
        generation_max_length=params.generation_max_length,
        report_to="tensorboard",
        dataloader_drop_last = True,
        dataloader_num_workers=32

    )
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset={"topic_valid":tokenized_datasets["topic_valid"],
                      "content_train_sample":tokenized_datasets["content_train_sample"], 
                      "topic_train_sample":tokenized_datasets["topic_train_sample"],
                      "topic_tree_train_sample":tokenized_datasets["topic_tree_train_sample"]},            #[tokenized_datasets["validation"],tokenized_datasets["mem_validation"]]
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers = optimizers
        
    )
    params_dict = params.params_dict
    with open(os.path.join(params_dict["output_dir"],"params.json"), "w") as write_file:
        json.dump(params_dict, write_file, indent=4)
    trainer.train()

    return model, dataset


if __name__ == "__main__":

    model, dataset = train(params)