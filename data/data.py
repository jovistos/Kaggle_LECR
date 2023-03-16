import re
import pandas as pd
import numpy as np
import pickle
import shutil


from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk




class LECR_prepare_data:
    def __init__(self, tokenizer = None, 
                 DEBUG = False, 
                 content_multiplier = 28, 
                 topic_multiplier = 1,
                 topic_tree_multiplier=14,
                 max_input_length = 100 , 
                 max_target_length=50, 
                 load_tokenized_datasets=False,
                 save_dataset = True,
                 topic_train_explode=False,
                 TOKENIZED_DATASETS_PATH = "/home/jovis/Documents/WORK/Kaggle/LECR/LECR/datasets_tokenized",
                 SEMANTIC_EMBEDING = "/home/jovis/Documents/WORK/Kaggle/LECR/Neural-Corpus-Indexer-NCI/Data_process/NQ_dataset/kmeans/semantic_embeding_v3_FlanEncoder_base.pkl",
                 DATA_PATH = "/home/jovis/Documents/WORK/Kaggle/LECR/data/learning-equality-curriculum-recommendations/",
                 reduce_topics = True
                     ):

        self.TOKENIZED_DATASETS_PATH = TOKENIZED_DATASETS_PATH
        self.DATA_PATH = DATA_PATH
        self.SEMANTIC_EMBEDING = SEMANTIC_EMBEDING
        self.topics = pd.read_csv(self.DATA_PATH + "topics_v2.csv")
        self.content = pd.read_csv(self.DATA_PATH + "content.csv")
        self.correlations = pd.read_csv(self.DATA_PATH + "correlations.csv")
        self.DEBUG = DEBUG
        self.content_multiplier = content_multiplier
        self.topic_train_explode = topic_train_explode
        self.topic_multiplier = topic_multiplier
        self.topic_tree_multiplier = topic_tree_multiplier
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.load_tokenized_datasets = load_tokenized_datasets
        self.save_dateset = save_dataset
        self.reduce_topics = reduce_topics
        self.load_data()
        

    def load_data(self):

        self.preprocess()
        print("preprocess_done")
        self.split()
        print("split_done")
        self.tokenize_data()
        print("tokenize_done")
        self.concat_data()
        self.print_counts()
        print("###############################################################################################")
    
    @staticmethod
    def pickle_dump(obj_, filename):
        with open(f'{filename}', 'wb') as f:
            pickle.dump(obj_, f)

    @staticmethod
    def pickle_load(path_):
        with open(f'{path_}', 'rb') as f:
            obj_ = pickle.load(f)
        return obj_
    
    @staticmethod
    def add_zero_infront(word):
        if len(word)==1:
            word = "0"+word
        return word

    @staticmethod
    def ids_to_semantic(ids,df_semantic_ready):
        semantic_ids = [df_semantic_ready[y] for y in ids.split()]
        semantic_ids = [x for x in semantic_ids if x not in ["Nan"]] 
        semantic_ids.sort()
        return " ".join(semantic_ids)

    
    def preprocess_function(self,examples):

        model_inputs = self.tokenizer(
            examples["source"],
            max_length=self.max_input_length,
            truncation=True,
            padding="longest",  #max_length  longest
            # return_attention_mask=True,
            # add_special_tokens=True,
            return_tensors = "pt"
        )
        labels = self.tokenizer(
            examples["semantic_target"], 
            max_length=self.max_target_length, 
            truncation=True,
            padding="longest",   # max_length   semantic_target
            # return_attention_mask=True,
            # add_special_tokens=True,
            return_tensors = "pt"
        )
        lbls = labels["input_ids"]
        lbls[lbls==0] = -100
        model_inputs["labels"] = lbls 
        model_inputs["labels_mask"] = labels["attention_mask"]
        return model_inputs

    def clean_text(self,text):
        
        text = re.sub(r"-", " ", text)
        text = re.sub(r"\|", " ", text)

        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        return text
    
    def preprocess(self):

        content = self.content
        topics = self.topics

        content = content.add_prefix('content_')
        content = content.replace(np.nan, 'NaN')
        # topics = topics.add_prefix('topic_')

        topics = topics.replace(np.nan, 'NaN')
        if self.reduce_topics==True:
            topics = topics[(topics.topic_category=="supplemental")|(topics.topic_category=="aligned")].reset_index(drop=True)
        print("number of topics",len(topics))
        topics["source"] = " %1 " + topics.topic_level.astype(str) + " %2 "+topics.topic_language + " %3 "+ topics.topic_channel+ " %4 "+topics.topic_category+ " %5 "+ topics.topic_title + " %6 "+ topics.topic_description

        
        self.topic_tree_df = topics[["source","topic_tree_id"]].copy(deep=True)
        self.topic_tree_df["source"]= "%C" + self.topic_tree_df.source
        self.topic_tree_df["semantic_target"]= self.topic_tree_df["topic_tree_id"]


        topics["source"]= "%A" + topics.source
        df_1 = pd.merge(topics,self.correlations,  on='topic_id', how='left')   #[["topic_title","topic_description","topic_id"]]
        df_1 = df_1[df_1['content_ids'].notna()]
        print("number of correlation samples",len(df_1))

        # df_1['topic_title'] = df_1['topic_title'].replace(np.nan, "NaN")
        # df_1['topic_description'] = df_1['topic_description'].replace(np.nan, "NaN")

        # df_1_ready = pd.DataFrame()
        # df_1_ready["source"] ="((A:)) " + " ((topic_title)) "+ df_1.topic_title + " ((topic_description)) "+ df_1.topic_description
        # df_1_ready["source"] = df_1_ready["source"].apply(lambda x: self.clean_text(x))
        df_1["target"] = df_1["content_ids"] 


        # content['content_title'] = content['content_title'].replace(np.nan, "NaN")
        # content['content_description'] = content['content_description'].replace(np.nan, "NaN")
        # content['content_text'] = content['content_text'].replace(np.nan, "NaN")


        # content = pd.DataFrame()
        # content["source"]= "((B:)) " + " ((content_title)) "+ content.content_title + " ((content_description)) "+ content.content_description+ " ((content_text)) "+ content.content_text
        content["source"] ="%B" + " %7 " + content.content_kind + " %8 "+content.content_language + " %9 "+ content.content_title+ " %10 "+content.content_description+ " %11 "+ content.content_text 
        content["source"] = content["source"].apply(lambda x: self.clean_text(x))

        content["target"] = content["content_id"] 



        df_semantic = self.pickle_load(self.SEMANTIC_EMBEDING)


        df_semantic_ready = {}
        for i in range(len(content)):
            target = content.target[i]
            try:
                semantic_target = df_semantic[target]
                df_semantic_ready[target] = "-".join([self.add_zero_infront(str(y)) for y in semantic_target])
            except:
                df_semantic_ready[target] = "Nan"
        
        content["semantic_target"] = content["target"].apply(lambda x: df_semantic_ready[x])
        
        content = content[content["semantic_target"]!="Nan"]
        
        self.content_df = content


        df_1["semantic_target"] = df_1["target"].apply(lambda x: self.ids_to_semantic(x,df_semantic_ready) )

        self.topic_df = df_1

    def sample_doc(self,semantic_target):
            try:
                semantic_target = semantic_target.split()[0]
            except:
                semantic_target = "None"
            return semantic_target

    def split(self):

        self.topic_df_train, self.topic_df_valid, self.topic_df_test = \
              np.split(self.topic_df.sample(frac=1, random_state=42), 
                       [int(.9*len(self.topic_df)), int(.95*len(self.topic_df))])
        
        self.topic_df_valid = self.topic_df_valid[(self.topic_df_valid.topic_category=="supplemental")|(self.topic_df_valid.topic_category=="aligned")].reset_index(drop=True)


        # self.topic_df_train["semantic_target"] = self.topic_df_train["semantic_target"].apply(lambda x: self.sample_doc(x))  
        # self.topic_df_train = self.topic_df_train[self.topic_df_train["semantic_target"] != "None"].reset_index(drop=True)
        
        if self.topic_train_explode ==True:
            self.topic_df_train.semantic_target = self.topic_df_train.semantic_target.str.split()
            self.topic_df_train.target = self.topic_df_train.target.str.split()
            self.topic_df_train = self.topic_df_train[self.topic_df_train.target.apply(lambda x:len(x))==self.topic_df_train.semantic_target.apply(lambda x:len(x))]
            self.topic_df_train = self.topic_df_train.explode(["semantic_target","target"])
            print("number_of_exploded_samples",len(self.topic_df_train))
            

        self.content_df_train_sample = self.content_df.sample(frac=0.01, random_state=42)

        self.topic_df_train_sample = self.topic_df_train.sample(frac=0.01, random_state=42)

        self.topic_tree_df_train_sample = self.topic_tree_df.sample(frac=0.01, random_state=42)

    def tokenize_data(self):
        
        self.topic_dataset_train = Dataset.from_pandas(self.topic_df_train[["source","semantic_target"]])#.remove_columns(['__index_level_0__'])
        self.topic_tree_dataset_train = Dataset.from_pandas(self.topic_tree_df[["source","semantic_target"]])#.remove_columns(['__index_level_0__'])   #
        self.content_dataset_train = Dataset.from_pandas(self.content_df[["source","semantic_target"]]).remove_columns(['__index_level_0__'])   ##
        self.topic_dataset_valid = Dataset.from_pandas(self.topic_df_valid[["source","semantic_target"]])#.remove_columns(['__index_level_0__'])
        self.topic_dataset_test = Dataset.from_pandas(self.topic_df_test[["source","semantic_target"]]).remove_columns(['__index_level_0__'])
        self.content_dataset_train_sample = Dataset.from_pandas(self.content_df_train_sample[["source","semantic_target"]]).remove_columns(['__index_level_0__'])
        self.topic_dataset_train_sample = Dataset.from_pandas(self.topic_df_train_sample[["source","semantic_target"]]).remove_columns(['__index_level_0__'])
        self.topic_tree_dataset_train_sample = Dataset.from_pandas(self.topic_tree_df_train_sample[["source","semantic_target"]])#.remove_columns(['__index_level_0__'])
        
        if self.DEBUG == False:
            datasets = DatasetDict({"topic_train":self.topic_dataset_train,"topic_tree_train":self.topic_tree_dataset_train,"content_train":self.content_dataset_train, "topic_test":self.topic_dataset_test ,
                                    "topic_valid":self.topic_dataset_valid ,"content_train_sample":self.content_dataset_train_sample,
                                    "topic_train_sample":self.topic_dataset_train_sample,"topic_tree_train_sample":self.topic_tree_dataset_train_sample})
        elif self.DEBUG == True:
            datasets = DatasetDict({"topic_train":self.topic_dataset_valid,"content_train":self.topic_dataset_valid, "topic_test":self.topic_dataset_valid ,
                                    "topic_valid":self.topic_dataset_valid ,"content_train_sample":self.topic_dataset_valid,"topic_train_sample":self.topic_dataset_valid})

        
        if bool(self.load_tokenized_datasets)==True:
            print("loading saved tokenized dataset")
            self.tokenized_datasets = load_from_disk(self.TOKENIZED_DATASETS_PATH)
        else:
            print("tokenizing...")
            self.tokenized_datasets = datasets.map(self.preprocess_function, batched=True).remove_columns(['source', 'semantic_target'])  #, batched=True  , num_proc=32
        if self.save_dateset==True:
            shutil.rmtree(self.TOKENIZED_DATASETS_PATH)
            self.tokenized_datasets.save_to_disk(self.TOKENIZED_DATASETS_PATH)
        self.tokenized_datasets.set_format(type='pt', columns=['input_ids', 'attention_mask', 'labels', 'labels_mask'])

    def concat_data(self):

        train_full = concatenate_datasets(self.topic_multiplier*[self.tokenized_datasets["topic_train"]]+ self.content_multiplier*[ self.tokenized_datasets["content_train"]]+ self.topic_tree_multiplier*[ self.tokenized_datasets["topic_tree_train"]]).shuffle(seed=42)  #28
        self.tokenized_Dataset = DatasetDict({"train":train_full,"topic_valid":self.tokenized_datasets["topic_valid"], "topic_test":self.tokenized_datasets["topic_test"],
                                          "content_train_sample":self.tokenized_datasets["content_train_sample"],
                                          "topic_train_sample":self.tokenized_datasets["topic_train_sample"],"topic_tree_train_sample":self.tokenized_datasets["topic_tree_train_sample"]})
        
    def print_counts(self):
        print(f"Number of samples:")

        data_len = len(self.tokenized_datasets["topic_train"])
        print(f"topic_train: {data_len}")

        data_len = len(self.tokenized_datasets["content_train"])
        print(f"content_train: {data_len}")

        data_len = len(self.tokenized_Dataset["train"])
        print(f"train: {data_len}")

        data_len = len(self.tokenized_datasets["topic_valid"])
        print(f"topic_valid: {data_len}")

        data_len = len(self.tokenized_datasets["topic_test"])
        print(f"topic_test: {data_len}")

        data_len = len(self.tokenized_datasets["content_train_sample"])
        print(f"content_train_sample: {data_len}")

        data_len = len(self.tokenized_datasets["topic_train_sample"])
        print(f"topic_train_sample: {data_len}")

        data_len = len(self.tokenized_datasets["topic_tree_train_sample"])
        print(f"topic_tree_train_sample: {data_len}")