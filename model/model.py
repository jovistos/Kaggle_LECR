
from transformers import AutoConfig,AutoModelForSeq2SeqLM


def give_model(model_checkpoint=None, 
               dropout_rate=0,
               num_decoder_layers=0, 
               freze_embed=True, 
               freeze_encoder=False,
               freze_layers=[], **kwargs):
    
    config = AutoConfig.from_pretrained(model_checkpoint)
    config.dropout_rate = dropout_rate
    if num_decoder_layers!=0:
        config.num_decoder_layers = num_decoder_layers
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, config=config)

    

    total_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"total number of params: {total_num_params}")

    def freeze_params(model):
        for par in model.parameters():
            par.requires_grad = False
    if freze_embed == True:            
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)

    if bool(freze_layers) == True:
        for name, param in model.named_parameters():
            # Set True only for params in the list 'params_to_train'
            # f_layers = ["shared","encoder.block.0","encoder.block.1","block.3","block.4","block.5"]  # ,"block.1","block.2" ,"block.3","block.4","block.5"

            param.requires_grad = False if any([x in name for x in freze_layers]) else True


    if freeze_encoder==True:
        for param in model.encoder.parameters():
            param.requires_grad = False
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable params: {num_trainable_params}")

    perc = (num_trainable_params/total_num_params)*100
    print(f"percentage of total params that are training: {perc}")

    print("###############################################################################################")

    return model