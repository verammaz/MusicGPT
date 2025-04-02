import wandb
import os
import sys
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator

from src.models.gpt import GPT
from src.train.train import Trainer
from src.utils.general_utils import set_seed, setup_logging, CfgNode as CN
from src.utils.data_utils import get_data, split_data


wandb.login()

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out'

    # pipeline
    C.pipeline = CN()
    C.pipeline.train_token = False
    C.pipeline.train_gpt = True
    C.pipeline.evaluate = True
    C.pipeline.sample = True

    # data
    C.data = None

    # model
    C.model = GPT.get_default_config()

    # trainer
    C.gpt_trainer = Trainer.get_default_config()

    return C


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and override from the command line
    config = get_config()
    config.merge_from_args(sys.argv[1:])

    # check that train data is provided
    if config.data is None:
        print("No data path provided. Specify --data= argument.")
        sys.exit(1)

    set_seed(config.system.seed)

    if config.model.model_name is None:
        config.model.name = f'{config.model.model_type}_l{config.model.n_layer}_q{config.model.n_query_head}_kv{config.model.n_kv_head}'

    else:
        config.model.name = f'{config.model.model_type}_{config.model.model_name}_l{config.model.n_layer}_q{config.model.n_query_head}_kv{config.model.n_kv_head}'

    if config.model.rope : config.model.name += '_rope'

    # set up tokenizer 
    if not config.pipeline.train_token:
        tokenizer_config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
        tokenizer = REMI(tokenizer_config)
    else:
        pass

    # construct the model
    config.model.vocab_size = len(tokenizer)
    
    print(config)
    model = GPT(config.model)
    
    if config.model.pretrained != None:
        print("Loading pretrained model...")
        pretrained_state_dict = torch.load(config.model.pretrained, weights_only=True)
        model.load_state_dict(pretrained_state_dict, strict=False) 
    
    #setup_logging(config)
    out_dir = os.path.join(config.system.work_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    wandb.init(project="MusicGen", config=config)

    dataloader = get_data(tokenizer, config.data, max_seq_len=config.model.block_size, batch_size=config.gpt_trainer.batch_size,
                              subsets=False, return_datasets=False, split=True, augment=True)
    
    if config.pipeline.train_gpt:

        # construct trainer object
        trainer = Trainer(config.gpt_trainer, model, dataloader)

        # iteration callback
        def batch_end_trainer_callback(trainer):
            
            wandb.log({"n_examples" : trainer.n_examples, "train_loss": trainer.loss})
            
            ckpt_path = os.path.join(out_dir, f'{config.model.name}.pt')

            if (trainer.n_iter + 1) % 200 == 0:
                model.eval()
                torch.save(model.state_dict(), ckpt_path)
            
                # revert model to training mode
                model.train()

        trainer.set_callback('on_batch_end', batch_end_trainer_callback)

        # run the optimization
        trainer.run()

    # sample
    if config.pipeline.sample:
        
        sampled_tokens = model.sample(max_new_tokens=1024, device=None, verbose=True, bos_token_id=1, pad_token_id=0)
        outmidi = os.path.join(out_dir, "scratch_sample.mid")
        tokenizer(sampled_tokens[0]).dump_midi(outmidi)


        for batch_idx, encodings in enumerate(dataloader):

            if batch_idx > 1:
                break

            input_ids = encodings["input_ids"]  # shape (B, T)
            
            # Pick a random sequence from the batch
            random_idx = np.random.randint(0, input_ids.size(0))
            
            # Get the tokens for that sequence as a list
            seed_sequence = input_ids[random_idx].tolist()
            
            # Optionally, pick a random chunk of it if you want partial context
            # e.g., random chunk of length <= T
            #random_length = np.random.randint(1, len(seed_sequence) + 1)
            #seed_sequence = seed_sequence[:random_length]
            
            # Now feed this partial sequence as a prompt to the model
            # We'll generate, say, 128 more tokens
            generated = model.sample(
                start_tokens=seed_sequence,
                size=1,            # we want 1 sequence
                temperature=1.0,
                max_len=512,       # or max_new_tokens, depending on your function
                device=None
            )

            outmidi = os.path.join(out_dir, "train_sample.mid")
            tokenizer(seed_sequence[0]).dump_midi(outmidi)

            outmidi = os.path.join(out_dir, "continued_sample.mid")
            tokenizer(generated[0]).dump_midi(outmidi)
            
            
    # evaluate
    if config.pipeline.evaluate:
        pass
    

    