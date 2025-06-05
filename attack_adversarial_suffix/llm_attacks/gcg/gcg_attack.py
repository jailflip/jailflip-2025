import gc
import traceback
import os
from copy import deepcopy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)         # a tensor in size 32000,4096
    one_hot = torch.zeros(                              # in shape length_control * vocabulary, e.g.in shape 20*32000
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),            # in shape 1*control_length
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()                               # in shape length_control * vocabulary, e.g.in shape 20*32000
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)  # in shape length_control * 4096, e.g.in shape 20*4096

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
                        [
                            embeds[:,:input_slice.start,:],
                            input_embeds,
                            embeds[:,input_slice.stop:,:]
                        ],
                        dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss(reduction='none')(logits[0,loss_slice,:], targets)
    loss = torch.mean(loss)
    loss.backward()
    return one_hot.grad.clone()

class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def grad(self, model):
        # in shape length_control * vocabulary, e.g.in shape 20*32000
        # this process is for sampling

        return token_gradients(
            model,
            self.input_ids.to(model.device),
            self._control_slice,
            self._target_slice,
            self._loss_slice,
        )

class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):
        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.inf
        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)

        new_token_pos = torch.arange(
            0,
            len(control_toks),
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)

        new_token_val = torch.gather(
            top_indices[new_token_pos], 1,
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks

class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self,
            batch_size=1024,
            topk=256,
            temp=1,
            allow_non_ascii=True,
            target_weight=1,
            control_weight=0.1,     # for progressive attack, this is zero
            verbose=False,          # for individual prompt attack, this is True
            filter_cand=True,       # for individual prompt attack as well as progressive, this is True
            current_step=None
        ):

        main_device = self.models[0].device
        control_cands = []

        # what's effectively going on is
        # grad is in shape length_control * vocabulary, e.g.in shape 20*32000
        for j, worker in enumerate(self.workers):
            assert j==0
            grad = self.prompts[j].grad( worker.model )
            grad = grad / grad.norm(dim=-1, keepdim=True)


        with torch.no_grad():
            # to sample on the last but the only one self.prompts
            # is in shape batch_size * control_length, e.g. 256*20
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            del grad
            torch.cuda.empty_cache()
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))

        del control_cand ; gc.collect()
        torch.cuda.empty_cache()

        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)

        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                progress = tqdm( range(len(self.prompts[0])), total=len(self.prompts[0]) ) if verbose else enumerate(self.prompts[0])
                for i in progress:      # e.g. 25

                    for k, worker in enumerate(self.workers):
                        assert k==0
                        logits, ids = self.prompts[k][i].logits(worker.model, cand, return_ids=True)
                        logits = (logits,)
                        ids = (ids,)

                    torch.cuda.empty_cache()
                    temp_gcg_loss = sum([
                        target_weight * self.prompts[k][i].target_loss(logit, id).to(main_device)
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    loss[ j*batch_size : (j+1)*batch_size ] += temp_gcg_loss

                    if control_weight != 0:     # to compute the control loss in one step, might be regulaizing
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])

                    if True:                    # original GCG loss
                        overall_loss = loss     # no augmentation situation, e.g. our prior DSN attack (https://github.com/DSN-2024/DSN)
                    del logits, ids; gc.collect()
                    torch.cuda.empty_cache()

                    if verbose:                 # pass the info into the tqdm slide
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = overall_loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], overall_loss[min_idx]

            # to store the loss wrt each suffix in the batch
            loss_wrt_whole_ctrl = (overall_loss/len(self.prompts[0])/len(self.workers)).tolist()

        if True: # to store two loss term during each step into a pth file
            logfile = self.para.result_prefix
            if logfile.endswith('.json'):
                loss_his_path = deepcopy(logfile)
                loss_his_path = loss_his_path.replace('.json', '_loss_history@step')

            if current_step == 1:
                store_gcg_loss_history_previous = []
            else:
                temp_loss_dict = torch.load(loss_his_path+f"{current_step-1}.pth", weights_only=False)
                store_gcg_loss_history_previous = temp_loss_dict['gcg']

            store_gcg_loss_history = [loss[min_idx].item() / len(self.prompts[0]) / len(self.workers)]
            loss_history = {
                'gcg':store_gcg_loss_history_previous + store_gcg_loss_history,
                }

            torch.save(loss_history, loss_his_path+f"{current_step-1}.pth")
            os.rename(loss_his_path+f"{current_step-1}.pth", loss_his_path+f"{current_step}.pth")

        if self.para.debug_mode or True:     # for debugging
            print('#'*75)
            print("the loss pth file stores the latest updated loss value of gcg-loss:")
            print(f"GCG loss:{loss_history['gcg'][-1]:.4f}",end=' '*8)
            print(loss_his_path+f"{current_step}.pth")
            print('#'*75)
            print()
            print()

        del loss, overall_loss ; gc.collect()
        torch.cuda.empty_cache()

        output_temp_str = 'Current length: '
        output_temp_str += str( len(self.workers[0].tokenizer(next_control, add_special_tokens=False).input_ids) )
        output_temp_str += ' the adv suffix is now:'

        if self.logger is None:         # output by print
            print(output_temp_str)
            print(next_control)
        else:                           # output in the logger!
            self.logger.info(output_temp_str)
            self.logger.info(next_control)
        del output_temp_str

        if self.para.debug_mode or False:
            assert False

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers), (control_cands[0], loss_wrt_whole_ctrl)