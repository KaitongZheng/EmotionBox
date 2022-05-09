import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel

from collections import namedtuple
import numpy as np
from progress.bar import Bar
from config import device

# pylint: disable=E1101,E1102


class PerformanceRNN_label(nn.Module):
    def __init__(self, event_dim, label_dim, init_dim, hidden_dim,
                 gru_layers=3, gru_dropout=0.3):
        super().__init__()

        self.event_dim = event_dim
        self.label_dim = label_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.concat_dim = event_dim + 1 + label_dim
        self.input_dim = hidden_dim
        self.output_dim = event_dim

        self.primary_event = self.event_dim - 1

        self.inithid_fc = nn.Linear(init_dim, gru_layers * hidden_dim)
        self.inithid_fc_activation = nn.Tanh()

        self.event_embedding = nn.Embedding(event_dim, event_dim)
        self.concat_input_fc = nn.Linear(self.concat_dim, self.input_dim)
        self.concat_input_fc_activation = nn.LeakyReLU(0.1, inplace=True)

        self.gru = nn.GRU(self.input_dim, self.hidden_dim,
                          num_layers=gru_layers, dropout=gru_dropout)
        self.output_fc = nn.Linear(hidden_dim * gru_layers, self.output_dim)
        self.output_fc_activation = nn.Softmax(dim=-1)

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_normal_(self.event_embedding.weight)
        nn.init.xavier_normal_(self.inithid_fc.weight)
        self.inithid_fc.bias.data.fill_(0.)
        nn.init.xavier_normal_(self.concat_input_fc.weight)
        nn.init.xavier_normal_(self.output_fc.weight)
        self.output_fc.bias.data.fill_(0.)

    def _sample_event(self, output, greedy=True, temperature=1.0):
        if greedy:
            return output.argmax(-1)
        else:
            output = output / temperature
            probs = self.output_fc_activation(output)
            return Categorical(probs).sample()

    def forward(self, event, label=None, hidden=None):
        # One step forward

        assert len(event.shape) == 2
        assert event.shape[0] == 1
        batch_size = event.shape[1]
        event = self.event_embedding(event)      #1*8->1*8*240 word embedding 0-240->240

        if label is None:
            default = torch.ones(1, batch_size, 1).to(device)
            label = torch.zeros(1, batch_size, self.label_dim).to(device)
        else:
            default = torch.zeros(1, batch_size, 1).to(device)  #1*8(batch_size)*1 all zero
            assert label.shape == (1, batch_size, self.label_dim) #label 1*8*240

        concat = torch.cat([event, default, label], -1) #concat 1*8*125  training1*64*125
        input = self.concat_input_fc(concat) #concat 1*8*512
        input = self.concat_input_fc_activation(input)

        _, hidden = self.gru(input, hidden) #input:1*batch*512    hidden:3*batch*512
        output = hidden.permute(1, 0, 2).contiguous() #8*3*512
        output = output.view(batch_size, -1).unsqueeze(0) #1*8*1536
        output = self.output_fc(output) #1*8*240
        return output, hidden
    
    def get_primary_event(self, batch_size):
        return torch.LongTensor([[self.primary_event] * batch_size]).to(device)
    
    def init_to_hidden(self, init):
        # [batch_size, init_dim]
        batch_size = init.shape[0]
        out = self.inithid_fc(init)
        out = self.inithid_fc_activation(out)
        out = out.view(self.gru_layers, batch_size, self.hidden_dim)
        return out
    
    def expand_labels(self, labels, steps):
        # [1 or steps, batch_size, label_dim] 训练的时候不用，生成的时候需要重复
        assert len(labels.shape) == 3
        assert labels.shape[2] == self.label_dim
        if labels.shape[0] > 1:
            assert labels.shape[0] >= steps
            return labels[:steps]
        return labels.repeat(steps, 1, 1)
    
    def generate(self, init, steps, events=None, labels=None, greedy=1.0,
                 temperature=1.0, teacher_forcing_ratio=1.0, output_type='index', verbose=False):
        # init [batch_size64, init_dim32] steps音乐样本长度
        # events [steps200, batch_size] indeces
        # labels [1 or steps, batch_size, label_dim]

        batch_size = init.shape[0]
        assert init.shape[1] == self.init_dim
        assert steps > 0

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            assert len(events.shape) == 2
            assert events.shape[0] >= steps - 1
            events = events[:steps-1]       #这里其实是和外面重复的，都是取1-199

        event = self.get_primary_event(batch_size)
        use_label = labels is not None
        if use_label:
            labels = self.expand_labels(labels, steps)  # length X batch X label_dim
        hidden = self.init_to_hidden(init)

        outputs = []
        step_iter = range(steps)
        if verbose:
            step_iter = Bar('Generating').iter(step_iter)

        for step in step_iter:
            label = labels[step].unsqueeze(0) if use_label else None #1X BATCH x control_dim
            output, hidden = self.forward(event, label, hidden) # generate output for sampling and hiddenlayer for next iter

            use_greedy = np.random.random() < greedy
            event = self._sample_event(output, greedy=use_greedy,temperature=temperature)  #用greedy直接就选择概率最大的 1*batch_size

            if output_type == 'index':
                outputs.append(event)
            elif output_type == 'softmax':
                outputs.append(self.output_fc_activation(output))
            elif output_type == 'logit':
                outputs.append(output)
            else:
                assert False

            if use_teacher_forcing and step < steps - 1: # avoid last one
                if np.random.random() <= teacher_forcing_ratio:
                    event = events[step].unsqueeze(0)
        
        return torch.cat(outputs, 0)

    # def beam_search(self, init, steps, beam_size, labels=None,
    #                 temperature=1.0, stochastic=False, verbose=False):
    #     assert len(init.shape) == 2 and init.shape[1] == self.init_dim
    #     assert self.event_dim >= beam_size > 0 and steps > 0
    #     
    #     batch_size = init.shape[0]
    #     current_beam_size = 1
    #     
    #     if labels is not None:
    #         labels = self.expand_labels(labels, steps) # [steps, batch_size, label_dim]
    # 
    #     # Initial hidden weights
    #     hidden = self.init_to_hidden(init) # [gru_layers, batch_size, hidden_size]
    #     hidden = hidden[:, :, None, :] # [gru_layers, batch_size, 1, hidden_size]
    #     hidden = hidden.repeat(1, 1, current_beam_size, 1) # [gru_layers, batch_size, beam_size, hidden_dim]
    # 
    #     
    #     # Initial event
    #     event = self.get_primary_event(batch_size) # [1, batch]
    #     event = event[:, :, None].repeat(1, 1, current_beam_size) # [1, batch, 1]
    # 
    #     # [batch, beam, 1]   event sequences of beams
    #     beam_events = event[0, :, None, :].repeat(1, current_beam_size, 1)
    # 
    #     # [batch, beam] log probs sum of beams
    #     beam_log_prob = torch.zeros(batch_size, current_beam_size).to(device)
    #     
    #     if stochastic:
    #         # [batch, beam] Gumbel perturbed log probs of beams
    #         beam_log_prob_perturbed = torch.zeros(batch_size, current_beam_size).to(device)
    #         beam_z = torch.full((batch_size, beam_size), float('inf'))
    #         gumbel_dist = Gumbel(0, 1)
    # 
    #     step_iter = range(steps)
    #     if verbose:
    #         step_iter = Bar(['', 'Stochastic '][stochastic] + 'Beam Search').iter(step_iter)
    # 
    #     for step in step_iter:
    #         if labels is not None:
    #             label = labels[step, None, :, None, :] # [1, batch, 1, label]
    #             label = label.repeat(1, 1, current_beam_size, 1) # [1, batch, beam, label]
    #             label = label.view(1, batch_size * current_beam_size, self.label_dim) # [1, batch*beam, label]
    #         else:
    #             label = None
    #         
    #         event = event.view(1, batch_size * current_beam_size) # [1, batch*beam0]
    #         hidden = hidden.view(self.gru_layers, batch_size * current_beam_size, self.hidden_dim) # [grus, batch*beam, hid]
    # 
    #         logits, hidden = self.forward(event, label, hidden)
    #         hidden = hidden.view(self.gru_layers, batch_size, current_beam_size, self.hidden_dim) # [grus, batch, cbeam, hid]
    #         logits = (logits / temperature).view(1, batch_size, current_beam_size, self.event_dim) # [1, batch, cbeam, out]
    #         
    #         beam_log_prob_expand = logits + beam_log_prob[None, :, :, None] # [1, batch, cbeam, out]
    #         beam_log_prob_expand_batch = beam_log_prob_expand.view(1, batch_size, -1) # [1, batch, cbeam*out]
    #         
    #         if stochastic:
    #             beam_log_prob_expand_perturbed = beam_log_prob_expand + gumbel_dist.sample(beam_log_prob_expand.shape)
    #             beam_log_prob_Z, _ = beam_log_prob_expand_perturbed.max(-1) # [1, batch, cbeam]
    #             # print(beam_log_prob_Z)
    #             beam_log_prob_expand_perturbed_normalized = beam_log_prob_expand_perturbed
    #             # beam_log_prob_expand_perturbed_normalized = -torch.log(
    #             #     torch.exp(-beam_log_prob_perturbed[None, :, :, None])
    #             #     - torch.exp(-beam_log_prob_Z[:, :, :, None])
    #             #     + torch.exp(-beam_log_prob_expand_perturbed)) # [1, batch, cbeam, out]
    #             # beam_log_prob_expand_perturbed_normalized = beam_log_prob_perturbed[None, :, :, None] + beam_log_prob_expand_perturbed # [1, batch, cbeam, out]
    #             
    #             beam_log_prob_expand_perturbed_normalized_batch = \
    #                 beam_log_prob_expand_perturbed_normalized.view(1, batch_size, -1) # [1, batch, cbeam*out]
    #             _, top_indices = beam_log_prob_expand_perturbed_normalized_batch.topk(beam_size, -1) # [1, batch, cbeam]
    #             
    #             beam_log_prob_perturbed = \
    #                 torch.gather(beam_log_prob_expand_perturbed_normalized_batch, -1, top_indices)[0] # [batch, beam]
    # 
    #         else:
    #             _, top_indices = beam_log_prob_expand_batch.topk(beam_size, -1)
    #         
    #         beam_log_prob = torch.gather(beam_log_prob_expand_batch, -1, top_indices)[0] # [batch, beam]
    #         
    #         beam_index_old = torch.arange(current_beam_size)[None, None, :, None] # [1, 1, cbeam, 1]
    #         beam_index_old = beam_index_old.repeat(1, batch_size, 1, self.output_dim) # [1, batch, cbeam, out]
    #         beam_index_old = beam_index_old.view(1, batch_size, -1) # [1, batch, cbeam*out]
    #         beam_index_new = torch.gather(beam_index_old, -1, top_indices)
    #         
    #         hidden = torch.gather(hidden, 2, beam_index_new[:, :, :, None].repeat(4, 1, 1, 1024))
    #         
    #         event_index = torch.arange(self.output_dim)[None, None, None, :] # [1, 1, 1, out]
    #         event_index = event_index.repeat(1, batch_size, current_beam_size, 1) # [1, batch, cbeam, out]
    #         event_index = event_index.view(1, batch_size, -1) # [1, batch, cbeam*out]
    #         event = torch.gather(event_index, -1, top_indices) # [1, batch, cbeam*out]
    #         
    #         beam_events = torch.gather(beam_events[None], 2, beam_index_new.unsqueeze(-1).repeat(1, 1, 1, beam_events.shape[-1]))
    #         beam_events = torch.cat([beam_events, event.unsqueeze(-1)], -1)[0]
    #         
    #         current_beam_size = beam_size
    # 
    #     best = beam_events[torch.arange(batch_size).long(), beam_log_prob.argmax(-1)]
    #     best = best.contiguous().t()
    #     return best
