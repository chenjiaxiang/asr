import warnings
from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from asr.models import ASRModel
from asr.modules import Linear
from asr.search import BeamSearchRNNTransducer
from asr.tokenizers.tokenizer import Tokenizer
from asr.utils import get_class_name


class ASRTransducerModel(ASRModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ASRTransducerModel, self).__init__(configs, tokenizer)
        self.encoder = None
        self.decoder = None
        self.decode = self.greedy_decode

        if hasattr(self.configs.model, "encoder_dim"):
            in_features = self.configs.model.encoder_dim + self.configs.model.decoder_output_dim
        elif hasattr(self.configs.model, "output_dim"):
            in_features = self.configs.model.output_dim << 1
        else:
            raise ValueError("Transducer model must contain `encoder_dim` for `encoder_hidden_state_dim` config.") # TODO maybe wrong.
        
        self.fc = nn.Sequential(
            Linear(in_features=in_features, out_features=in_features),
            nn.Tanh(),
            Linear(in_features=in_features, out_features=self.num_classes),
        )

    def set_beam_decoder(self, beam_size: int = 3, expand_beam: float = 2.3, state_beam: float = 4.6):
        """Setting beam search decode"""
        self.deocde = BeamSearchRNNTransducer(
            joint=self.joint,
            decoder=self.decoder,
            beam_size=beam_size,
            expand_beam=expand_beam,
            state_beam=state_beam,
            blank_id=self.tokenizer.blank_id,
        )

    def collect_outputs(
            self,
            stage: str,
            logits: torch.FloatTensor,
            input_lengths: torch.IntTensor,
            targets: torch.IntTensor,
            target_lengths: torch.IntTensor,
    ) -> OrderedDict:
        predictions = logits.max(-1)[1]

        loss = self.criterion(
            logits=logits,
            targets=targets[:, 1:].contiguous().int(),
            input_lengths=input_lengths.int(),
            target_lengths=target_lengths.int(),
        )

        self.info(
            {
                f"{stage}_loss": loss,
                "learning_rate": self.get_lr(),
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "predictions": predictions,
                "targets": targets,
                "logits": logits,
            }
        )
    
    def _expand_for_joint(self, encoder_outpus: Tensor, decoder_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        input_lengths = encoder_outpus.size(1)
        target_lengths = decoder_outputs.size(1)

        encoder_outpus = encoder_outpus.unsqueeze(2)
        decoder_outputs = decoder_outputs.unsqueeze(1)

        encoder_outpus = encoder_outpus.repeat([1, 1, target_lengths, 1])
        decoder_outputs = decoder_outputs.repeat([1, input_lengths, 1, 1])

        return encoder_outpus, decoder_outputs

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        r"""
        Joint `encoder_outputs` and `decoder_outputs`.

        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
            decoder_output (torch.FloatTensor): A output sequence of deocders. `FloatTensor` of size ``
            (batch, seq_length, dimension)``
        
        Returns:
            outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outpus.`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            encoder_outputs, decoder_outputs = self._expand_for_joint(encoder_outputs, decoder_outputs)
        else:
            assert encoder_outputs.dim() == decoder_outputs.dim()

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=1)
        outputs = self.fc(outputs).log_softmax(dim=-1)
        
        return outputs

    def greedy_decode(self, encoder_outputs: Tensor, max_length: int) -> Tensor:
        r"""
        Decode `encode_outputs`.

        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` fo size
                ``(batch, seq_length, dimension)``
            max_length (int): max decoding time step
        
        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        outputs = list()

        for encoder_output in encoder_outputs:
            pred_tokens = list()
            decoder_input = encoder_output.new_zeros(1, 1).fill_(self.decoder.sos_id).long()
            decoder_output, hidden_state = self.decoder(decoder_input)

            for t in range(max_length):
                step_output = self.joint(encoder_output[t].view(-1), decoder_output.view(-1))

                pred_token = step_output.argmax(dim=0)
                pred_token = int(pred_token.item())
                pred_tokens.append(pred_token)

                decoder_input = torch.LongTensor([[pred_token]])
                if torch.cuda.is_available():
                    decoder_input = decoder_input.cuda()

                decoder_input, hidden_state = self.decoder(decoder_input, hidden_state=hidden_state)

            outputs.append(torch.LongTensor(pred_tokens))

        return torch.stack(outputs, dim=0)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Dict[str, Tensor]:
        r"""
        Decode `encoder_outputs`.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length. dimension)``.
            input_lengths: (torch.LongTensor): The length of input tensor. ``(batch)``
        
        Returns:
            dict (dict): Result of model predictions that contains `predictions`,
                `encoder_outputs`, `encoder_output_lengths`
        """
        if get_class_name(self.encoder) in ["ConformerEncoder", "ContextNetEncoder"]:
            encoder_outputs, _, output_lengths = self.encoder(inputs, input_lengths)
        else:
            encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        predictions = self.decode(encoder_outputs, max_length)

        return {
            "predictions": predictions,
            "encoder_outputs": encoder_outputs,
            "encoder_output_lengths": output_lengths,
        }

    def training_step(self, batch: tuple, batch_idxL int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for training.
        
        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch
            
        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch
        if get_class_name(self.encoder) in ["ConformerEncoder", "ContextNetEncoder"]:
            encoder_outputs, _, output_lengths = self.encoder(inputs, input_lengths)
        else:
            encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        
        decoder_outputs, _ = self.decoder(targets, target_lengths)
        logits = self.joint(encoder_outputs, decoder_outputs)

        return self.collect_outputs(
            "train",
            logits=logits,
            input_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.
        
        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch
            
        Returns:
            loss (torch.Tensor): loss for validation.
        """
        inputs, targets, input_lengths, target_lengths = batch
        if get_class_name(self.encoder) in ["ConformerEncoder", "ContextNetEncoder"]:
            encoder_outputs, _, output_lengths = self.encoder(inputs, input_lengths)
        else:
            encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        
        decoder_outputs, _ = self.decoder(targets, target_lengths)
        logits = self.joint(encoder_outputs, decoder_outputs)

        return self.collect_outputs(
            "val",
            logits=logits,
            input_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for test.
        
        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch
            
        Returns:
            loss (torch.Tensor): loss for test.
        """
        inputs, targets, input_lengths, target_lengths = batch
        if get_class_name(self.encoder) in ["ConformerEncoder", "ContextNetEncoder"]:
            encoder_outputs, _, output_lengths = self.encoder(inputs, input_lengths)
        else:
            encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        
        decoder_outputs, _ = self.decoder(targets, target_lengths)
        logits = self.joint(encoder_outputs, decoder_outputs)

        return self.collect_outputs(
            "test",
            logits=logits,
            input_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )