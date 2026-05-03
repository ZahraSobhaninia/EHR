"""
Module: corebehrt_module

This module defines customized EHR-focused BERT models built on top of ModernBertModel:

- CorebehrtEncoder: replaces token embeddings with temporal EHR embeddings and causal encoder layers.
- CorebehrtForPretraining: extends the encoder for masked language model pretraining on EHR sequences.
- CorebehrtForFineTuning: extends the encoder for downstream classification/regression tasks on EHR data.
"""

import logging
from typing import Tuple

from corebehrt.functional.modeling import attention
import torch
import torch.nn as nn
from transformers import ModernBertModel
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead




from corebehrt.constants.data import (
    ABSPOS_FEAT,
    AGE_FEAT,
    ATTENTION_MASK,
    CONCEPT_FEAT,
    DEFAULT_VOCABULARY,
    PAD_TOKEN,
    SEGMENT_FEAT,
    TARGET,
)
from corebehrt.constants.model import (
    TIME2VEC_ABSPOS_SCALE,
    TIME2VEC_ABSPOS_SHIFT,
    TIME2VEC_AGE_SCALE,
    TIME2VEC_AGE_SHIFT,
)
from corebehrt.functional.modeling.attention import make_attention_causal
from corebehrt.modules.model.embeddings import EhrEmbeddings
from corebehrt.modules.model.heads import FineTuneHead

logger = logging.getLogger(__name__)


#What it does:
    #Takes medical data (concepts, age, time, etc.)
    #Converts them into embeddings (numbers the model understands)
    #Sends them into a transformer (ModernBERT)

#Inputs:
    #concept → medical codes (like diagnosis)
    #segment → visit grouping
    #age → patient age at each event
    #abspos → time position

class CorebehrtEncoder(ModernBertModel):
    """
    Encoder backbone for EHR data using ModernBert.

    Attributes:
        embeddings (EhrEmbeddings): custom embeddings for concepts, segments, age, and absolute position.
        layers (nn.ModuleList): list of causal encoder layers replacing standard BERT layers.
    """

    def __init__(self, config):
        super().__init__(config)
        # config.is_decoder = True
        # config.add_cross_attention = False
        self.embeddings = EhrEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            embedding_dropout=config.embedding_dropout,
            pad_token_id=config.pad_token_id,
            age_scale=getattr(config, "age_scale", TIME2VEC_AGE_SCALE),
            age_shift=getattr(config, "age_shift", TIME2VEC_AGE_SHIFT),
            abspos_scale=getattr(config, "abspos_scale", TIME2VEC_ABSPOS_SCALE),
            abspos_shift=getattr(config, "abspos_shift", TIME2VEC_ABSPOS_SHIFT),
        )
        self.is_causal = getattr(config, "is_causal", False)

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass building embeddings and attention mask, then calling ModernBertModel.

        Args:
            batch (dict): must contain:
                - "concept": Tensor of token indices (B, L)
                - "segment": Tensor of segment IDs (B, L)
                - "age": Tensor of patient ages (B, L)
                - "abspos": Tensor of absolute position values (B, L)
            **kwargs: Additional arguments to pass to the ModernBertModel forward method

        Returns:
            BaseModelOutput: output of ModernBertModel with last_hidden_state, etc.
        """
        if ATTENTION_MASK in batch:
            attention_mask = batch[ATTENTION_MASK]
        else:
            attention_mask = (
                batch[CONCEPT_FEAT] != DEFAULT_VOCABULARY[PAD_TOKEN]
            ).float()

        #feature space creation
        #Each event is converted independently and has no connection to the other events.
        #"diabetes" → [0.2, -0.5, 1.1, 0.7]
        
        inputs_embeds = self.embeddings(
            input_ids=batch[CONCEPT_FEAT],
            segments=batch[SEGMENT_FEAT],
            age=batch[AGE_FEAT],
            abspos=batch[ABSPOS_FEAT],
        )
        #Attehtion operation:
        #The model looks at the whole sequence of events and decides which ones are important to focus
        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )
     # This is an internal function in the Transformer
    #forward()
    #↓
    #prepare mask
    #  ↓
    #_update_attention_mask()   ← 🔥
    #   ↓
    #attention layers
    def _update_attention_mask(
        self, attention_mask: torch.Tensor, output_attentions: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls super()._update_attention_mask and adds causal masking if self.is_causal=True.
        Returns:
            Tuple of:
            - Global attention mask
            - Sliding window mask for local attention
        """
        global_attention_mask, sliding_window_mask = super()._update_attention_mask(
            attention_mask, output_attentions
        )
        if self.is_causal:
            global_attention_mask = make_attention_causal(global_attention_mask)
            sliding_window_mask = make_attention_causal(sliding_window_mask)

        return global_attention_mask, sliding_window_mask
    #global_attention_mask -> every token can attend to every other token (full attention)
    #sliding_window_mask -> each token can only attend to a fixed number of tokens before it (local attention)
 

# Define MLM model on top of encoder

class CorebehrtForPretraining(CorebehrtEncoder):
    """
    Masked Language Model head for EHR pretraining.

    Adds a prediction head and linear decoder on top of CorebehrtEncoder.
    """
#Initialize base encoder (Transformer)
    def __init__(self, config):
        super().__init__(config)
        #Loss for classification over vocab
        self.loss_fct = nn.CrossEntropyLoss()
        #Prediction head that transforms hidden states to vocab space
        #Transform hidden states before prediction to improve performance (as in BERT)
        self.head = ModernBertPredictionHead(config)
        
        # Linear layer that maps from hidden size to vocab size for MLM prediction
        # Map hidden → vocab logits
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )
        #Whether to train only masked tokens (sparse prediction) or all tokens (dense prediction)
        self.sparse_prediction = self.config.sparse_prediction
        #Index in labels to ignore for sparse prediction (non-masked tokens)
        #For example, if using -100 as ignore index, then labels for non-masked tokens should be set to -100 and will not contribute to loss or backprop.
    
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

    # Inspiration from ModernBertForMaskedLM

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass for masked language modeling.

        Args:
            batch (dict): must contain 'concept', 'segment', 'age', 'abspos';
                          optional 'target' for labels (B, L).
            **kwargs: Additional arguments to pass to the encoder forward method

        Returns:
            BaseModelOutput: with logits and optional loss/labels if targets provided.
        """
        ## Run Transformer
        outputs = super().forward(batch, **kwargs)

        #Get contextual representations for each token from the last hidden state of the encoder
        last_hidden_state = outputs[0]

        #Get ground truth labels for masked language modeling (if provided)
        labels = batch.get(TARGET)

        #Only train on masked tokens if sparse_prediction=True
        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            #Flatten hidden states to (B*L, H) so we can filter out non-masked tokens
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            #Select valid (masked) tokens where labels are not equal to the ignore index
            # For example, if using -100 as ignore index, then only positions where labels != -100 will be selected for training.
            # فقط جاهایی که جواب داریم رو نگه دار
            #-100 یعنی: "اینجا رو نادیده بگیر"
            mask_tokens = labels != self.sparse_pred_ignore_index
            #Keep only masked positions
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        #hidden → processed → vocab logits 
        logits = self.decoder(self.head(last_hidden_state))
        #Store predictions in outputs
        outputs.logits = logits

        #If labels exist → compute loss

        if labels is not None:
            #Calculate MLM loss 
            outputs.loss = self.get_loss(logits, labels)
            #Store labels 
            outputs.labels = labels

        #Return everything (logits, loss, labels if exist)
        return outputs

    def get_loss(self, logits, labels):
        """Calculate loss for masked language model."""
        #CrossEntropy over vocab
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


#Add classification head on top of Transformer
class CorebehrtForFineTuning(CorebehrtEncoder):
    """
    Fine-tuning head for downstream classification on EHR sequences.

    Adds a binary classification head (BCEWithLogits) on top of sequence outputs.
    """

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "pos_weight", None):
            pos_weight = torch.tensor(config.pos_weight)
        else:
            pos_weight = None

        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.cls = FineTuneHead(hidden_size=config.hidden_size)

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass for fine-tuning.

        Args:
            batch (dict): must contain 'concept', 'segment', 'age', 'abspos', 'attention_mask';
                          optional 'target' as labels.
            **kwargs: Additional arguments to pass to the encoder forward method

        Returns:
            BaseModelOutput: with logits and optional loss if target provided.
        """
        outputs = super().forward(batch, **kwargs)

        sequence_output = outputs[0]  # Last hidden state
        logits = self.cls(sequence_output, batch[ATTENTION_MASK])
        outputs.logits = logits

        if batch.get(TARGET) is not None:
            outputs.loss = self.get_loss(logits, batch[TARGET])

        return outputs

    def get_loss(self, hidden_states, labels):
        return self.loss_fct(hidden_states.view(-1), labels.view(-1))
    

##
class CorebehrtForMultiTaskFineTuning(CorebehrtEncoder):
    """
    Multi-task fine-tuning head for multiple clinical outcomes.
    """

    def __init__(self, config):
        super().__init__(config)
        
        # task names از config
        task_names = getattr(config, 'tasks', ['mortality'])
        
        # یه head برای هر task
        self.task_heads = nn.ModuleDict({
            task: FineTuneHead(hidden_size=config.hidden_size)
            for task in task_names
        })
        
        # یه loss برای هر task
        pos_weights = getattr(config, 'pos_weights', {})
        self.loss_fcts = nn.ModuleDict({
            task: nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weights[task]) 
                if task in pos_weights else None
            )
            for task in task_names
        })
        
        self.task_names = task_names

    def forward(self, batch: dict, **kwargs):
        outputs = super().forward(batch, **kwargs)
        sequence_output = outputs[0]

        # هر task یه logit جداگانه داره
        all_logits = {}
        for task in self.task_names:
            all_logits[task] = self.task_heads[task](
                sequence_output, batch[ATTENTION_MASK]
            )

        # combined logits
        outputs.logits = torch.stack(
            [all_logits[task] for task in self.task_names], dim=1
        )

        # loss
        if batch.get(TARGET) is not None:
            total_loss = 0
            labels = batch[TARGET]  # shape: (batch, n_tasks)
            for i, task in enumerate(self.task_names):
                task_loss = self.loss_fcts[task](
                    all_logits[task].view(-1),
                    labels[:, i].view(-1)
                )
                total_loss += task_loss
            outputs.loss = total_loss

        return outputs