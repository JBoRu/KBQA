import math

import torch
import torch.utils.checkpoint
import torch.nn as nn

from NSM.Modules.TransformerStructure.activations import ACT2FN


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config, word_embeddings):
        super(BertEmbeddings, self).__init__()
        self.config = config
        if word_embeddings is not None:
            self.word_embeddings = word_embeddings
            device = self.word_embeddings(torch.LongTensor([1])).device
            self.cls = nn.Parameter(torch.rand((1, config["word_dim"]))).to(device)
            if self.config["input_trans"]:
                self.input_trans = nn.Linear(config["word_dim"], config["hidden_size"])
        else:
            self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"], padding_idx=config["pad_token_id"])
        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["word_dim"])
        if config["no_type_encoding"]:
            self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["word_dim"])

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config["max_position_embeddings"]).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        bs, seq_length = input_shape

        if position_ids is None:
            position_ids = self.position_ids[:, 0:seq_length+1]

        if token_type_ids is None:
            token_type_ids = torch.ones((bs, seq_length+1), dtype=torch.long, device=self.position_ids.device)
            token_type_ids[:, 0] = 0

        inputs_embeds = self.word_embeddings(input_ids) # (bs, seq_len, word_emb)
        cls = self.cls.unsqueeze(dim=0).repeat(bs, 1, 1) # (bs, 1, word_emb)
        inputs_embeds = torch.cat((cls, inputs_embeds), dim=1) # (bs, 1+seq_len, word_emb)
        if self.config["no_type_encoding"]:
            token_type_embeddings = self.token_type_embeddings(token_type_ids) # (bs, 1+seq_len, type_emb)
            embeddings = inputs_embeds + token_type_embeddings
        else:
            embeddings = inputs_embeds

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if self.config["input_trans"]:
            embeddings = self.input_trans(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config["hidden_size"] % config["num_attention_heads"] != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config["hidden_size"], config["num_attention_heads"])
            )

        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = int(config["hidden_size"] / config["num_attention_heads"])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config["hidden_size"], self.all_head_size)
        self.key = nn.Linear(config["hidden_size"], self.all_head_size)
        self.value = nn.Linear(config["hidden_size"], self.all_head_size)

        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config["max_position_embeddings"]
            self.distance_embedding = nn.Embedding(2 * config["max_position_embeddings"] - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config["hidden_size"], config["intermediate_size"])
        if isinstance(config["hidden_act"], str):
            self.intermediate_act_fn = ACT2FN[config["hidden_act"]]
        else:
            self.intermediate_act_fn = config["hidden_act"]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        num_hidden_layers = self.config["num_hidden_layers"]
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(num_hidden_layers)])

    def forward(self, step, hidden_states, attention_mask=None, output_attentions=False):
        for i, layer_module in enumerate(self.layer):
            if i == step:
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)
                hidden_states = layer_outputs[0]
                break
        return hidden_states

class BertModel(nn.Module):
    def __init__(self, config, word_embeddings):
        super(BertModel, self).__init__()
        self.config = config
        if not self.config["bert_encoding"]:
            self.embeddings = BertEmbeddings(config, word_embeddings)
        self.encoder = BertEncoder(config)
        self.extended_attention_mask = None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_extended_attention_mask(self, attention_mask, input_shape):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def forward(
        self,
        step=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        hidden_states=None,
        output_attentions=None,
        type=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config["output_attentions"]

        if input_ids is not None and hidden_states is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None and type == "embed":
            device = input_ids.device if input_ids is not None else hidden_states.device
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            seq_length = seq_length + 1 # add cls
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            else:
                attention_mask_cls = torch.ones_like(attention_mask)[:, 0:1]
                attention_mask = torch.cat((attention_mask_cls, attention_mask), dim=1)
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            self.extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids
            )
            return embedding_output
        elif hidden_states is not None and type == "selfatt":
            input_shape = hidden_states.size()
            if attention_mask is not None:
                self.extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
            hidden_states = hidden_states
            encoder_outputs = self.encoder(
                step=step,
                hidden_states=hidden_states,
                attention_mask=self.extended_attention_mask,
                output_attentions=output_attentions
            )
            sequence_output = encoder_outputs
            return sequence_output
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")