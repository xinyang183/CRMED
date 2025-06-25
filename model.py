import random
import torch
import torch.nn as nn
import copy
from torch_geometric.data import Data
from utterance_encoder import UtteranceEncoder
from graph_encoder import GraphEncoder
from intent_predictor import IntentPredictor
from graph_traverser import GraphTraverser
from explicit_recommender import ExplicitRecommender


class RecSystemPro(nn.Module):
    #beta=tanh(QW+KU)*V
    def __init__(self, device_str='cuda:1', rnn_type="RNN_TANH", use_bert=True, 
                 utter_enc_size=64, dropout_rate=0.5, max_turns=10, 
                 relation_count=12, base_count=15, graph_enc_size=64, 
                 attn_hidden=20, neg_sample_ratio=5, dataset="redial", word_net=False):
        super(RecSystemPro, self).__init__()
        self.dataset = dataset
        if dataset == "redial":
            self.null_id = 30458
            self.node_total = 30471
        else:
            self.null_id = 19307
            self.node_total = 19308
        
        self.word_total = 29308
        self.use_bert = use_bert
        self.utter_enc_size = utter_enc_size
        self.dropout_rate = dropout_rate
        self.lambda1 = 0.1
        self.lambda2 = 0.01
        self.word_net = word_net
        
        self.max_turns = max_turns
        self.relation_count = relation_count
        self.base_count = base_count
        self.graph_enc_size = graph_enc_size
        self.attn_hidden = attn_hidden

        self.device = torch.device(device_str)
        self.neg_sample_ratio = neg_sample_ratio

        self.utter_encoder = UtteranceEncoder(rnn_type, use_bert, utter_enc_size, 
                                             dropout_rate, max_turns, word_net=word_net)
        self.graph_encoder = GraphEncoder(node_total=self.node_total, 
                                         enc_size=graph_enc_size, 
                                         device_str=device_str, word_net=word_net)
        self.intent_predictor = IntentPredictor(utter_enc_size, attn_hidden)
        self.graph_traverser = GraphTraverser(attention_hidden_dim=attn_hidden,
                                             graph_enc_size=graph_enc_size,
                                             utterance_enc_size=utter_enc_size,
                                             device_str=device_str,
                                             neg_sample_ratio=neg_sample_ratio,
                                             word_net=word_net)
        if self.dataset == "gorecdial":
            self.explicit_recommender = ExplicitRecommender(
                utterance_enc_size=utter_enc_size, 
                graph_enc_size=graph_enc_size
            )

        self.walk_loss_fn1 = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.walk_loss_fn2 = nn.BCEWithLogitsLoss(reduction="sum")
        self.intent_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.rec_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        self.align_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        self.align_word_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        self.Wa = nn.Linear(utter_enc_size, graph_enc_size, bias=False)
        self.Ww = nn.Linear(utter_enc_size, graph_enc_size, bias=False)


    def pretrain_forward(self, tokenized_dialog, seq_lengths, max_len, initial_hidden, 
                         edge_types, edge_indices, align_idx, align_batch_idx, 
                         align_labels, intent_labels, align_word_idx=None, 
                         align_word_batch_idx=None, align_word_labels=None):
        
        utter_encoding = self.utter_encoder.forward(tokenized_dialog, seq_lengths, max_len, initial_hidden)
        intent_pred = self.intent_predictor.forward(utter_encoding)
        graph_encoding, word_encoding = self.graph_encoder.forward(edge_types, edge_indices)
        graph_features = graph_encoding.index_select(0, align_idx)
        
        tiled_utter = self.graph_traverser.tile_context(utter_encoding, align_batch_idx)
        logits = torch.sum(self.Wa(tiled_utter) * graph_features, dim=-1)

        loss_align = self.align_loss_fn(logits, align_labels)
        loss_intent = self.intent_loss_fn(intent_pred, intent_labels)

        if self.word_net:
            tiled_utter_word = self.graph_traverser.tile_context(utter_encoding, align_word_batch_idx)
            word_features = word_encoding.index_select(0, align_word_idx)
            logits_w = torch.sum(self.Ww(tiled_utter_word) * word_features, dim=-1)
            loss_align_w = self.align_word_loss_fn(logits_w, align_word_labels)
            return loss_align + loss_align_w + loss_intent
        else:
            return loss_align + loss_intent

    
    def prepare_regularization(self, mention_history, dialog_history, intent=None, rec_candidates=None):
        align_indices = []
        align_batch_indices = []
        align_labels = []
        
        for batch_idx, items in enumerate(mention_history):
            mentioned_nodes = []
            for node in items:
                if rec_candidates is not None and node in rec_candidates[batch_idx]:
                    continue
                align_indices.append(node)
                align_batch_indices.append(batch_idx)
                align_labels.append(1)
                mentioned_nodes.append(node)
            positive_count = len(mentioned_nodes)

            if rec_candidates is not None:
                align_indices.append(rec_candidates[batch_idx][0])
                align_batch_indices.append(batch_idx)
                align_labels.append(1)

            if len(mentioned_nodes) == 0:
                positive_count += 1
                mentioned_nodes.append(self.null_id)
                align_indices.append(self.null_id)
                align_batch_indices.append(batch_idx)
                align_labels.append(1)
            else:
                align_indices.append(self.null_id)
                align_batch_indices.append(batch_idx)
                align_labels.append(0)
            
            sampled_nodes = []

            if rec_candidates is not None:
                for neg_idx in range(1, 5):
                    sampled_nodes.append(rec_candidates[batch_idx][neg_idx])
                    align_indices.append(rec_candidates[batch_idx][neg_idx])
                    align_batch_indices.append(batch_idx)
                    align_labels.append(0)
            for _ in range(4 * positive_count):
                candidate = random.sample(range(self.node_total), 1)[0]
                while candidate in mentioned_nodes or candidate in sampled_nodes:
                    candidate = random.sample(range(self.node_total), 1)[0]
                sampled_nodes.append(candidate)
                align_indices.append(candidate)
                align_batch_indices.append(batch_idx)
                align_labels.append(0)
        
        align_indices = torch.Tensor(align_indices).long().to(device=self.device)
        align_batch_indices = torch.Tensor(align_batch_indices).long().to(device=self.device)
        align_labels = torch.FloatTensor(align_labels).to(device=self.device)

        align_word_indices = None
        align_word_batch_indices = None
        align_word_labels = None

        if self.word_net:
            tokenized_dialog, seq_lengths, max_len, initial_hidden, word_idx, word_batch_idx, raw_history = \
                self.utter_encoder.prepare_data(dialog_history, self.device, raw_history=True)
            align_word_indices = []
            align_word_batch_indices = []
            align_word_labels = []
            for batch_idx, items in enumerate(raw_history):
                mentioned_words = []
                for word in items:
                    align_word_indices.append(word)
                    align_word_batch_indices.append(batch_idx)
                    align_word_labels.append(1)
                    mentioned_words.append(word)
                positive_count = len(mentioned_words)
                if len(mentioned_words) == 0:
                    positive_count += 1
                    mentioned_words.append(0)
                    align_word_indices.append(0)
                    align_word_batch_indices.append(batch_idx)
                    align_word_labels.append(1)
                
                for _ in range(positive_count):
                    candidate = random.sample(range(self.word_total), 1)[0]
                    while candidate in mentioned_words or candidate in sampled_nodes:
                        candidate = random.sample(range(self.word_total), 1)[0]
                    sampled_nodes.append(candidate)
                    align_word_indices.append(candidate)
                    align_word_batch_indices.append(batch_idx)
                    align_word_labels.append(0)
            align_word_indices = torch.Tensor(align_word_indices).long().to(device=self.device)
            align_word_batch_indices = torch.Tensor(align_word_batch_indices).long().to(device=self.device)
            align_word_labels = torch.FloatTensor(align_word_labels).to(device=self.device)
        
        return align_indices, align_batch_indices, align_labels, align_word_indices, align_word_batch_indices, align_word_labels

    
    def prepare_pretrain_data(self, mention_history, dialog_history, intent, edge_types, edge_indices, rec_candidates=None):
        align_indices = []
        align_batch_indices = []
        align_labels = []
        for batch_idx, items in enumerate(mention_history):
            mentioned_nodes = []
            for node in items:
                if rec_candidates is not None and node in rec_candidates[batch_idx]:
                    continue
                align_indices.append(node)
                align_batch_indices.append(batch_idx)
                align_labels.append(1)
                mentioned_nodes.append(node)
            positive_count = len(mentioned_nodes)

            if rec_candidates is not None:
                align_indices.append(rec_candidates[batch_idx][0])
                align_batch_indices.append(batch_idx)
                align_labels.append(1)

            if len(mentioned_nodes) == 0:
                positive_count += 1
                mentioned_nodes.append(self.null_id)
                align_indices.append(self.null_id)
                align_batch_indices.append(batch_idx)
                align_labels.append(1)
            else:
                align_indices.append(self.null_id)
                align_batch_indices.append(batch_idx)
                align_labels.append(0)
            
            sampled_nodes = []

            if rec_candidates is not None:
                for neg_idx in range(1, 5):
                    sampled_nodes.append(rec_candidates[batch_idx][neg_idx])
                    align_indices.append(rec_candidates[batch_idx][neg_idx])
                    align_batch_indices.append(batch_idx)
                    align_labels.append(0)
            for _ in range(4 * positive_count):
                candidate = random.sample(range(self.node_total), 1)[0]
                while candidate in mentioned_nodes or candidate in sampled_nodes:
                    candidate = random.sample(range(self.node_total), 1)[0]
                sampled_nodes.append(candidate)
                align_indices.append(candidate)
                align_batch_indices.append(batch_idx)
                align_labels.append(0)
        
        align_indices = torch.Tensor(align_indices).long().to(device=self.device)
        align_batch_indices = torch.Tensor(align_batch_indices).long().to(device=self.device)
        align_labels = torch.FloatTensor(align_labels).to(device=self.device)

        tokenized_dialog, seq_lengths, max_len, initial_hidden, word_idx, word_batch_idx, raw_history = \
            self.utter_encoder.prepare_data(dialog_history, self.device, raw_history=True)
        edge_indices = edge_indices.to(device=self.device)
        edge_types = edge_types.to(device=self.device)
        all_intent = ["chat", "question", "recommend"]
        batch_size = len(intent)
        intent_labels = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            intent_labels[i] = all_intent.index(intent[i])
        intent_labels = intent_labels.long()

        align_word_indices = None
        align_word_batch_indices = None
        align_word_labels = None

        if self.word_net:
            align_word_indices = []
            align_word_batch_indices = []
            align_word_labels = []
            for batch_idx, items in enumerate(raw_history):
                mentioned_words = []
                for word in items:
                    align_word_indices.append(word)
                    align_word_batch_indices.append(batch_idx)
                    align_word_labels.append(1)
                    mentioned_words.append(word)
                positive_count = len(mentioned_words)
                if len(mentioned_words) == 0:
                    positive_count += 1
                    mentioned_words.append(0)
                    align_word_indices.append(0)
                    align_word_batch_indices.append(batch_idx)
                    align_word_labels.append(1)
                
                for _ in range(positive_count):
                    candidate = random.sample(range(self.word_total), 1)[0]
                    while candidate in mentioned_words or candidate in sampled_nodes:
                        candidate = random.sample(range(self.word_total), 1)[0]
                    sampled_nodes.append(candidate)
                    align_word_indices.append(candidate)
                    align_word_batch_indices.append(batch_idx)
                    align_word_labels.append(0)
            align_word_indices = torch.Tensor(align_word_indices).long().to(device=self.device)
            align_word_batch_indices = torch.Tensor(align_word_batch_indices).long().to(device=self.device)
            align_word_labels = torch.FloatTensor(align_word_labels).to(device=self.device)

        return tokenized_dialog, seq_lengths, max_len, initial_hidden, edge_types, edge_indices, align_indices, align_batch_indices, align_labels, intent_labels, align_word_indices, align_word_batch_indices, align_word_labels

    
    def forward(self, tokenized_dialog, seq_lengths, max_len, initial_hidden, edge_types, edge_indices,
                mention_idx, mention_batch_idx, sel_indices, sel_batch_indices, sel_group_indices, 
                grp_batch_indices, last_indices, intent_indices, intent_labels, label1, label2, 
                score_masks, align_indices, align_batch_indices, align_labels, word_idx=None, 
                word_batch_idx=None, align_word_idx=None, align_word_batch_idx=None, align_word_labels=None):
        utter_encoding = self.utter_encoder.forward(tokenized_dialog, seq_lengths, max_len, initial_hidden)
        graph_encoding, word_encoding = self.graph_encoder.forward(edge_types, edge_indices)
        intent_pred = self.intent_predictor.forward(utter_encoding)
        paths = self.graph_traverser.forward(graph_encoding, utter_encoding, mention_idx, mention_batch_idx,
                                           sel_indices, sel_batch_indices, sel_group_indices, grp_batch_indices,
                                           last_indices, intent_indices, score_masks, word_embed=word_encoding,
                                           word_batch_idx=word_batch_idx, word_index=word_idx)
        
        loss_walk1 = self.walk_loss_fn1(paths[0], label1)
        loss_walk2 = self.walk_loss_fn2(paths[1], label2)
        loss_intent = self.intent_loss_fn(intent_pred, intent_labels)

        graph_features = graph_encoding.index_select(0, align_indices)
        tiled_utter = self.graph_traverser.tile_context(utter_encoding, align_batch_indices)
        logits = torch.sum(self.Wa(tiled_utter) * graph_features, dim=-1)
        loss_reg = self.align_loss_fn(logits, align_labels)
        
        if self.word_net:
            word_features = word_encoding.index_select(0, align_word_idx)
            tiled_utter_word = self.graph_traverser.tile_context(utter_encoding, align_word_batch_idx)
            logits_w = torch.sum(self.Ww(tiled_utter_word) * word_features, dim=-1)
            loss_reg += self.align_word_loss_fn(logits_w, align_word_labels)

        total_loss = loss_walk1 + loss_walk2 + loss_intent + 0.025 * loss_reg
        return intent_pred, paths, total_loss

    
    def forward_gorecdial(self, tokenized_dialog, seq_lengths, max_len, initial_hidden, edge_types, edge_indices,
                         mention_idx, mention_batch_idx, sel_indices, sel_batch_indices, sel_group_indices, 
                         grp_batch_indices, last_indices, intent_indices, bow_encoding, rec_idx, rec_batch_idx, 
                         rec_golden, intent_labels, label1, label2, score_masks, align_indices, align_batch_indices, 
                         align_labels, word_idx=None, word_batch_idx=None, align_word_idx=None, align_word_batch_idx=None, align_word_labels=None):

        utter_encoding = self.utter_encoder.forward(tokenized_dialog, seq_lengths, max_len, initial_hidden)
        graph_encoding, word_encoding = self.graph_encoder.forward(edge_types, edge_indices)
        intent_pred = self.intent_predictor.forward(utter_encoding)
        paths, user_profile = self.graph_traverser.forward(graph_encoding, utter_encoding, mention_idx, mention_batch_idx,
                                                         sel_indices, sel_batch_indices, sel_group_indices, grp_batch_indices,
                                                         last_indices, intent_indices, score_masks, ret_portrait=True,
                                                         word_embed=word_encoding, word_batch_idx=word_batch_idx, word_index=word_idx)

        rec_pred = self.explicit_recommender.forward(utter_encoding, graph_encoding, user_profile, bow_encoding, rec_idx, rec_batch_idx)

        loss_walk1 = self.walk_loss_fn1(paths[0], label1)
        loss_walk2 = self.walk_loss_fn2(paths[1], label2)
        loss_intent = self.intent_loss_fn(intent_pred, intent_labels)
        loss_rec = self.rec_loss_fn(rec_pred, rec_golden)

        graph_features = graph_encoding.index_select(0, align_indices)
        tiled_utter = self.graph_traverser.tile_context(utter_encoding, align_batch_indices)
        logits = torch.sum(self.Wa(tiled_utter) * graph_features, dim=-1)
        loss_reg = self.align_loss_fn(logits, align_labels)
        
        if self.word_net:
            word_features = word_encoding.index_select(0, align_word_idx)
            tiled_utter_word = self.graph_traverser.tile_context(utter_encoding, align_word_batch_idx)
            logits_w = torch.sum(self.Ww(tiled_utter_word) * word_features, dim=-1)
            loss_reg += self.align_word_loss_fn(logits_w, align_word_labels)

        total_loss = loss_walk1 + loss_walk2 + loss_intent + loss_rec + 0.025 * loss_reg
        return intent_pred, paths, total_loss



    def predict_intent(self, tokenized_dialog, seq_lengths, max_len, initial_hidden):
        utter_encoding = self.utter_encoder.forward(tokenized_dialog, seq_lengths, max_len, initial_hidden)
        intent_pred = self.intent_predictor.forward(utter_encoding)
        return intent_pred
        

    def inference_gorecdial(self, intent, tokenized_dialog, seq_lengths, max_len, initial_hidden, edge_types, edge_indices,
                          mention_idx, mention_batch_idx, sel_idx, sel_batch_idx, sel_group_idx, grp_batch_idx, 
                          last_idx, score_mask, word_idx, word_batch_idx, bow_encoding=None, sel_idx_ex=None, 
                          sel_batch_idx_ex=None, last_weights=None, layer=0):
        utter_encoding = self.utter_encoder.forward(tokenized_dialog, seq_lengths, max_len, initial_hidden)
        graph_encoding, word_encoding = self.graph_encoder.forward(edge_types, edge_indices)

        user_profile = self.graph_traverser.get_user_portrait(mention_idx, mention_batch_idx, graph_encoding, word_idx, word_batch_idx, word_encoding)

        step, weight, partial_score = self.graph_traverser.forward_single_layer(layer, utter_encoding, user_profile,
                                                                              graph_encoding, sel_idx, sel_batch_idx,
                                                                              sel_group_idx, grp_batch_idx, last_idx,
                                                                              intent, score_mask, last_weights, True)
        if bow_encoding is not None:
            rec_pred = self.explicit_recommender.forward(utter_encoding, graph_encoding, user_profile, bow_encoding, sel_idx_ex, sel_batch_idx_ex)
            return step, weight, partial_score, rec_pred
        else:
            return step, weight, partial_score
    
    def inference_redial(self, intent, tokenized_dialog, seq_lengths, max_len, initial_hidden, edge_types, edge_indices,
                        mention_idx, mention_batch_idx, sel_idx, sel_batch_idx, sel_group_idx, grp_batch_idx, 
                        last_idx, score_mask, word_idx, word_batch_idx, last_weights=None, layer=0):
        utter_encoding = self.utter_encoder.forward(tokenized_dialog, seq_lengths, max_len, initial_hidden)
        graph_encoding, word_encoding = self.graph_encoder.forward(edge_types, edge_indices)

        user_profile = self.graph_traverser.get_user_portrait(mention_idx, mention_batch_idx, graph_encoding, word_idx, word_batch_idx, word_encoding)

        step, weight, partial_score = self.graph_traverser.forward_single_layer(layer, utter_encoding, user_profile,
                                                                              graph_encoding, sel_idx, sel_batch_idx,
                                                                              sel_group_idx, grp_batch_idx, last_idx,
                                                                              intent, score_mask, last_weights, ret_partial_score=True)
        return step, weight, partial_score

    def prepare_data_redial(self, dialog_history, mention_history, intent, node_candidate1, node_candidate2, edge_types, edge_indices, label1, label2, gold_pos, attribute_dict, sample=False):
        tokenized_dialog, seq_lengths, max_len, initial_hidden, word_idx, word_batch_idx = \
            self.utter_encoder.prepare_data(dialog_history, self.device)
        mention_idx, mention_batch_idx, sel_indices, sel_batch_indices, sel_group_indices, grp_batch_indices, last_indices, intent_indices, label_1, label_2, score_masks = \
            self.graph_traverser.prepare_data(mention_history, intent, node_candidate1, node_candidate2, label1, label2,
                                            attribute_dict, self.device, gold_pos, sample=sample, dataset="redial")
        edge_indices = edge_indices.to(device=self.device)
        edge_types = edge_types.to(device=self.device)
        all_intent = ["chat", "question", "recommend"]
        
        batch_size = len(intent)
        intent_labels = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            intent_labels[i] = all_intent.index(intent[i])
        intent_labels = intent_labels.long()

        return tokenized_dialog, seq_lengths, max_len, initial_hidden, edge_types, edge_indices, mention_idx, mention_batch_idx, sel_indices, sel_batch_indices, sel_group_indices, grp_batch_indices, last_indices, intent_indices, intent_labels, label_1, label_2, score_masks, word_idx, word_batch_idx
    

    def prepare_data_interactive(self, dialog_history, mention_history, edge_types, edge_indices):
        tokenized_dialog, seq_lengths, max_len, initial_hidden, _, _ = self.utter_encoder.prepare_data(dialog_history, self.device)

        edge_indices = edge_indices.to(device=self.device)
        edge_types = edge_types.to(device=self.device)
        mention_idx = []
        mention_batch_idx = []
        for batch_idx, items in enumerate(mention_history):
            mentioned_nodes = []
            for node in items:
                mention_idx.append(node)
                mention_batch_idx.append(batch_idx)
                mentioned_nodes.append(node)
            if len(mentioned_nodes) == 0:
                mention_idx.append(30458)
                mention_batch_idx.append(batch_idx)
        mention_idx = torch.Tensor(mention_idx).long().to(device=self.device)
        mention_batch_idx = torch.Tensor(mention_batch_idx).long().to(device=self.device)
        return tokenized_dialog, seq_lengths, max_len, initial_hidden, edge_types, edge_indices, mention_idx, mention_batch_idx
           
    def prepare_data_gorecdial(self, dialog_history, mention_history, intent, node_candidate1, node_candidate2, edge_types, edge_indices, label1, label2, attribute_dict, rec_candidates, all_bows):
        tokenized_dialog, seq_lengths, max_len, initial_hidden, word_idx, word_batch_idx = \
            self.utter_encoder.prepare_data(dialog_history, self.device)
        mention_idx, mention_batch_idx, sel_indices, sel_batch_indices, sel_group_indices, grp_batch_indices, last_indices, intent_indices, label_1, label_2, score_masks = \
            self.graph_traverser.prepare_data(mention_history, intent, node_candidate1, node_candidate2, label1, label2,
                                            attribute_dict, self.device, sample=False, dataset="gorecdial")
        edge_indices = edge_indices.to(device=self.device)
        rec_idx, rec_batch_idx, rec_golden = self.explicit_recommender.prepare_data(rec_candidates, self.device)
        edge_types = edge_types.to(device=self.device)
        all_intent = ["chat", "question", "recommend"]

        batch_size = len(intent)
        intent_labels = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            intent_labels[i] = all_intent.index(intent[i])
        intent_labels = intent_labels.long()

        bow_encoding = all_bows.to(device=self.device)

        return tokenized_dialog, seq_lengths, max_len, initial_hidden, edge_types, edge_indices, mention_idx, mention_batch_idx, sel_indices, sel_batch_indices, sel_group_indices, sel_group_indices, grp_batch_indices, last_indices, intent_indices, bow_encoding, rec_idx, rec_batch_idx, rec_golden, intent_labels, label_1, label_2, score_masks, word_idx, word_batch_idx


    def prepare_rectest_data(self, rec_candidates):
        golden_labels = []
        batch_indices = []
        selected_nodes = []
        shuffled_candidates = []
        for batch_idx, items in enumerate(rec_candidates):
            correct_node = items[0]
            item_copy = copy.deepcopy(items)
            random.shuffle(item_copy)
            golden_labels.append(item_copy.index(correct_node))
            shuffled_candidates.append(item_copy)
            for node in item_copy:
                batch_indices.append(batch_idx)
                selected_nodes.append(node)
        rec_idx = torch.Tensor(selected_nodes).long().to(device=self.device)
        batch_idx_tensor = torch.Tensor(batch_indices).long().to(device=self.device)
        batch_size = len(rec_candidates)

        intent_idx = []
        for i in range(batch_size):
            intent_idx.append(2)
        intent_idx = torch.Tensor(intent_idx).long().to(device=self.device)
        
        return intent_idx, rec_idx, batch_idx_tensor, shuffled_candidates, golden_labels
