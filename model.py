import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class transformer_fusion(nn.Module):
    def __init__(self, seq_len, num_class_verb, num_class_noun, num_class_action, feat_in, hidden, dropout=0.5):
        super(transformer_fusion, self).__init__()
        self.layer_norm_11 = nn.LayerNorm(feat_in)
        self.layer_norm_12 = nn.LayerNorm(feat_in)
        self.layer_norm_13 = nn.LayerNorm(352)

        self.pos_encoder = PositionalEncoding(d_model=feat_in, max_len=seq_len)
        self.pos_encoder_2 = PositionalEncoding(d_model=feat_in*2+352, max_len=seq_len)
        self.pos_encoder_3 = PositionalEncoding(d_model=(feat_in*2+352)*2, max_len=seq_len)

        self.pos_encoder_obj = PositionalEncoding(d_model=352, max_len=seq_len)

        encoder_layers = TransformerEncoderLayer(d_model=feat_in, nhead=8, dim_feedforward=hidden,
                                                 dropout=dropout, )
        encoder_layers_obj = TransformerEncoderLayer(d_model=352, nhead=8, dim_feedforward=hidden,
                                                 dropout=dropout, )
        encoder_layers_2 = TransformerEncoderLayer(d_model=feat_in*2+352, nhead=8, dim_feedforward=feat_in*2+352,
                                                 dropout=dropout, )
        encoder_layers_3 = TransformerEncoderLayer(d_model=(feat_in*2+352)*2, nhead=8,
                                                 dropout=dropout, )

        # transformer for rgb
        self.transformer_1a_v = TransformerEncoder(encoder_layers, 1)
        # transformer for flow
        self.transformer_1b_v = TransformerEncoder(encoder_layers, 1)
        # # transformer for hidden
        self.transformer_1c_v = TransformerEncoder(encoder_layers_obj, 1)
        # transformer for fusion
        self.transformer_1f_v = TransformerEncoder(encoder_layers_2, 1)

        # transformer for rgb
        self.transformer_1a_n = TransformerEncoder(encoder_layers, 1)
        # transformer for flow
        self.transformer_1b_n = TransformerEncoder(encoder_layers, 1)
        # # transformer for hidden
        self.transformer_1c_n = TransformerEncoder(encoder_layers_obj, 1)
        # transformer for fusion
        self.transformer_1f_n = TransformerEncoder(encoder_layers_2, 1)

        # transformer for rgb
        self.transformer_2a_v = TransformerEncoder(encoder_layers, 1)
        # transformer for flow
        self.transformer_2b_v = TransformerEncoder(encoder_layers, 1)
        # # transformer for hidden
        self.transformer_2c_v = TransformerEncoder(encoder_layers_obj, 1)
        # transformer for fusion
        self.transformer_2f_v = TransformerEncoder(encoder_layers_2, 1)

        # transformer for rgb
        self.transformer_2a_n = TransformerEncoder(encoder_layers, 1)
        # transformer for flow
        self.transformer_2b_n = TransformerEncoder(encoder_layers, 1)
        # # transformer for hidden
        self.transformer_2c_n = TransformerEncoder(encoder_layers_obj, 1)
        # transformer for fusion
        self.transformer_2f_n = TransformerEncoder(encoder_layers_2, 1)

        # transformer for first block
        self.transformer_1_cross = TransformerEncoder(encoder_layers_2, 1)
        self.transformer_2_cross = TransformerEncoder(encoder_layers_2, 1)
        self.transformer_3_cross = TransformerEncoder(encoder_layers_3, 1)
        self.transformer_4_cross = TransformerEncoder(encoder_layers_3, 1)


        self.classifier_verb = nn.Linear(hidden*2+352, num_class_verb)
        self.classifier_noun = nn.Linear(hidden*2+352, num_class_noun)
        #self.classifier_action_fusion = nn.Linear(hidden, num_class_action)
        self.classifier_action = nn.Linear((hidden*2+352)*2, num_class_action)

        self.dropout_verb = torch.nn.Dropout(dropout)
        self.dropout_noun = torch.nn.Dropout(dropout)
        self.dropout_action = torch.nn.Dropout(dropout)

        self.classifier_verb_s1 = nn.Linear(hidden*2+352, num_class_verb)
        self.classifier_noun_s1 = nn.Linear(hidden*2+352, num_class_noun)
        #self.classifier_action_fusion = nn.Linear(hidden, num_class_action)
        self.classifier_action_s1 = nn.Linear((hidden*2+352)*2, num_class_action)

        self.dropout_verb_s1 = torch.nn.Dropout(dropout)
        self.dropout_noun_s1 = torch.nn.Dropout(dropout)
        self.dropout_action_s1 = torch.nn.Dropout(dropout)

    def forward(self, x1, x2, x3):
        B, L, _ = x1.shape

        x1 = x1.permute([1, 0, 2])
        x2 = x2.permute([1, 0, 2])
        x3 = x3.permute([1, 0, 2])

        x1 = self.layer_norm_11(x1)
        x2 = self.layer_norm_12(x2)
        x3 = self.layer_norm_13(x3)

        ######### Block 1 ##########
        x1 = self.pos_encoder(x1)
        x2 = self.pos_encoder(x2)
        x3 = self.pos_encoder_obj(x3)

        # v branch
        x1_v = self.transformer_1a_v(x1)
        x2_v = self.transformer_1b_v(x2)
        x3_v = self.transformer_1c_v(x3)

        xf_v = self.pos_encoder_2(torch.cat((x1_v, x2_v, x3_v), dim=-1))
        #xf_v = self.pos_embedding_3 + torch.cat((x1_v, x2_v, x3_v), dim=0)  #  3*L * B * C
        xf_v = self.transformer_1f_v(xf_v)
        xf_v_s1 = xf_v

        # n branch
        x1_n = self.transformer_1a_n(x1)
        x2_n = self.transformer_1b_n(x2)
        x3_n = self.transformer_1c_n(x3)
        #xf_n = self.pos_embedding_3 + torch.cat((x1_n, x2_n, x3_n), dim=0)
        xf_n = self.pos_encoder_2(torch.cat((x1_n, x2_n, x3_n), dim=-1))
        xf_n = self.transformer_1f_n(xf_n)
        xf_n_s1 = xf_n

        # cross branch
        xf_vn = (torch.cat((self.pos_encoder_2(xf_v), self.pos_encoder_2(xf_n)), dim=0))

        # print(xf_n.shape)
        # print(xf_vn.shape)
        xf_vn = self.transformer_1_cross(xf_vn) # 2*L * B * 2*C

        x1_v = xf_vn[:L, :, :1024]
        x2_v = xf_vn[:L, :, 1024:1024*2]
        x3_v = xf_vn[:L, :, 1024*2:]

        x1_n = xf_vn[L:, :, :1024]
        x2_n = xf_vn[L:, :, 1024:1024*2]
        x3_n = xf_vn[L:, :, 1024*2:]

        xf_vn_s1 = torch.cat((xf_vn[:L, :, :], xf_vn[L:, :, :]), dim=-1)
        xf_vn_s1 = self.transformer_3_cross(self.pos_encoder_3(xf_vn_s1))

        ########## Block 2 ##########
        # v branch
        #x1_v = self.layer_norm_21_v(x1_v)
        #x2_v = self.layer_norm_22_v(x2_v)
        #x3_v = self.layer_norm_23_v(x3_v)

        x1_v = self.pos_encoder(x1_v)
        x2_v = self.pos_encoder(x2_v)
        x3_v = self.pos_encoder_obj(x3_v)

        x1_v = self.transformer_2a_v(x1_v)
        x2_v = self.transformer_2b_v(x2_v)
        x3_v = self.transformer_2c_v(x3_v)

        xf_v = self.pos_encoder_2(torch.cat((x1_v, x2_v, x3_v), dim=-1))
        xf_v = self.transformer_2f_v(xf_v)

        # n branch
        #x1_n = self.layer_norm_21_n(x1_n)
        #x2_n = self.layer_norm_22_n(x2_n)
        #x3_n = self.layer_norm_23_n(x3_n)

        x1_n = self.pos_encoder(x1_n)
        x2_n = self.pos_encoder(x2_n)
        x3_n = self.pos_encoder_obj(x3_n)

        x1_n = self.transformer_2a_n(x1_n)
        x2_n = self.transformer_2b_n(x2_n)
        x3_n = self.transformer_2c_n(x3_n)

        xf_n = self.pos_encoder_2(torch.cat((x1_n, x2_n, x3_n), dim=-1))
        xf_n = self.transformer_2f_n(xf_n)

        # cross branch
        xf_vn = torch.cat((self.pos_encoder_2(xf_v), self.pos_encoder_2(xf_n)), dim=0)
        xf_vn = self.transformer_2_cross(xf_vn)  # 2*L * B * 3*C

        xf_vn = torch.cat((xf_vn[:L, :, :], xf_vn[L:, :, :]), dim=-1)
        xf_vn = self.transformer_4_cross(self.pos_encoder_3(xf_vn))

        ########### Prediction ###############
        prob_v_s1 = self.classifier_verb(self.dropout_verb_s1(xf_v_s1).view(B * L, -1)).reshape(L, B, -1)
        prob_v_s1 = torch.mean(prob_v_s1, dim=0, keepdim=False)  # problem may come here!!!
        #
        prob_n_s1 = self.classifier_noun(self.dropout_noun_s1(xf_n_s1).view(B * L, -1)).reshape(L, B, -1)
        prob_n_s1 = torch.mean(prob_n_s1, dim=0, keepdim=False)

        prob_a_s1 = self.classifier_action(self.dropout_action_s1(xf_vn_s1).view(B * L, -1)).reshape(L, B, -1)
        prob_a_s1 = torch.mean(prob_a_s1, dim=0, keepdim=False)

        prob_v = self.classifier_verb(self.dropout_verb(xf_v).view(B * L, -1)).reshape(L, B, -1)
        prob_v = torch.mean(prob_v, dim=0, keepdim=False) # problem may come here!!!
        #
        prob_n = self.classifier_noun(self.dropout_noun(xf_n).view(B * L, -1)).reshape(L, B, -1)
        prob_n = torch.mean(prob_n, dim=0, keepdim=False)
        #
        prob_a = self.classifier_action(self.dropout_action(xf_vn).view(B * L, -1)).reshape(L, B, -1)
        prob_a = torch.mean(prob_a, dim=0, keepdim=False)

        return prob_v, prob_n, prob_a, prob_v_s1, prob_n_s1, prob_a_s1