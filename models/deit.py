import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer

class DeiT_S_16(nn.Module):
    def __init__(self, num_classes=2):
        super(DeiT_S_16, self).__init__()
        # Initialize ViT model from timm
        self.model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=num_classes)
        self.num_classes = num_classes
        self.embed_dim = self.model.embed_dim
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        self.feature_size = 384
        
    def intermediate_forward(self, x):
        x = self.model.patch_embed(x)
        n = x.shape[0]
        cls_tokens = self.model.cls_token.expand(n, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        
        # Extract intermediate features
        features = []
        for blk in self.model.blocks:
            x = blk(x)
            features.append(x)
        
        x = self.model.norm(x)
        return x, features  # Return final output and intermediate features

    def forward(self, x, return_feature=False, return_feature_list=False):
        if return_feature_list:
            x, feature_list = self.intermediate_forward(x)
        # Reshape and permute the input tensor
        else:
            x = self.model.forward_features(x)

        classifier_output = self.model.head(x[:, 0])  # Assuming classifier token is the first token
        
        if return_feature:
            return classifier_output, x[:, 0]
        if return_feature_list:
            return classifier_output, feature_list
        else:
            return classifier_output

    def forward_threshold(self, x, threshold):
        # Reshape and permute the input tensor
        x = self.model.forward_features(x)
        feature = x.clip(max=threshold)
        logits_cls = self.model.head(feature)

        return logits_cls

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head
  
    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))

    