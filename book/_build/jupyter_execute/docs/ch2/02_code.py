#!/usr/bin/env python
# coding: utf-8

# # Vision Transformer Code

# ## Patch embedding

# In[ ]:


class patch_embedding(nn.Module) :
    def __init__(self, patch_size, img_size, embed_size) :
        super(patch_embedding, self).__init__()
        
        self.patch_embedding = nn.Conv2d(3, embed_size, 
                                         kernel_size=patch_size, 
                                         stride=patch_size)
        # cls token을 패치 앞에 하나 더 붙여줌
        self.cls_token = nn.Parameter(torch.rand(1,1,embed_size))
        
        # cls token 1개가 더 붙었기 때문에 총 patch 개수에 + 1을 해줌
        self.position = nn.Parameter(torch.rand((img_size//patch_size)**2 + 1, embed_size))
    
    def forward(self, x) :
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(2,1)

        ct = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([ct, x],dim=1)
        x += self.position
        return x


# ## Multi-head Attention

# In[ ]:


class multi_head_attention(nn.Module) :
    def __init__(self, embed_size, num_head, dropout_rate=0.1) :
        super(multi_head_attention, self).__init__()
        
        self.q = nn.Linear(embed_size, embed_size)
        self.k = nn.Linear(embed_size, embed_size)
        self.v = nn.Linear(embed_size, embed_size)
        
        self.fc = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.num_head = num_head
        self.embed_size = embed_size
    
    def qkv_reshape(self, value, num_head) :
        b, n, emb = value.size()
        dim = emb // num_head
        return value.view(b, num_head, n, dim)
        
    def forward(self, x) :
        q = self.qkv_reshape(self.q(x), self.num_head)
        k = self.qkv_reshape(self.k(x), self.num_head)
        v = self.qkv_reshape(self.v(x), self.num_head)
        
        qk = torch.matmul(q, k.transpose(3,2))
        att = F.softmax(qk / (self.embed_size ** (1/2)), dim=-1)
        att = torch.matmul(att, v)
        
        b, h, n, d = att.size()
        x = att.view(b, n, h*d)
        x = self.fc(x)
        x = self.dropout(x)
        return x


# ## MLP

# In[ ]:


class MLP(nn.Module) :
    def __init__(self, embed_size, expansion, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_size, embed_size*expansion)
        self.fc2 = nn.Linear(embed_size*expansion, embed_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x) :
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ## Encoder Block

# In[ ]:


class EncoderBlock(nn.Module) :
    def __init__(self, 
                 embed_size, 
                 num_head, 
                 expansion, 
                 dropout_rate):
        super(EncoderBlock, self).__init__()
        
        self.skip_connection1 = skip_connection(
            nn.Sequential(
                nn.LayerNorm(embed_size),
                multi_head_attention(embed_size, num_head, dropout_rate=0.1)
            )
        )
        
        self.skip_connection2 = skip_connection(
            nn.Sequential(
                nn.LayerNorm(embed_size),
                MLP(embed_size, expansion, dropout_rate=0.1)
            )
        )
    
    def forward(self, x) :
        x = self.skip_connection1(x)
        x = self.skip_connection2(x)
        return x

class skip_connection(nn.Module) :
	def __init__(self, layer):
		super(skip_connection, self).__init__()
		self.layer = layer
	
	def forward (self, x):
		return self.layer(x) + x


# ## Classifier Head

# In[ ]:


class Classifier_Head(nn.Module) :
    def __init__(self, embed_size, num_classes):
        super(Classifier_Head, self).__init__()
        
        self.avgpool1d = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

	  def forward(self, x) :
        x = x.transpose(2,1)
        x = self.avgpool1d(x).squeeze(2)
        x = self.fc(x)
        return x


# ## ViT

# In[ ]:


class VIT(nn.Module) :
    def __init__(self, 
                 patch_size=16, 
                 img_size=224, 
                 embed_size=768, 
                 num_head = 8,
                 expansion = 4,
                 dropout_rate = 0.1,
                 encoder_depth = 12,
                 num_classes = 10) :
        super(VIT, self).__init__()

        self.PatchEmbedding = patch_embedding(patch_size, img_size, embed_size)
        self.EncoderBlocks = self.make_layers(encoder_depth, embed_size, num_head, expansion, dropout_rate)
        self.ClassifierHead = Classifier_Head(embed_size, num_classes)
        
    def make_layers(self, encoder_depth, *args):
        layers = []
        for _ in range(0, encoder_depth) :
            layers.append(EncoderBlock(*args))
        return nn.Sequential(*layers)
    
    def forward(self, x) :
        x = self.PatchEmbedding(x)
        x = self.EncoderBlocks(x)
        x = self.ClassifierHead(x)
        
        return x


# Author by `임중섭`  
# Edit by `김주영`
