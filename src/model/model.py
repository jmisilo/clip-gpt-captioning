import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer

class ImageEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super(ImageEncoder, self).__init__()
        
        self.device = device

        self.preprocessor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').vision_model.to(self.device)

    def forward(self, image):
        # only one image at a time
        image = self.preprocessor(images=image, return_tensors='pt').to(self.device)
        image_features = self.model(**image)
        
        return image_features.pooler_output

class Mapping(nn.Module):
    def __init__(
        self, 
        num_layers,
        embed_size, 
        n_heads, 
        forward_expansion, 
        dropout, 
        device='cpu'
    ):
        super(Mapping, self).__init__()
        
        self.device = device

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size, 
                nhead=n_heads, 
                dim_feedforward=embed_size*forward_expansion, 
                dropout=dropout, 
                batch_first=True, 
                device=device
            ),
            num_layers=num_layers
        ).to(self.device)

        self.init_weights()

    def forward(self, img_embedded):
        return self.transformer_encoder(img_embedded)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class TextDecoder(nn.Module):
    def __init__(self, device='cpu'):
        super(TextDecoder, self).__init__()
        
        self.device = device
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, embedding, attention_mask=None):
        text_features = self.model(inputs_embeds=embedding, attention_mask=None)
        
        return text_features.logits

class Net(nn.Module):
    def __init__(self, num_layers, n_heads, forward_expansion, dropout, max_len, device='cpu'):
        '''
            num_layers: number of layers in the TransformerEncoder
            n_heads: number of heads in the MultiHeadAttention
            forward_expansion: expansion factor for the feedforward layer
            dropout: dropout probability
            max_len: maximum length of the generated text
        '''
        super(Net, self).__init__()

        self.ie = ImageEncoder(device=device)
        self.mp = Mapping(num_layers=num_layers, embed_size=self.ie.model.config.hidden_size, n_heads=n_heads, forward_expansion=forward_expansion, dropout=dropout, device=device)
        self.td = TextDecoder(device=device)
        
        assert self.ie.model.config.hidden_size == self.td.model.config.n_embd, "Embedding size of models mismatch"

        self.max_len = max_len

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.td.tokenizer.pad_token_id)

        self.freeze_layers()

    def freeze_layers(self):
        for p in *list(self.ie.parameters()), *list(self.td.parameters())[14:-14]: # freeze everything, except 1st and last transformer layer in Decoder
            p.requires_grad = False

    def forward(self, img):
        # only one image at a time

        with torch.no_grad():
            img_embedded = self.ie(img)

            # (1, embed_size)
            img_mapped = self.mp(img_embedded)
            sos_emb = self.td.model.transformer.wte(torch.tensor(self.td.tokenizer.bos_token_id))
            
            # sos_emb shape embed_size -> (1, embed_size)
            sos_emb = sos_emb.expand_as(img_mapped)

            # (2, embed_size)
            start_emb = torch.cat([sos_emb, img_mapped], dim=0)

            tokens = []
            for _ in range(self.max_len):
                if len(tokens):
                    tok_emb = self.td.model.transformer.wte(torch.tensor(tokens))

                    emb = torch.cat([start_emb, tok_emb], dim=0)
                else:
                    emb = start_emb

                # add positional enc
                pos_emb = self.td.model.transformer.wpe(torch.arange(emb.shape[0]).to(self.td.device))

                emb += pos_emb
                pred = self.td(emb)

                _, pred = torch.max(pred, dim=1)

                last_token = pred[-1].item()

                tokens.append(last_token)

                if last_token == self.td.tokenizer.eos_token_id:
                    break

            decoded = self.td.tokenizer.decode(tokens)
            
            return decoded, tokens

    def train_forward(self, img_emb, trg_cap, att_mask):
        # method should get embedded by CLIP images and trg_text without last token.
        # dataset should contain image, embedded image, text

        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]
        y = trg_cap[:, 1:]

        # img_emb - (N, embed_size)
        # trg_capt - (N, len)
        img_mapped = self.mp(img_emb)

        # N, 1, embed_size
        img_mapped = img_mapped.unsqueeze(1)

        # embed all texts and con cat with map sos
        text_emb = self.td.model.transformer.wte(x)

        # N, len, embed_size
        x = torch.concat([img_mapped, text_emb], dim=1)

        pos_emb = self.td.model.transformer.wpe(torch.arange(x.shape[1]).to(self.td.device))
        pos_emb = pos_emb.expand_as(x)

        x += pos_emb

        # N, len, vocab_size
        res = self.td(x, attention_mask=x_mask)
        res = torch.softmax(res, dim=2)

        loss = self.criterion(res[:, 1:, :].reshape(-1, res.shape[-1]), y.reshape(-1))
        
        return loss

if __name__ == '__main__':
    
    m = Net(
        num_layers=6,
        n_heads=16, 
        forward_expansion=4, 
        dropout=0.1, 
        max_len=20
    )

    r = m(torch.randn(3, 224, 224))
    print(r)
    l = m.train_forward(torch.rand(10, 768), torch.randint(1, 50000, (10, 20)))
    print(l)