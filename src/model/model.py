import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, GPT2Tokenizer, GPT2LMHeadModel

class ImageEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super(ImageEncoder, self).__init__()
        
        self.device = device

        self.preprocessor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').vision_model.to(self.device)

    def forward(self, image):
        # check size
        # if more than 1 image - use self.processor.batch_decode
        image = self.preprocessor(images=image, return_tensors='pt').to(self.device)
        image_features = self.model(**image)
        
        return image_features.pooler_output

class Mapping(nn.Module):
    def __init__(self, embed_size, device='cpu'):
        super(Mapping, self).__init__()
        
        self.device = device

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=16),
            num_layers=6
        ).to(self.device)

    def forward(self, img_embedded):
        return self.transformer_encoder(img_embedded)

class TextDecoder(nn.Module):
    def __init__(self, device='cpu'):
        super(TextDecoder, self).__init__()
        
        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)

    def forward(self, embedding):
        # text = self.tokenizer(text, padding=True, return_tensors='pt').to(self.device)
        text_features = self.model(inputs_embeds=embedding)
        
        return text_features.logits

class Net(nn.Module):
    def __init__(self, max_len):
        super(Net, self).__init__()
       

        self.ie = ImageEncoder()
        self.mp = Mapping(self.ie.model.config.hidden_size)
        self.td = TextDecoder()

        assert self.ie.model.config.hidden_size == self.td.model.config.n_embd, "Embedding size of models mismatch"

        self.max_len = max_len

    def forward(self, img):

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

        return tokens

    def train(self, img, trg_capt):
        # method should get embedded by CLIP images and trg_text without last token.
        # dataset should contain image, embedded image, text

        # img - N, 3, 224, 224
        img_embedded = self.ie.model(pixel_values=img)

        print(img_embedded.shape)
        img_mapped = self.mp(img_embedded)
        print(img_mapped.shape)

        # N, 1, embed_size
        img_mapped = img_mapped.unsqueeze(1)
        print(img_mapped.shape)

        sos_emb = self.td.model.transformer.wte(torch.tensor(self.td.tokenizer.bos_token_id))
        print(sos_emb.shape)
        sos_emb = sos_emb.expand_as(img_mapped)
        print(sos_emb.shape)

        # embed all texts and con cat with map sos

        x = torch.concat([sos_emb, img_mapped], dim=1)
        print(x.shape)
        res = self.td(x)
        print(res.shape)
        return torch.softmax(res, dim=2)