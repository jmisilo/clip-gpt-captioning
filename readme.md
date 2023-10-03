# CLIPxGPT Captioner

## Description

**`CLIPxGPT Captioner`** is Image Captioning Model based on [OpenAI's](https://openai.com/) [CLIP](https://openai.com/blog/clip/) and [GPT-2](https://openai.com/blog/better-language-models/). The Model uses a Mapping module to "translate" CLIP embeddings ​​to GPT-2. The model is trained on the [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) dataset, downloaded from [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

**The goal** of the project was to find out about the possibility of CLIP + GPT-2 connection and to check whether, with a relatively short training time and a small dataset, the model will be able to recognize situations in the pictures. The model achieved satisfactory results.

The Model uses prefixes as in the [ClipCap](https://arxiv.org/abs/2111.09734) paper. In my original idea, the length of the prefix was 1, but after reading publication, the length of the prefix was changed to 4, thanks to which the performance increased.

The Model was trained with a frozen CLIP, a fully trained Mapping Module (5-6x Transformer Encoder Layers) and with partially frozen GPT-2 (the first and last 14 layers were trained).

The training process was carried out using the [Kaggle](https://www.kaggle.com/) P100 GPU.

### Model Versions

> **Small** - [Download](https://drive.google.com/file/d/1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF/view?usp=sharing)
>
> - Text Model - GPT-2 Small - 124M parameters
> - Mapping Module - 6x Transformer Encoder Layers
> - CLIP Base - Patch 32 model
> - 256M Parameters

> **Large** - [Download](https://drive.google.com/file/d/1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG/view?usp=sharing)
>
> - Text Model - GPT-2 Medium - 355M parameters
> - Mapping Module - 5x Transformer Encoder Layers
> - CLIP Large - Patch 14 model
> - 736M Parameters

## Example results

![Example1](./examples/23012796-RS.jpg)
![Example2](./examples/36979-RS.jpg)
![Example3](./examples/89407459-RL.jpg)

## Usage

Clone repository using:

```bash
git clone https://github.com/jmisilo/clip-gpt-captioning

cd clip-gpt-captioning
```

Create environment and install requirements:

```bash
python -m venv venv
# for windows
.\venv\Scripts\activate
# for linux/mac
source venv/bin/activate

pip install -r requirements.txt
```

And run prediction:

```bash
python .\src\predict.py -I <image_path> -S <model_size [S/L]> -C <checkpoint_name>
```

### References:

- [OpenAI](https://openai.com/)
- [CLIP](https://openai.com/blog/clip/)
- [GPT-2](https://openai.com/blog/better-language-Models/)
- [ClipCap; Mokady, Ron and Hertz, Amir and Bermano, Amit H](https://arxiv.org/abs/2111.09734)
- [HuggingFace](https://huggingface.co/)
