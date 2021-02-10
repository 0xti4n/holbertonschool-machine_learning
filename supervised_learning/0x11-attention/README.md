# Attention

## Specializations - Machine Learning ― Supervised Learning

## Description

* This repository contains some Attention exercises

## Learning Objectives

**Understand:**

* What is the attention mechanism?
* How to apply attention to RNNs
* What is a transformer?
* How to create an encoder-decoder transformer model
* What is GPT?
* What is BERT?
* What is self-supervised learning?
* How to use BERT for specific NLP tasks
* What is SQuAD? GLUE?

## Dependencies
```
Python 3.5
numpy 1.16
tensorflow 1.15
```

## Repo content

* **Main Folder that contains all main of the following tasks:**

| Task | Description |
| --- | --- |
|**0. RNN Encoder**| class RNNEncoder that inherits from tensorflow.keras.layers.Layer to encode for machine translation
|**1. Self Attention**| calculate the attention for machine translation based on paper
|**2. RNN Decoder**| class RNNDecoder that inherits from tensorflow.keras.layers.Layer to decode for machine translation
|**3. Positional Encoding**| calculates the positional encoding for a transformer
|**4. Scaled Dot Product Attention**| calculates the scaled dot product attention
|**5. Multi Head Attention**| class MultiHeadAttention that inherits from tensorflow.keras.layers.Layer to perform multi head attention
|**6. Transformer Encoder Block**| class EncoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer
|**7. Transformer Decoder Block**|  class DecoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer
|**8. Transformer Encoder**| class Encoder that inherits from tensorflow.keras.layers.Layer to create the encoder for a transformer
|**9. Transformer Decoder**| class Decoder that inherits from tensorflow.keras.layers.Layer to create the decoder for a transformer
|**10. Transformer Network**| class Transformer that inherits from tensorflow.keras.Model to create a transformer network

## Usage
* Clone the repo and execute the main files

## Author
- [Cristian G](https://github.com/cristian-fg)

## License
[MIT](https://choosealicense.com/licenses/mit/)
