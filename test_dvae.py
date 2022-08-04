from utils import create_d_vae
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel

d_vae_model = create_d_vae(
    d_vae_type='dall-e',
    weight_path='../dall_e_tokenizer_weight',
    image_size=224,
    device='cpu'
)
model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[2]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
encoding = processor(image, words, boxes=boxes, return_tensors="pt")

# 4D-TENSOR
input_ids = d_vae_model.get_codebook_indices(encoding['pixel_values']).flatten(1) # (B,28,28)
print(encoding['pixel_values'].shape,input_ids.shape)
# print(input_ids)
outputs = model(**encoding)
word_len = encoding['input_ids'].shape[1]
last_hidden_states = outputs.last_hidden_state
print('Last Hidden Shape:',last_hidden_states.shape)
image_out = last_hidden_states[:,word_len:,:]
print('Image Shape:',image_out.shape)

img_embed = model.forward_image(encoding['pixel_values'])