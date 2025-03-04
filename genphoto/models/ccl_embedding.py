import torch

from transformers import DistilBertTokenizer, DistilBertModel
from torch.nn.functional import cosine_similarity

class FastLightweightTextEncoder:
    def __init__(self, model_name='distilbert-base-uncased', cache_dir='/path/to/your/cache'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.text_encoder = DistilBertModel.from_pretrained(model_name).eval().cuda()

    def encode_texts(self, prompts):
        # Batch processing the prompts to get their embeddings
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()

        with torch.no_grad():
            embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Normalize embeddings to get consistent vector representations
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        # Print shape of embeddings
     #   print(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def calculate_differences(self, embeddings):
        # Calculate differences between consecutive embeddings
        differences = []
        for i in range(1, embeddings.size(0)):
            diff = embeddings[i] - embeddings[i - 1]
            print('diff shape', diff.shape)
            differences.append(diff.unsqueeze(0))  # Add batch dimension
            print('differences shape', differences.shape)

        # Concatenate differences along the batch dimension (f-1)
        concatenated_differences = torch.cat(differences, dim=0)  # Shape: (f-1, sequence_length, hidden_size)
        return concatenated_differences

# Example usage
if __name__ == '__main__':
    prompts = [
        "A smiling dog. Focal length: 24mm.",
        "A smiling dog. Focal length: 25mm.",
        "A smiling dog. Focal length: 26mm.",
        "A smiling dog. Focal length: 30mm.",
        "A smiling dog. Focal length: 36mm.",
    ]

    # Initialize the FastLightweightTextEncoder
    text_encoder = FastLightweightTextEncoder(cache_dir='/home/yuan418/lab/users/Yu/modules/')

    # Encode the prompts
    embeddings = text_encoder.encode_texts(prompts)
    print('a')
    print('embeddings', embeddings)
    print('embeddings shape', embeddings.shape)

    # Calculate and concatenate differences
    concatenated_diffs = text_encoder.calculate_differences(embeddings)

    print("Concatenated differences shape:", concatenated_diffs.shape)


