from sentence_transformers import SentenceTransformer
import torch

class SentenceEncoder:
    def __init__(self, device='cpu', model_name='all-MiniLM-L6-v2'):
        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)

    def encode(self, sentences):
        with torch.no_grad():
            # Return embeddings as PyTorch tensor on the correct device
            embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings.to(self.device)

# Example usage (can be removed if only used in training scripts)
if __name__ == "__main__":
    encoder = SentenceEncoder(device='cuda' if torch.cuda.is_available() else 'cpu')
    sample = ["A red bird sitting on a tree branch"]
    embedding = encoder.encode(sample)
    print(embedding.shape)  # (1, 384) for MiniLM
