import torch
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from PIL import Image
import numpy as np

class ClipEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading CLIP model '{model_name}' to {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def get_embedding(self, frames):
        """
        Takes a list of numpy array RGB frames, converts to PIL, runs through CLIP,
        averages the embeddings, and returns a single embedding vector (512,).
        """
        if not frames:
            return None

        # Convert numpy frames to PIL images
        pil_images = [Image.fromarray(frame) for frame in frames]

        # Prepare inputs
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Normalize features? CLIP typically returns raw, but often L2 normalized is used.
        # But here we just get raw -> average -> probably want to normalize the average or just return raw average.
        # The prompt says "Average the embeddings to produce a single 512-dimensional vector".
        
        # Averaging embeddings for the video (shape: [num_frames, 512]) -> [1, 512]
        avg_embedding = torch.mean(image_features, dim=0).cpu().numpy()
        
        return avg_embedding

if __name__ == "__main__":
    # Test
    embedder = ClipEmbedder()
    dummy_frames = [(np.random.rand(224, 224, 3) * 255).astype(np.uint8) for _ in range(3)]
    emb = embedder.get_embedding(dummy_frames)
    print("Embedding shape:", emb.shape)
