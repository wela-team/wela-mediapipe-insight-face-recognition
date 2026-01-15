"""Utility for building face embeddings from sample images."""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from insightface.app import FaceAnalysis

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from configs.config import EMBEDS_DIR, FACES_DIR


class EmbeddingBuilder:
    """Builds face embeddings from sample images."""

    def __init__(self, model_name: str = "buffalo_l", providers: list = None):
        """
        Initialize the embedding builder.

        Args:
            model_name: InsightFace model name
            providers: Execution providers (default: CPU)
        """
        if providers is None:
            providers = ["CPUExecutionProvider"]
        
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0)

    def build_embeddings(
        self,
        samples_dir: Path,
        embeddings_dir: Path = EMBEDS_DIR
    ) -> int:
        """
        Build embeddings from all images in a directory.
        Each image is assigned a unique label based on its filename.

        Args:
            samples_dir: Directory containing face images
            embeddings_dir: Directory to save .npy embedding files

        Returns:
            Number of successfully processed images
        """
        samples_path = Path(samples_dir)
        embeds_path = Path(embeddings_dir)
        
        # Create embeddings directory
        embeds_path.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        img_files = [
            f for f in samples_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ]

        if not img_files:
            print(f"⚠️ No images found in {samples_path}")
            return 0

        processed_count = 0

        for img_path in img_files:
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"⚠️ Failed to read {img_path}, skipping...")
                continue

            faces = self.app.get(img)

            if faces:
                # Use filename (without extension) as unique label
                label = img_path.stem
                embedding = faces[0].embedding
                
                # Convert to numpy array
                embedding_array = np.array(embedding)
                
                # Normalize embedding to unit vector for better cosine similarity
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    embedding_array = embedding_array / norm
                else:
                    print(f"⚠️ Zero-norm embedding for {img_path.name}, skipping...")
                    continue
                
                # Save as .npy file
                npy_file = embeds_path / f"{label}.npy"
                np.save(npy_file, embedding_array)
                
                print(f"✔ Processed {img_path.name} → label: {label} → saved to {npy_file} (shape: {embedding_array.shape}, norm: {np.linalg.norm(embedding_array):.4f})")
                processed_count += 1
            else:
                print(f"⚠️ No face detected in {img_path.name}")

        print(f"✅ Embeddings saved to {embeds_path} for {processed_count} images")
        return processed_count


def build_embeddings(
    samples_dir: str = FACES_DIR,
    embeddings_dir: Optional[str] = None
) -> int:
    """
    Convenience function to build embeddings from sample images.

    Args:
        samples_dir: Directory containing face images
        embeddings_dir: Directory to save embeddings (default: from config)

    Returns:
        Number of successfully processed images
    """
    if embeddings_dir is None:
        embeddings_dir = EMBEDS_DIR
    else:
        embeddings_dir = Path(embeddings_dir)
    
    builder = EmbeddingBuilder()
    return builder.build_embeddings(Path(samples_dir), embeddings_dir)


if __name__ == "__main__":
    import sys
    
    samples_dir = sys.argv[1] if len(sys.argv) > 1 else FACES_DIR
    build_embeddings(samples_dir)
