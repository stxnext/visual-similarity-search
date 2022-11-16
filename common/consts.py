import os

from pathlib import Path


PROJECT_PATH = Path(__file__).parent.parent
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
MINIO_MAIN_PATH = Path(os.getenv("MINIO_MAIN_PATH"))
INTERACTIVE_ASSETS_DICT = {
    "logo_img_dir": "interactive/assets/stxnext_web_color@1x.png",
    "app_style_file": PROJECT_PATH / "interactive" / "style.css",
    "app_title": "Visual Similarity Search Engine",
    "app_first_paragraph": """
            Returns a set number of images from a selected category. 
            This set contains images with the highest degree of similarity to the uploaded/selected image.
            Returned images are pulled from the local or cloud storage and similarity is calculated based on the vectors 
            stored in the Qdrant database.
            Algorithm uses image embeddings and deep neural networks to determine a value of cosine similarity metric.
        """,
}
