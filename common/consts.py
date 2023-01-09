import os
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
MINIO_MAIN_PATH = Path(os.getenv("MINIO_MAIN_PATH"))
INTERACTIVE_ASSETS_DICT = {
    "logo_img_dir": "interactive/assets/logo_stx.jpg",
    "widget_style_file": PROJECT_PATH / "interactive" / "widget_style.css",
    "app_title": "Visual Similarity Search Engine",
    "app_first_paragraph": """
            Returns a set number of images from a selected category. 
            This set contains images with the highest degree of similarity to the uploaded/selected image.
            Returned images are pulled from the local or cloud storage and similarity is calculated based on the vectors 
            stored in the Qdrant database.
            Algorithm uses image embeddings and deep neural networks to determine a value of cosine similarity metric.
        """,
    "github_link": "https://github.com/stxnext/visual-similarity-search",
    "our_ml_site_link": "https://www.stxnext.com/services/machine-learning/?utm_source=github&utm_medium=mlde&utm_campaign=visearch-demo",
}
GRID_NROW_NUMBER = 3
EXAMPLE_PATH = PROJECT_PATH / "interactive" / "examples"
CATEGORY_DESCR = {
    "dogs": {
        "description": "Dogs - 120 breeds of dogs from around the world.",
        "business_usage": """
            Dogs are just an example and following use cases can be extrapolated to other animal species. \n
            **USE CASE #1**: Identifying a breed of the dog based on its picture. 
            It may be useful for veterinarians and other animal-specialized occupations. \n
            **USE CASE #2**: Finding other breeds based on the similarity factor. 
            It may be useful for breeders and people looking to buy an animal.
        """,
        "source": "https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset",
        "bootstrap_icon": "bag",
        "image_examples": [
            {
                "id": "dog_1",
                "path": EXAMPLE_PATH / "dogs_img_1.png",
                "label": "Cane Corso",
            },
            {
                "id": "dog_2",
                "path": EXAMPLE_PATH / "dogs_img_2.jpg",
                "label": "St. Bernard",
            },
        ],
    },
    "shoes": {
        "description": "Shoes - 4 major categories with individual brands.",
        "business_usage": """
            Shoes are just an example and following use cases can be extrapolated to other fashion-based categories. \n
            **USE CASE #1**: Identifying a brand of the shoes in the picture. It may be useful for users of online shops. \n
            **USE CASE #2**: Identifying clothes that match the uploaded picture of shoes. 
            It may be useful for users of online shops. 
        """,
        "source": "https://www.kaggle.com/datasets/aryashah2k/large-shoe-dataset-ut-zappos50k",
        "bootstrap_icon": "bag",
        "image_examples": [
            {
                "id": "shoes_1",
                "path": EXAMPLE_PATH / "shoes_img_1.jpg",
                "label": "Colorful Boots",
            },
            {
                "id": "shoes_2",
                "path": EXAMPLE_PATH / "shoes_img_2.jpg",
                "label": "Transparent Shoes",
            },
        ],
    },
    "celebrities": {
        "description": "Celebrities - almost 4000 famous people found on IMDB.",
        "business_usage": """
            Celebrities are just an example and following use cases can be extrapolated to singers, athletes, etc. \n
            **USE CASE #1**: Identifying a closest celebrity match with uploaded image. 
            It may be useful for marketing purposes.
        """,
        "source": "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki",
        "bootstrap_icon": "person-square",
        "image_examples": [
            {
                "id": "celebrities_1",
                "path": EXAMPLE_PATH / "celebrities_img_1.jpg",
                "label": "Clint Eastwood",
            },
            {
                "id": "celebrities_2",
                "path": EXAMPLE_PATH / "celebrities_img_2.jpg",
                "label": "Jason Momoa",
            },
            {
                "id": "celebrities_2",
                "path": EXAMPLE_PATH / "celebrities_img_3.jpeg",
                "label": "Meryl Streep",
            },
        ],
    },
    "logos": {
        "description": "Logos - 3000 logos of international companies.",
        "business_usage": """
            **USE CASE #1**: Identifying whether a given logotype already exists in the market. 
            It may be useful when creating a new logo for the company and limiting legal issues when a close match is found.
        """,
        "source": "https://www.kaggle.com/datasets/lyly99/logodet3k",
        "bootstrap_icon": "images",
        "image_examples": [
            {
                "id": "logos_1",
                "path": EXAMPLE_PATH / "logos_img_1.jpg",
                "label": "STX Next",
            },
            {
                "id": "logos_2",
                "path": EXAMPLE_PATH / "logos_img_2.png",
                "label": "Ermlab Software",
            },
        ],
    },
    "waste": {
        "description": "Waste - 12 categories of household garbage.",
        "business_usage": """
            **USE CASE #1**: Identifying general category of recyclable waste for the purpose of sorting. 
            It may be useful in machines that automatically split mixed garbage to its recyclable and organic components.
            **USE CASE #2**: Identifying a type of recyclable waste for the purpose of household sorting. 
            It may be useful when a target category of waste is difficult to identify.
        """,
        "source": "https://www.kaggle.com/datasets/mostafaabla/garbage-classification",
        "bootstrap_icon": "trash",
        "image_examples": [
            {
                "id": "waste_1",
                "path": EXAMPLE_PATH / "waste_img_1.jpg",
                "label": "Cardboard",
            },
            {
                "id": "waste_2",
                "path": EXAMPLE_PATH / "waste_img_2.png",
                "label": "Glass",
            },
        ],
    },
}
