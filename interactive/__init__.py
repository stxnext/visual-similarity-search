from common.global_utils import PROJECT_PATH


LOGO_IMG_DIR = "interactive/assets/stxnext_web_color@1x.png"
APP_TITLE = "Visual Similarity Search Engine"
GRID_NROW_NUMBER = 3

CATEGORY_DESCR = {
    "dogs": {
        "description": "Dogs - 120 breeds of dogs from around the world.",
        "source": "https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset",
    },
    "shoes": {
        "description": "Shoes - 4 major categories with individual brands.",
        "source": "https://www.kaggle.com/datasets/aryashah2k/large-shoe-dataset-ut-zappos50k",
    },
}
EXAMPLE_PATH = PROJECT_PATH / "interactive" / "examples"
IMAGE_EXAMPLES = {
    "dogs": [
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
    "shoes": [
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
}
