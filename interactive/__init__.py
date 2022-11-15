from common.consts import PROJECT_PATH


GRID_NROW_NUMBER = 3
EXAMPLE_PATH = PROJECT_PATH / "interactive" / "examples"
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
