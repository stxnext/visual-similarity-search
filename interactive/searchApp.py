import streamlit as st

from loguru import logger
from PIL import Image
from streamlit_image_select import image_select

from interactive import CATEGORY_DESCR, IMAGE_EXAMPLES
from metrics.consts import METRIC_COLLECTION_NAMES
from metrics.core import MetricClient

logger.info("Store the initial value of widgets in session state")
if "client" not in st.session_state:
    st.session_state.client = MetricClient()
    st.session_state.category_desc_option = None
    st.session_state.category_option = None
    st.session_state.similar_img_number = None
    st.session_state.upload_option = None
    st.session_state.selected_img = None
    st.session_state.pull_random_img_number = None
    st.session_state.refresh_random_images = True
    st.session_state.random_captions = None
    st.session_state.random_imgs = None
    st.session_state.similar_images_found = None
    st.session_state.grid_nrow_number = None


st.write(""" # Visual Similarity Search Engine """)
st.write(
    """ 
    Returns a set of user-defined number of images from a selected category. 
    This set contains images with the highest degree of similarity to the uploaded image.
    Returned images are pulled from the cloud storage and similarity is calculated based on the vectors stored in the Qdrant database.
    Algorithm uses image embeddings and deep neural networks to determine a value of cosine similarity metric.
"""
)

st.write(""" ## Find Similar Images within Category """)

st.session_state.category_desc_option = st.selectbox(
    label="Which category you would like to search from?",
    options=tuple(
        [CATEGORY_DESCR[cat]["description"] for cat in METRIC_COLLECTION_NAMES]
    ),
)

st.session_state.category_option = [
    cat
    for cat, d in CATEGORY_DESCR.items()
    if d["description"] == st.session_state.category_desc_option
][0]

st.session_state.similar_img_number = st.number_input(
    label="Insert a number of similar images to show.",
    value=10,
    min_value=1,
    max_value=25,
    format="%i",
)

col1, col2 = st.columns(2)

with col1:
    st.session_state.upload_option = st.selectbox(
        label="How would you like to add an image?",
        options=(
            "File Upload",
            "Image Chosen From the Example List",
            "Pull Randomly from Cloud Storage",
        ),
    )
with col2:
    if st.session_state.upload_option == "File Upload":
        byte_img = st.file_uploader("Choose a file.")
        if byte_img is not None:
            st.session_state.selected_img = Image.open(byte_img)
        else:
            st.session_state.selected_img = False
    elif st.session_state.upload_option == "Image Chosen From the Example List":
        st.session_state.selected_img = image_select(
            f"Choose an image from {st.session_state.category_option} category",
            images=[
                Image.open(s_img["path"])
                for s_img in IMAGE_EXAMPLES[st.session_state.category_option]
            ],
            captions=[
                s_img["label"]
                for s_img in IMAGE_EXAMPLES[st.session_state.category_option]
            ],
        )
        if st.button("Reset Images"):
            st.session_state.refresh_random_images = True
            st.session_state.random_captions = None
            st.session_state.random_imgs = None
            st.session_state.selected_img = None
    elif st.session_state.upload_option == "Pull Randomly from Cloud Storage":
        st.write(f"Choose an image from {st.session_state.category_option} category")
        st.session_state.pull_random_img_number = st.number_input(
            label="Insert a number of similar images to show.",
            value=5,
            min_value=1,
            max_value=10,
            format="%i",
        )
        if st.button("Generate Images"):
            if st.session_state.refresh_random_images:
                (
                    st.session_state.random_captions,
                    st.session_state.random_imgs,
                ) = st.session_state.client._get_random_images_from_collection(
                    collection_name=st.session_state.category_option,
                    k=st.session_state.pull_random_img_number,
                )
                st.session_state.refresh_random_images = False
            else:
                st.write('Use "Reset Images" button first.')
        if not st.session_state.refresh_random_images:
            if st.button("Reset Images"):
                st.session_state.refresh_random_images = True
                st.session_state.random_captions = None
                st.session_state.random_imgs = None
                st.session_state.selected_img = None
        if (
            st.session_state.random_captions is not None
            and st.session_state.random_imgs is not None
        ):
            images_zip = dict(
                zip(st.session_state.random_captions, st.session_state.random_imgs)
            )
            img_selection = st.selectbox("Choose an item", images_zip)
            st.session_state.selected_img = images_zip[img_selection]
            st.image(st.session_state.selected_img)
        else:
            st.session_state.selected_img = None

if (
    st.session_state.category_option
    and st.session_state.similar_img_number
    and st.session_state.selected_img
):
    st.write(""" ### Similar Images Search """)
    st.session_state.grid_nrow_number = st.number_input(
        label="Insert a maximum number of images in the row.",
        value=3,
        min_value=1,
        max_value=5,
        format="%i",
    )
    if st.button("Find Similar Images"):

        def search_with_show(collection: str, k: int, grid_nrow: int, file):
            (
                anchor,
                similars,
            ) = st.session_state.client._get_best_choice_for_uploaded_image(
                img=file,
                collection_name=collection,
                k=k,
                grid_nrow=grid_nrow,
            )
            st.write(
                f"Searched for {st.session_state.similar_img_number} images in {st.session_state.category_desc_option} category."
            )
            st.image(similars)

        try:
            search_with_show(
                file=st.session_state.selected_img,
                collection=st.session_state.category_option,
                k=st.session_state.similar_img_number,
                grid_nrow=st.session_state.grid_nrow_number,
            )
            st.session_state.similar_images_found = True
        except:
            st.write("File not selected")

        if st.button("Reset"):
            st.session_state.similar_images_found = False
