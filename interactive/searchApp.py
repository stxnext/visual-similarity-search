import streamlit as st
import streamlit.components.v1 as components

from loguru import logger
from PIL import Image

from interactive import CATEGORY_DESCR, IMAGE_EXAMPLES
from metrics.consts import METRIC_COLLECTION_NAMES
from interactive.st_utils import (
    add_bg_from_local,
    initialize_state,
    reset_all_states_button,
    reset_states_after_upload_option_button,
    add_filter_option,
)


logger.info("Set main graphical options and initial paragraph.")
st.set_page_config(layout="wide")
add_bg_from_local("interactive/assets/green_overlay.png")
button_component = components.html("""
<script>
const button_elements = window.parent.document.querySelectorAll('.stButton > button')
button_elements[0].style.backgroundColor = 'lightcoral'
button_elements[1].style.backgroundColor = 'lightblue'
button_elements[2].style.backgroundColor = 'lightgreen'
</script>
""",
    height=0,
    width=0
)
st.title("Visual Similarity Search Engine")
st.write(
    """ 
    Returns a set number of images from a selected category. 
    This set contains images with the highest degree of similarity to the uploaded/selected image.
    Returned images are pulled from the local or cloud storage and similarity is calculated based on the vectors stored
    in the Qdrant database.
    Algorithm uses image embeddings and deep neural networks to determine a value of cosine similarity metric.
"""
)

logger.info("Create and store initial values of widgets in a session state dictionary.")
initialize_state()

logger.info("Sidebar - Input Options")
st.sidebar.image("interactive/assets/stxnext_web_color@1x.png")
st.sidebar.header("Input Options")
if st.sidebar.button("Reset All"):
    reset_all_states_button()

logger.info("Sidebar - Category Selection")
add_filter_option(text="Which category would you like to search from?")
for category in METRIC_COLLECTION_NAMES:
    if st.sidebar.button(CATEGORY_DESCR[category]["description"]):
        st.session_state.category_desc_option = CATEGORY_DESCR[category]["description"]
        st.session_state.category_option = category

logger.info("Sidebar - Image Provisioning")
upload_options = [
    "Example List",
    "Pull Randomly from the Storage",
    "File Upload",
]
if st.session_state.category_option is not None:
    add_filter_option(text="How would you like to add an image?")
    st.session_state.upload_option = st.sidebar.radio(
        label="",
        options=tuple(upload_options),
    )

logger.info("Sidebar - Image Selection")
if st.session_state.category_option is not None:
    if st.session_state.upload_option == upload_options[0]:
        reset_states_after_upload_option_button()
        st.session_state.example_captions = [
            s_img["label"] for s_img in IMAGE_EXAMPLES[st.session_state.category_option]
        ]
        st.session_state.example_imgs = [
            Image.open(s_img["path"])
            for s_img in IMAGE_EXAMPLES[st.session_state.category_option]
        ]
        example_images_zip = dict(
            zip(st.session_state.example_captions, st.session_state.example_imgs)
        )
        logger.info("Sidebar - Category Selection")
        add_filter_option(text=f"Choose an image - {st.session_state.category_option}.")
        img_selection = st.sidebar.selectbox("", example_images_zip)
        st.session_state.selected_img = example_images_zip[img_selection]
        st.session_state.show_input_img = True
    elif st.session_state.upload_option == upload_options[1]:
        reset_states_after_upload_option_button()
        add_filter_option(
            text=f"Choose random images from {st.session_state.category_option} category."
        )
        st.session_state.pull_random_img_number = st.sidebar.number_input(
            label="",
            value=5,
            min_value=1,
            format="%i",
        )
        if st.sidebar.button("Generate Images"):
            (
                st.session_state.random_captions,
                st.session_state.random_imgs,
            ) = st.session_state.client._get_random_images_from_collection(
                collection_name=st.session_state.category_option,
                k=st.session_state.pull_random_img_number,
            )
        if (
            st.session_state.random_captions is not None
            and st.session_state.random_imgs is not None
        ):
            random_images_zip = dict(
                zip(st.session_state.random_captions, st.session_state.random_imgs)
            )
            img_selection = st.sidebar.selectbox("Choose an item", random_images_zip)
            st.session_state.selected_img = random_images_zip[img_selection]
            st.session_state.show_input_img = True
        else:
            st.session_state.selected_img = None
    elif st.session_state.upload_option == upload_options[2]:
        reset_states_after_upload_option_button()
        byte_img = st.sidebar.file_uploader("Choose a file.")
        if byte_img is not None:
            st.session_state.selected_img = Image.open(byte_img)
            st.session_state.show_input_img = True
        else:
            st.session_state.selected_img = None

logger.info("Main View - Input Image")
if st.session_state.show_input_img:
    st.header("Input Image")
    input_img_placeholder = st.empty()
    input_img_placeholder.image(st.session_state.selected_img)


if st.session_state.category_option and st.session_state.selected_img:
    logger.info("Sidebar - Search Options Selection")
    st.sidebar.write("")
    st.sidebar.header("Search Options")
    add_filter_option(
        text="Insert a number of similar images to show.", no_linebreaks=True
    )
    st.session_state.similar_img_number = st.sidebar.number_input(
        label="",
        value=9,
        min_value=1,
        format="%i",
    )
    add_filter_option(
        text="Insert a benchmark similarity value (in %).", no_linebreaks=True
    )
    st.session_state.benchmark_similarity_value = st.sidebar.number_input(
        label="",
        value=50,
        min_value=0,
        max_value=100,
        format="%i",
    )
    st.session_state.grid_nrow_number = 3
    add_filter_option(text="")

    logger.info("Main View - Output Images")
    if st.button("Find Similar Images"):

        def search_with_show(
            collection: str, k: int, grid_nrow: int, benchmark: int, file
        ):
            (
                anchor,
                similars,
            ) = st.session_state.client._get_best_choice_for_uploaded_image(
                base_img=file,
                collection_name=collection,
                k=k,
                benchmark=benchmark,
            )
            if similars is not None:
                st.write(
                    f'Found {st.session_state.similar_img_number} images in the "{st.session_state.category_option}" category.'
                )
                st.write(
                    f'In the top left corner of every image a similarity coefficient is presented - it shows a level of similarity between a given image and an input image.'
                )
                col_nr = min(grid_nrow, len(similars))
                for i, col in enumerate(st.columns(col_nr)):
                    col_imgs = similars[i::col_nr]
                    with col:
                        for col_img in col_imgs:
                            st.image(col_img)
            else:
                st.write(f"No images found for the benchmark of {benchmark}%.")

        st.header("Similar Images")
        search_with_show(
            file=st.session_state.selected_img,
            collection=st.session_state.category_option,
            k=st.session_state.similar_img_number,
            grid_nrow=st.session_state.grid_nrow_number,
            benchmark=st.session_state.benchmark_similarity_value,
        )

        if st.button("Reset Images"):
            reset_all_states_button()
