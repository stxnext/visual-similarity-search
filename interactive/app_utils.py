import streamlit as st

from PIL import Image

from interactive import (
    APP_TITLE,
    LOGO_IMG_DIR,
    CATEGORY_DESCR,
    IMAGE_EXAMPLES,
)
from common import env_handler

from metrics.consts import METRIC_COLLECTION_NAMES
from metrics.core import MetricClient


class ModuleManager:
    """
    List of components used for building the app.
    """
    def __init__(self, states_manager):
        self.states_manager = states_manager

    @staticmethod
    def create_header():
        """
        Creates initial global formatting and a header structure.
        """
        col_logo_1, col_logo_2, col_logo_3 = st.columns([1, 1, 1])
        with col_logo_1:
            st.write("")
        with col_logo_2:
            st.image(LOGO_IMG_DIR)
        with col_logo_3:
            st.write("")
        st.title(APP_TITLE)
        st.write(
            """ 
                Returns a set number of images from a selected category. 
                This set contains images with the highest degree of similarity to the uploaded/selected image.
                Returned images are pulled from the local or cloud storage and similarity is calculated based on the vectors 
                stored in the Qdrant database.
                Algorithm uses image embeddings and deep neural networks to determine a value of cosine similarity metric.
            """
        )

    def create_main_filters(self):
        """
        Adds initial header, reset all filers button, and category selection buttons.
        """
        st.header("Input Options")  # header
        if st.button("Reset All"):  # reset all button
            self.states_manager.reset_all_states_button()
        st.write("")
        st.markdown(
            f'<p class="big-font">Which category would you like to search from?</p>',
            unsafe_allow_html=True,
        )  # header
        for category in METRIC_COLLECTION_NAMES:
            if st.button(CATEGORY_DESCR[category]["description"]):  # category buttons
                st.session_state.category_desc_option = CATEGORY_DESCR[category][
                    "description"
                ]
                st.session_state.category_option = category
        st.write("")

    @staticmethod
    def create_image_provisioning_options(provisioning_options: list[str]):
        """
        Adds list of image provisioning options that define following steps.
        """
        if st.session_state.category_option is not None:
            st.session_state.provisioning_options = st.radio(
                label="How would you like to add an image?",
                options=tuple(provisioning_options),
            )

    def create_image_provision_for_examples(self):
        """
        Resets state of previous provisioning selection and creates a category-specific list of image examples
        that a user can select from.
        """
        self.states_manager.reset_states_after_image_provisioning_list()
        st.session_state.example_captions = [
            s_img["label"] for s_img in IMAGE_EXAMPLES[st.session_state.category_option]
        ]  # get captions
        st.session_state.example_imgs = [
            Image.open(s_img["path"])
            for s_img in IMAGE_EXAMPLES[st.session_state.category_option]
        ]  # get images
        example_images_zip = dict(
            zip(st.session_state.example_captions, st.session_state.example_imgs)
        )
        img_selection = st.selectbox(
            f"Choose an image - {st.session_state.category_option}.",
            example_images_zip,
        )  # select image
        st.session_state.selected_img = example_images_zip[img_selection]
        st.session_state.show_input_img = True

    def create_image_provision_for_random_storage_pull(self):
        """
        Resets state of previous provisioning selection and pulls from local/cloud storage a category-specific list of
        image examples that a user can select from the list. Additionally, a button for re-running random selection is
        implemented together with the input option for the number of sampled images.
        """
        self.states_manager.reset_states_after_image_provisioning_list()
        st.session_state.pull_random_img_number = st.number_input(
            label=f"Choose random images from {st.session_state.category_option} category.",
            value=5,
            min_value=1,
            format="%i",
        )
        if st.button("Generate Images"):
            (
                st.session_state.random_captions,
                st.session_state.random_imgs,
            ) = env_handler.get_random_images_from_collection(
                collection_name=st.session_state.category_option,
                k=st.session_state.pull_random_img_number,
            )  # Pulls a sampled set of images from local/cloud storage
        if (
            st.session_state.random_captions is not None
            and st.session_state.random_imgs is not None
        ):
            random_images_zip = dict(
                zip(
                    st.session_state.random_captions,
                    st.session_state.random_imgs,
                )
            )
            img_selection = st.selectbox("Choose an image", random_images_zip)
            st.session_state.selected_img = random_images_zip[
                img_selection
            ]  # returns an image based on selection
            st.session_state.show_input_img = True

    def create_image_provision_for_manual_upload(self):
        """
        Resets state of previous provisioning selection and provides upload button for a user.
        Any RGB image can be uploaded.
        """
        self.states_manager.reset_states_after_image_provisioning_list()
        st.markdown(f'<p class="big-font">Choose a file.</p>', unsafe_allow_html=True)
        byte_img = st.file_uploader("Upload an image from a local disk.")
        if byte_img is not None:
            st.session_state.selected_img = Image.open(byte_img)
            st.session_state.show_input_img = True

    def create_similarity_search_filters(self):
        """
        Creates a set of similarity-search-specific filters - number of shown images an benchmark for
        minimum similarity value in %.
        """
        st.subheader("Search Options")
        col_search_1, col_search_2 = st.columns([1, 1])
        with col_search_1:
            st.session_state.similar_img_number = st.number_input(
                label="Insert a number of similar images to show.",
                value=9,
                min_value=1,
                format="%i",
            )
        with col_search_2:
            st.session_state.benchmark_similarity_value = st.number_input(
                label="Insert a benchmark similarity value (in %).",
                value=50,
                min_value=0,
                max_value=100,
                format="%i",
            )

    def search_with_show(
        self, collection: str, k: int, grid_nrow: int, benchmark: int, file
    ):
        """
        Shows images in order of their similarity to the original input image.
        """
        (anchor, similars,) = MetricClient().get_best_choice_for_uploaded_image(
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
                f"In the top left corner of every image a similarity coefficient is presented - it shows a level of similarity between a given image and an input image."
            )
            col_nr = min(grid_nrow, len(similars))
            for i, col in enumerate(st.columns(col_nr)):
                col_imgs = similars[i::col_nr]
                with col:
                    for col_img in col_imgs:
                        st.image(col_img)
        else:
            st.write(f"No images found for the similarity benchmark of {benchmark}%.")

    def extract_similar_images(self):
        """
        Shows images in order of their similarity to the original input image.
        """
        self.search_with_show(
            file=st.session_state.selected_img,
            collection=st.session_state.category_option,
            k=st.session_state.similar_img_number,
            grid_nrow=st.session_state.grid_nrow_number,
            benchmark=st.session_state.benchmark_similarity_value,
        )

        if st.button("Reset Images"):
            self.states_manager.reset_all_states_button()
