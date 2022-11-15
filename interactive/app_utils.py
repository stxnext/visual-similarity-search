import streamlit as st

from PIL import Image

from common.consts import INTERACTIVE_ASSETS_DICT
from interactive import (
    CATEGORY_DESCR,
    IMAGE_EXAMPLES,
)
from common import env_handler

from metrics.consts import MetricCollections
from metrics.core import MetricClient


class ModuleManager:
    """
    List of components used for building the app.
    """

    def __init__(self, states_manager) -> None:
        self.states_manager = states_manager
        self.metric_client = MetricClient()

    def create_header(self) -> None:
        """
        Creates initial global formatting and a header structure.
        """
        col_logo_1, col_logo_2, col_logo_3 = st.columns([1, 1, 1])
        with col_logo_1:
            st.write("")
        with col_logo_2:
            st.image(INTERACTIVE_ASSETS_DICT["logo_img_dir"])
        with col_logo_3:
            st.write("")
        st.title(INTERACTIVE_ASSETS_DICT["app_title"])
        st.write(INTERACTIVE_ASSETS_DICT["app_first_paragraph"])

    def create_main_filters(self) -> None:
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
        for category_enum in MetricCollections:
            if st.button(CATEGORY_DESCR[category_enum.value]["description"]):  # category buttons
                st.session_state.category_desc_option = CATEGORY_DESCR[category_enum.value][
                    "description"
                ]
                st.session_state.category_option = category_enum
        st.write("")

    def create_image_provisioning_options(
        self, provisioning_options: list[str]
    ) -> None:
        """
        Creates a global state with a list of image provisioning options. This is done based on radio button input.
        """
        if st.session_state.category_option:
            st.session_state.provisioning_options = st.radio(
                label="How would you like to add an image?",
                options=tuple(provisioning_options),
            )

    def create_image_provision_for_examples(self) -> None:
        """
        Resets state of previous provisioning selection and creates a category-specific list of image examples
        that a user can select from.
        """
        self.states_manager.reset_states_after_image_provisioning_list()
        st.session_state.example_captions = [
            s_img["label"] for s_img in IMAGE_EXAMPLES[st.session_state.category_option.value]
        ]  # get captions
        st.session_state.example_imgs = [
            Image.open(s_img["path"])
            for s_img in IMAGE_EXAMPLES[st.session_state.category_option.value]
        ]  # get images
        example_images_zip = dict(
            zip(st.session_state.example_captions, st.session_state.example_imgs)
        )
        img_selection = st.selectbox(
            f"Choose an image - {st.session_state.category_option.value}.",
            example_images_zip,
        )  # select image
        st.session_state.selected_img = example_images_zip[img_selection]
        st.session_state.show_input_img = True

    def create_image_provision_for_random_storage_pull(self) -> None:
        """
        Resets state of previous provisioning selection and pulls from local/cloud storage a category-specific list of
        image examples that a user can select from the list. Additionally, a button for re-running random selection is
        implemented together with the input option for the number of sampled images.
        """
        self.states_manager.reset_states_after_image_provisioning_list()
        st.session_state.pull_random_img_number = st.number_input(
            label=f"Choose random images from {st.session_state.category_option.value} category.",
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
        if st.session_state.random_captions and st.session_state.random_imgs:
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

    def create_image_provision_for_manual_upload(self) -> None:
        """
        Resets state of previous provisioning selection and provides upload button for a user.
        Any RGB image can be uploaded.
        """
        self.states_manager.reset_states_after_image_provisioning_list()
        st.markdown(f'<p class="big-font">Choose a file.</p>', unsafe_allow_html=True)
        byte_img = st.file_uploader("Upload an image from a local disk.")
        if byte_img:
            st.session_state.selected_img = Image.open(byte_img)
            st.session_state.show_input_img = True

    def create_similarity_search_filters(self) -> None:
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
        self, collection_name: MetricCollections, k: int, grid_nrow: int, benchmark: int, file
    ) -> None:
        """
        Shows images in order of their similarity to the original input image.
        """
        anchor, similars = self.metric_client.get_best_choice_for_uploaded_image(
            base_img=file,
            collection_name=collection_name,
            k=k,
            benchmark=benchmark,
        )
        if similars:
            st.write(
                f'Found {st.session_state.similar_img_number} images in the "{st.session_state.category_option.value}" category.'
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

    def extract_similar_images(self) -> None:
        """
        Shows images in order of their similarity to the original input image.
        """
        self.search_with_show(
            file=st.session_state.selected_img,
            collection_name=st.session_state.category_option,
            k=st.session_state.similar_img_number,
            grid_nrow=st.session_state.grid_nrow_number,
            benchmark=st.session_state.benchmark_similarity_value,
        )

        if st.button("Reset Images"):
            self.states_manager.reset_all_states_button()
