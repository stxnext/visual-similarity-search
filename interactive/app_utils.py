from typing import Any

import streamlit as st
from loguru import logger
from PIL import Image

from common import env_handler
from common.consts import CATEGORY_DESCR, GRID_NROW_NUMBER, INTERACTIVE_ASSETS_DICT
from metrics.consts import MetricCollections
from metrics.core import BestChoiceImagesDataset, MetricClient


class ModuleManager:
    """
    List of components used for building the app.
    """

    def __init__(self) -> None:
        self.metric_client = MetricClient()

    def widget_formatting(self) -> Any:
        """
        Defines Streamlit widget styles based on the input provided by style.css file.
        """
        with open(INTERACTIVE_ASSETS_DICT["widget_style_file"], "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def initialize_states(self) -> None:
        """
        Initializes states within the application. Those are distinct for every user.
        The "if" condition is necessary for the app to not reset it's whole session_state.
        """
        if "init" not in st.session_state:
            st.session_state.init = True
            st.session_state.category_desc_option = None
            st.session_state.category_option = None
            st.session_state.provisioning_options = None

            # Option 1 States - Example Images
            st.session_state.example_captions = None
            st.session_state.example_imgs = None
            st.session_state.example_list_pull = False  # NEW

            # Option 2 States - Storage Images
            st.session_state.pull_random_img_number = None
            st.session_state.refresh_random_images = None
            st.session_state.random_captions = None
            st.session_state.random_imgs = None
            st.session_state.img_storage_list_pull = False  # NEW

            # Option 3 States - Uploaded File Images
            st.session_state.file_upload_pull = False  # NEW

            # All Options States - set when an input image has been selected
            st.session_state.show_input_img = None
            st.session_state.selected_img = None

            # Search and Output States - set when an input image has been selected and before "Find Similar Images" is run
            st.session_state.similar_img_number = None
            st.session_state.benchmark_similarity_value = None
            st.session_state.grid_nrow_number = GRID_NROW_NUMBER

            # Search Completed State - set after "Find Similar Images" is completed
            # st.session_state.similar_images_found = None

    def reset_all_states_button(self) -> None:
        """
        Reset all application starting from category selection.
        """
        st.session_state.category_desc_option = None
        st.session_state.category_option = None
        st.session_state.provisioning_options = None
        st.session_state.show_input_img = None
        st.session_state.selected_img = None
        st.session_state.benchmark_similarity_value = None
        st.session_state.similar_img_number = None
        # st.session_state.similar_images_found = None

        st.session_state.example_list_pull = None  # NEW
        st.session_state.img_storage_list_pull = None  # NEW
        st.session_state.file_upload_pull = None  # NEW

    def reset_states_after_image_provisioning_list(self) -> None:
        """
        Reset all application states after radio selection for image provisioning.
        """
        st.session_state.show_input_img = None
        st.session_state.selected_img = None
        st.session_state.benchmark_similarity_value = None
        st.session_state.similar_img_number = None
        # st.session_state.similar_images_found = None

    def create_title_containers(self) -> None:
        """
        Creates initial global formatting and a header structure.
        """
        row_company_image = st.container()
        with row_company_image:
            st.image(Image.open(INTERACTIVE_ASSETS_DICT["logo_img_dir"]))

        row_app_title = st.container()
        with row_app_title:
            col_company_title_1, col_company_title_2 = st.columns([2, 1])
            with col_company_title_1:
                st.title(INTERACTIVE_ASSETS_DICT["app_title"])
        with st.expander("", expanded=True):
            with col_company_title_2:
                if st.button("Reset All"):  # reset all button
                    self.reset_all_states_button()
            st.write(INTERACTIVE_ASSETS_DICT["app_first_paragraph"])
            st.markdown(
                """
                <style>
                    div[data-testid="column"]:nth-of-type(1)
                    {
                        text-align: start;
                    }
                    div[data-testid="column"]:nth-of-type(2)
                    {
                        text-align: end;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

    def make_grid(self, cols, rows) -> Any:
        grid = [0] * cols
        for i in range(cols):
            with st.container():
                grid[i] = st.columns(rows)
        return grid

    def create_main_filters(self) -> None:
        """
        Adds initial header, reset all filers button, and category selection buttons.
        """
        st.markdown(
            "<h2 style='text-align: center;'>Input Options</h2>", unsafe_allow_html=True
        )
        with st.expander("", expanded=True):
            st.markdown(
                "<h3 style='text-align: center;'>Which category would you like to search from?</h3>",
                unsafe_allow_html=True,
            )
            mygrid = self.make_grid(4, (2, 2, 2, 2))
            for idx, category_enum in enumerate(MetricCollections):
                with mygrid[idx // 2][1 + idx % 2]:
                    if st.button(
                        CATEGORY_DESCR[category_enum.value]["description"]
                    ):  # category buttons
                        st.session_state.category_desc_option = CATEGORY_DESCR[
                            category_enum.value
                        ]["description"]
                        st.session_state.category_option = category_enum

            if st.session_state.category_option:
                st.markdown(
                    "<h3 style='text-align: center;'>Business Cases</h3>",
                    unsafe_allow_html=True,
                )
                st.write(
                    CATEGORY_DESCR[st.session_state.category_option.value][
                        "business_usage"
                    ]
                )
                st.write(
                    f"Source Dataset: [link]({CATEGORY_DESCR[st.session_state.category_option.value]['source']})"
                )

    def create_image_provision_for_examples(self) -> None:
        """
        Resets state of previous provisioning selection and creates a category-specific list of image examples
        that a user can select from.
        """
        if st.session_state.example_list_pull is True:
            self.reset_states_after_image_provisioning_list()

        st.session_state.example_captions = [
            s_img["label"]
            for s_img in CATEGORY_DESCR[st.session_state.category_option.value][
                "image_examples"
            ]
        ]  # get captions
        st.session_state.example_imgs = [
            Image.open(s_img["path"])
            for s_img in CATEGORY_DESCR[st.session_state.category_option.value][
                "image_examples"
            ]
        ]  # get images
        example_images_zip = dict(
            zip(st.session_state.example_captions, st.session_state.example_imgs)
        )
        img_selection = st.selectbox(
            f"Choose an image - {st.session_state.category_option.value}.",
            example_images_zip,
        )  # select image

        if st.session_state.example_list_pull is True:
            st.session_state.selected_img = example_images_zip[img_selection]
            st.session_state.show_input_img = True

    def create_image_provision_for_random_storage_pull(self) -> None:
        """
        Resets state of previous provisioning selection and pulls from local/cloud storage a category-specific list of
        image examples that a user can select from the list. Additionally, a button for re-running random selection is
        implemented together with the input option for the number of sampled images.
        """
        st.session_state.img_storage_list_pull = True  # TODO: Upload options other than storage pull temporarily turned off, remove this line when turning on
        if st.session_state.img_storage_list_pull is True:
            self.reset_states_after_image_provisioning_list()
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

            if st.session_state.img_storage_list_pull is True:
                st.session_state.selected_img = random_images_zip[
                    img_selection
                ]  # returns an image based on selection
                st.session_state.show_input_img = True

    def create_image_provision_for_manual_upload(self) -> None:
        """
        Resets state of previous provisioning selection and provides upload button for a user.
        Any RGB image can be uploaded.
        """
        if st.session_state.file_upload_pull is True:
            self.reset_states_after_image_provisioning_list()
        byte_img = st.file_uploader("Upload an image from a local disk.")
        if byte_img:
            if st.session_state.file_upload_pull is True:
                st.session_state.selected_img = Image.open(byte_img)
                st.session_state.show_input_img = True

    def create_similarity_search_filters(self) -> None:
        """
        Creates a set of similarity-search-specific filters - number of shown images an benchmark for
        minimum similarity value in %.
        """
        if st.session_state.show_input_img:
            st.markdown(
                "<h2 style='text-align: center;'>Input Image</h2>",
                unsafe_allow_html=True,
            )

            with st.expander("", expanded=True):
                col_search_1, col_search_2 = st.columns([1, 1])
                with col_search_1:
                    st.image(st.session_state.selected_img)
                with col_search_2:
                    st.markdown(
                        "<h4 style='text-align: center;'>Search Options</h4>",
                        unsafe_allow_html=True,
                    )
                    st.session_state.similar_img_number = st.number_input(
                        label="Insert a number of similar images to show.",
                        value=9,
                        min_value=1,
                        format="%i",
                    )
                    st.session_state.benchmark_similarity_value = st.number_input(
                        label="Insert a benchmark similarity value (in %).",
                        value=50,
                        min_value=0,
                        max_value=100,
                        format="%i",
                    )

                    logger.info("Similarity Search Button")
                    button = st.button(
                        "Find Similar Images", key="similar_images_found"
                    )

    def search_with_show(
        self,
        collection_name: MetricCollections,
        k: int,
        grid_nrow: int,
        benchmark: int,
        file,
    ) -> None:
        """
        Shows images in order of their similarity to the original input image.
        """
        best_images_dataset = (
            BestChoiceImagesDataset.get_best_choice_for_uploaded_image(
                client=self.metric_client,
                anchor=file,
                collection_name=collection_name,
                k=k,
                benchmark=benchmark,
            )
        )
        captions_dict = [
            {
                "file": r.payload["file"].split("/")[-1].split("\\")[-1],
                "class": r.payload["class"],
                "similarity": "{0:.2f}%".format(100 * round(r.score, 4)),
            }
            for r in best_images_dataset.results
        ]
        if best_images_dataset.similars:
            with st.expander("", expanded=True):
                results_text = f'Found {st.session_state.similar_img_number} images in the "{st.session_state.category_option.value}" category'
                st.markdown(
                    f"<h4 style='text-align: center;'>{results_text}</h4>",
                    unsafe_allow_html=True,
                )
                comment_text = f"In the top left corner of every image a similarity coefficient is presented - it shows a level of similarity between a given image and an input image."
                st.markdown(
                    f"<p style='text-align: center;'>{comment_text}</p>",
                    unsafe_allow_html=True,
                )
                col_nr = min(grid_nrow, len(best_images_dataset.similars))
                for i, col in enumerate(st.columns(col_nr)):
                    col_imgs = best_images_dataset.similars[i::col_nr]
                    col_imgs_captions_dict = captions_dict[i::col_nr]
                    with col:
                        for j, col_img in enumerate(col_imgs):
                            st.image(col_img)
                            st.markdown(
                                f"<p style='text-align: start; color: red; font-weight: bold;'>Similarity: {col_imgs_captions_dict[j]['similarity']}</p>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<p style='text-align: start; font-weight: bold;'>Class: {col_imgs_captions_dict[j]['class']}</p>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<p style='text-align: start; font-weight: bold;'>File: {col_imgs_captions_dict[j]['file']}</p>",
                                unsafe_allow_html=True,
                            )
                            st.write("")
                            st.write("")

        else:
            no_results_text = st.write(
                f"No images found for the similarity benchmark of {benchmark}%."
            )
            st.markdown(
                f"<h4 style='text-align: center; color: red;'>{no_results_text}</h4>",
                unsafe_allow_html=True,
            )

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
            self.reset_all_states_button()

    def create_image_load(self) -> None:
        """"""
        st.markdown(
            "<h2 style='text-align: center;'>Upload Options</h2>",
            unsafe_allow_html=True,
        )
        with st.expander("", expanded=True):
            st.markdown(
                "<h3 style='text-align: center;'>How would you like to add an image?</h3>",
                unsafe_allow_html=True,
            )

            # TODO: Upload options other than storage pull temporarily turned off, uncomment when turning on
            # provisioning_options = [
            #     "Example List",
            #     "Pull Randomly from the Storage",
            #     "File Upload",
            # ]

            col_image_load_1, col_image_load_2, col_image_load_3 = st.columns(3)

            # TODO: Upload options other than storage pull temporarily turned off, uncomment when turning on
            # with col_image_load_1:
            #     st.checkbox(
            #         f"{provisioning_options[0]}", value=False, key="example_list_pull"
            #     )
            #     self.create_image_provision_for_examples()

            with col_image_load_2:
                # TODO: Upload options other than storage pull temporarily turned off, uncomment when turning on
                # st.checkbox(
                #     f"{provisioning_options[1]}",
                #     value=False,
                #     key="img_storage_list_pull",
                # )
                self.create_image_provision_for_random_storage_pull()

            # TODO: Upload options other than storage pull temporarily turned off, uncomment when turning on
            # with col_image_load_3:
            #     st.checkbox(
            #         f"{provisioning_options[2]}", value=False, key="file_upload_pull"
            #     )
            #     self.create_image_provision_for_manual_upload()

            # checkbox_state_list = [
            #     st.session_state.example_list_pull,
            #     st.session_state.img_storage_list_pull,
            #     st.session_state.file_upload_pull,
            # ]
            # if sum(checkbox_state_list) > 1:
            #     st.write("")
            #     st.write("")
            #     st.markdown(
            #         "<h4 style='text-align: center; color: red;'>Only one checkbox can be active at a time.</h4>",
            #         unsafe_allow_html=True,
            #     )

    def run_app(self) -> None:
        logger.info("Set main graphical options.")
        st.set_page_config(page_title="visual-search.stxnext.pl", layout="wide")
        self.widget_formatting()

        logger.info("Create a title container.")
        self.create_title_containers()

        logger.info("Initialize states.")
        self.initialize_states()

        logger.info("Create Main Filters - till category search")
        self.create_main_filters()

        logger.info("Image Provisioning")
        if st.session_state.category_option:
            self.create_image_load()

        if st.session_state.category_option and st.session_state.selected_img:
            logger.info("Similarity Search Filters")
            self.create_similarity_search_filters()

            if st.session_state.similar_images_found:
                self.extract_similar_images()

        logger.info("GitHub Fork")

        with st.expander("", expanded=True):
            st.markdown(
                "<h3 style='text-align: start;'>Credits</h3>", unsafe_allow_html=True
            )
            fork_text = (
                f"Fork us on [GitHub]({INTERACTIVE_ASSETS_DICT['github_link']})."
            )
            contact_text = f"Want to talk about [Machine Learning Services]({INTERACTIVE_ASSETS_DICT['our_ml_site_link']}) visit our webpage."
            st.write(fork_text)
            st.write(contact_text)
