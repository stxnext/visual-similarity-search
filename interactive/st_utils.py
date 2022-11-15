import streamlit as st

from loguru import logger
from interactive import GRID_NROW_NUMBER

from common.consts import INTERACTIVE_ASSETS_DICT


class StatesManager:
    """
    Class defining list of available application states.
    """

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

            # Option 2 States - Storage Images
            st.session_state.pull_random_img_number = None
            st.session_state.refresh_random_images = None
            st.session_state.random_captions = None
            st.session_state.random_imgs = None

            # Option 3 States - Uploaded File Images
            # Placeholder

            # All Options States - set when an input image has been selected
            st.session_state.show_input_img = None
            st.session_state.selected_img = None

            # Search and Output States - set when an input image has been selected and before "Find Similar Images" is run
            st.session_state.similar_img_number = None
            st.session_state.benchmark_similarity_value = None
            st.session_state.grid_nrow_number = GRID_NROW_NUMBER

            # Search Completed State - set after "Find Similar Images" is completed
            st.session_state.similar_images_found = None

    def reset_states_after_image_provisioning_list(self) -> None:
        """
        Reset all application states after radio selection for image provisioning.
        """
        st.session_state.show_input_img = None
        st.session_state.selected_img = None
        st.session_state.benchmark_similarity_value = None
        st.session_state.similar_img_number = None
        st.session_state.similar_images_found = None

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
        st.session_state.similar_images_found = None


class WidgetManager:
    """
    Class for creating new Streamlit widgets and changing style of existing ones.
    """

    def __init__(self) -> None:
        logger.info("Setting up page layout.")
        st.set_page_config(layout="wide")

    def widget_formatting(self) -> None:
        with open(INTERACTIVE_ASSETS_DICT["app_style_file"], "r") as f:
            st.markdown(f"<style>{f.read}</style>", unsafe_allow_html=True)
