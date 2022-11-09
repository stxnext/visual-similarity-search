import streamlit as st

from loguru import logger

from interactive.st_utils import StatesManager, WidgetManager
from interactive.app_utils import ModuleManager


def main_streamlit(module_manager: ModuleManager):
    """
    Main Streamlit application.
    """
    logger.info("Create Main Filters - till category search")
    module_manager.create_main_filters()

    logger.info("Columnar split")
    col_image_1, col_image_2 = st.columns([1, 1])

    logger.info("Image Provisioning")
    with col_image_1:
        logger.info("Image Provisioning Type Selection")
        provisioning_options = [
            "Example List",
            "Pull Randomly from the Storage",
            "File Upload",
        ]
        module_manager.create_image_provisioning_options(
            provisioning_options=provisioning_options
        )
    with col_image_2:
        logger.info(
            "Image Selection - only if st.session_state.category_option of main filters was chosen beforehand."
        )
        if st.session_state.category_option is not None:
            if st.session_state.provisioning_options == provisioning_options[0]:
                module_manager.create_image_provision_for_examples()
            elif st.session_state.provisioning_options == provisioning_options[1]:
                module_manager.create_image_provision_for_random_storage_pull()
            elif st.session_state.provisioning_options == provisioning_options[2]:
                module_manager.create_image_provision_for_manual_upload()

    logger.info("Shows an Image selected in the provisioning stage")
    if st.session_state.show_input_img:
        st.header("Input Image")
        input_img_placeholder = st.empty()
        input_img_placeholder.image(st.session_state.selected_img)

    if st.session_state.category_option and st.session_state.selected_img:
        logger.info("Similarity Search Filters")
        module_manager.create_similarity_search_filters()

        logger.info("Similarity Search Button")
        if st.button("Find Similar Images"):
            module_manager.extract_similar_images()


if __name__ == "__main__":
    logger.info("Initialize modules.")
    widget_manager = WidgetManager()
    states_manager = StatesManager()
    module_manager = ModuleManager(states_manager=states_manager)

    logger.info("Set main graphical options.")
    widget_manager.widget_formatting()

    logger.info("Create a header with initial paragraph.")
    module_manager.create_header()

    logger.info("Initialize states.")
    states_manager.initialize_states()

    main_streamlit(module_manager=module_manager)
