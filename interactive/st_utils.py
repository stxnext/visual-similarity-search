import streamlit as st

from metrics.core import MetricClient


def initialize_state():
    if "client" not in st.session_state:
        # Main States - set at the beginning of session
        st.session_state.client = MetricClient()
        st.session_state.category_desc_option = None
        st.session_state.category_option = None
        st.session_state.upload_option = None

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
        st.session_state.grid_nrow_number = None

        # Search Completed State - set after "Find Similar Images" is completed
        st.session_state.similar_images_found = None


def reset_states_after_upload_option_button():
    st.session_state.show_input_img = None
    st.session_state.selected_img = None
    st.session_state.grid_nrow_number = None
    st.session_state.benchmark_similarity_value = None
    st.session_state.similar_img_number = None
    st.session_state.similar_images_found = None


def reset_all_states_button():
    st.session_state.category_desc_option = None
    st.session_state.category_option = None
    st.session_state.upload_option = None
    st.session_state.show_input_img = None
    st.session_state.selected_img = None
    st.session_state.grid_nrow_number = None
    st.session_state.benchmark_similarity_value = None
    st.session_state.similar_img_number = None
    st.session_state.similar_images_found = None


def widget_formatting():
    st.markdown(
        f"""
        <style>
        .row-widget.stRadio > label {{
            font-size: 18px;
            font-weight: bold;
        }}
        .row-widget.stSelectbox > label {{
            font-size: 18px;
            font-weight: bold;
        }}
        .stNumberInput > label {{
            font-size: 18px;
            font-weight: bold;
        }}
        .stFileUploader > label {{
            font-size: 18px;
            font-weight: bold;
        }}
        .big-font {{
            font-size:18px !important;
            font-weight:bold; 
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
