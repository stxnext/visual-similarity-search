#!/bin/sh

exec streamlit run interactive/searchApp.py --server.port=$INTERACTIVE_PORT --server.address=0.0.0.0