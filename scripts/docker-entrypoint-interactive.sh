#!/bin/sh

if [ "$TYPE" != "LOCAL" ]
then
    poetry run python ./scripts/preload_qdrant_data.py
fi

poetry run python -m streamlit run interactive/search_app.py --server.port=$INTERACTIVE_PORT --server.address=0.0.0.0
