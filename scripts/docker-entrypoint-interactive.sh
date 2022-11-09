#!/bin/sh


if [ "$TYPE" != "LOCAL" ]
then
    python ./scripts/preload_qdrant_data.py
fi

exec streamlit run interactive/search_app.py --server.port=$INTERACTIVE_PORT --server.address=0.0.0.0
