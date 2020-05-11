#!/bin/bash
virtualenv .venv -p /usr/local/bin/python3
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
deactivate
