**Prerequisites**
- Python 3.7 (Conda 4.9)
- RASA 2.2

**Installing the codeCAI Jupyter Plugin**
- conda env create -f environment.yml
- conda activate codecai_jupyter_nli
- pip install -e .
- jupyter labextension develop . --overwrite
- jlpm run build
- (or jlpm run watch)
- jupyter lab

**Installing RASA-Backend**
-  python3 -m virtualenv pyenv_rasa
   (if virtualenv is not installed: pip install --user virtualenv)
- source ~/pyenv_rasa/bin/activate
- pip3 install rasa 
- pip3 install rasa_core_sdk
- cd codeCAI/rasa
- pip install -e ../nl2codemodel/
- rasa run --enable-api --debug -m models/***.tar.gz --cors ["localhost:8888"]
- export NL2CODE_CONF=/path/to/nl2code.yaml
- rasa run actions -vv
