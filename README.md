1. Installation Jupyter Plugin in codeCAI/scads_jupyter_nli:
- conda env create -f environment.yml
- conda activate scads_jupyter_nli
- pip install -e .
- jupyter labextension develop . --overwrite
- jlpm run build
- (or jlpm run watch)
- jupyter lab

2. Installation RASA-Backend in codeCAI/rasa
- source ~/pyenv3.8.5_rasa/bin/activate
- pip install -e ../nl2codemodel/
- rasa run --enable-api --debug -m models/20211007-081137.tar.gz --cors ["localhost:8888"]
- export NL2CODE_CONF=/home/kthellmann/codeCAI/rasa/nl2code.yaml
- rasa run actions -vv
