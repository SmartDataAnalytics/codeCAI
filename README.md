# nl2code

Installation (Linux/CPU-only):

    conda env create -f environment-cpu.yml
    conda activate nltocode-cpu

    # Option 1: Development install
    pip install -e .
    
    # Option 2: Regular install
    python setup.py install

    # Option 3: Build wheel (for distribution)
    python setup.py bdist_wheel

    # Option 4: Docker (>=18.09)
    DOCKER_BUILDKIT=1 docker build -t nltocode .