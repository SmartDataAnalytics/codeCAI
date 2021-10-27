# syntax=docker/dockerfile:1.3
FROM continuumio/miniconda3:4.10.3p0 as nltocode-buildenv
ADD environment-cpu.yml /environment-cpu.yml
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    eval "$(conda shell.bash hook)" \
    && conda env create -f /environment-cpu.yml \
    && echo 'conda activate nltocode-cpu' >> ~/.bashrc

FROM nltocode-buildenv as nltocode
ADD . /nltocode_build
RUN eval "$(conda shell.bash hook)" \
    && cd /nltocode_build \
    && git clean -xfd \
    && git diff \
    && conda activate nltocode-cpu \
    && python setup.py bdist_wheel \
    && pip install dist/nltocode-*.whl
