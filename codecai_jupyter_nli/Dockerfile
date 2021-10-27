# syntax=docker/dockerfile:1.3
FROM continuumio/miniconda3:4.10.3p0 as nli-buildenv
ADD environment.yml /environment.yml
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    eval "$(conda shell.bash hook)" \
    && conda env create -f /environment.yml \
    && echo 'conda activate codecai_jupyter_nli' >> ~/.bashrc

FROM nli-buildenv as nli-build
ADD . /nli_build
RUN eval "$(conda shell.bash hook)" \
    && conda activate codecai_jupyter_nli \
    && cd /nli_build \
    && python setup.py bdist_wheel


FROM jupyter/scipy-notebook:lab-3.0.16
ARG NLI_VERSION=0.1.0
ENV JUPYTER_ENABLE_LAB=1
VOLUME [ "/home/jovyan/work" ]
WORKDIR /home/jovyan/work
COPY --from=nli-build /nli_build/dist/codecai_jupyter_nli-${NLI_VERSION}-py3-none-any.whl /tmp/
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/codecai_jupyter_nli-${NLI_VERSION}-py3-none-any.whl
