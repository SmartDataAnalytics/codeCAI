<div id="top"/>
<h1 align="center">codeCAI - Generating Code from Natural Language </h1>

<div align="center">
<img src="https://user-images.githubusercontent.com/5738212/170006685-b5275e1c-d857-4f16-aead-47e506cc5176.png" alt="codeCAI" width="600"></img>
</div>


<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-codecai">About codeCAI</a>
    </li>
    <li>
      <a href="#further-material">Further Material</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#jupyter-plugin">Jupyter Plugin</a></li>
        <li><a href="#rasa-backend">Rasa Backend</a></li>
      </ul>
    </li>
    <li>
       <a href="#usage">Usage</a>
       <ul>
         <li>
           <a href="#codecai-jupyter-plugin">codeCAI Jupyter Plugin</a>
         </li>
         <li>
           <a href="#codecai-rasa-backend">codeCAI Rasa Backend</a>
         </li>
      </ul>
    </li>
    <li>
      <a href="#license">License</a>
    </li>
    <li>
      <a href="#acknowledgments">Acknowledgments</a>
    </li>
  </ol>
</details>

## About codeCAI
codeCAI is a conversational assistant that enables analysts to specify data analyses using natural language, which are then translated into executable Python code statements.

The approach we used to realize an assistant capable of interpreting analytical instructions, is a Transformer-based language model with an adapted tree encoding scheme and a restrictive grammar model that learns to map natural language specifications to a tree-based representation of the output code. With this syntax-driven approach, we aim to enable the language model to capture hierarchical relationships in syntax trees representing Python code fragments.

We have implemented a RASA-based dialogue system, which we have integrated into the JupyterLab environment. Using this natural-language interface, data analyses can be specified at a high abstraction level, which are then automatically mapped to executable program instructions that are generated in a Jupyter Notebook

<div align="center">
<img src="https://user-images.githubusercontent.com/2452384/169987155-1f5d3f52-dbba-4ff8-ae17-173930704919.png" alt="Implementation overview" width="885"></img>
</div>

We performed the evaluation on two tasks, namely semantic parsing and code generation using the ZIH HPC cluster. In both cases, the goal is to generate formal meaning representations from natural language input, that is, lambda-calculus expressions or Python code.

Experimental results show that on one benchmark the tree encoding performs better than the sequential encoding used by the original Transformer architecture.
To test whether the tree-encoded Transformer learns to predict the AST structure correctly, we looked at the exact match accuracy and token- and sequence-level precision and recall. The takeaway from the analysis of correctly predicted prefixes is that string literals have a significant impact on the quality of the prediction and that longer sequences are more difficult to predict. We also found that tree encoding gives an improvement of up to 3.0% when excluding string literals over sequential encoding.

<video src="https://user-images.githubusercontent.com/2452384/169987182-51bb52bc-9d56-4eac-9f90-2567252b1fc3.mp4" controls="controls"></video>

## Further Material
1. In the ScaDS.AI Living Lab lecture, we presented an overview of state-of-the-art language models for program synthesis, introduced some basic characteristics of these models, and discussed several of their limitations. 
One possible direction of research that could help alleviate these limitations is the inclusion of structural knowledge - an attempt we have made in this regard and which we briefly introduced.

[![Language Models for Code Generation](https://user-images.githubusercontent.com/5738212/170039371-f3f2d87d-eef5-4da2-8be4-f5750c98674b.png)](https://www.youtube.com/watch?v=mto9XS1Bf1c "Language Models for Code Generation - ScaDS.AI Living Lab Lecture")

2. [codeCAI Poster](https://github.com/SmartDataAnalytics/codeCAI/files/8762855/19_Poster_Generating_Code_from_Natural_Language.pdf)


<p align="right">(<a href="#top">back to top</a>)</p>

## Installation

### Prerequisites
* Python 3.7 (Conda 4.9)
* Rasa 2.2

### Jupyter Plugin
1. Create and activate Python environment using conda
```sh
conda env create -f environment.yml
conda activate codecai_jupyter_nli
```
2. Install dependencies
```sh
pip install -e .
```
3. Create a link to the prebuilt output directory
```sh
jupyter labextension develop . --overwrite
```
4. Build the extension
```sh
jlpm run build
(or jlpm run watch)
```

### Rasa Backend
1. Create and activate Python environment using virtualenv
```sh
python3 -m virtualenv pyenv_rasa
(if virtualenv is not installed: pip install --user virtualenv)
source ~/pyenv_rasa/bin/activate
```
2. Install Rasa
```sh
pip3 install rasa
pip3 install rasa_core_sdk
```
3. Install nl2code model
```sh
cd codeCAI/rasa
pip install -e ../nl2codemodel/
```
<p align="right">(<a href="#top">back to top</a>)</p>

## Usage
### codeCAI Jupyter Plugin
1. Activate Python environment using conda
```sh
conda activate codecai_jupyter_nli
```
2. Run JupyterLab
```sh
jupyter lab
```

### codeCAI Rasa Backend
1. Activate Python environment
```sh
source ~/pyenv_rasa/bin/activate
```
2. Change to the Rasa project directory 
```sh
cd codeCAI/rasa
```
3. Run Rasa server
```sh
rasa run --enable-api --debug -m models/***.tar.gz --cors ["localhost:8888"]
```
4. Run Rasa action server
```sh
export NL2CODE_CONF=/path/to/nl2code.yaml
rasa run actions -vv
```
<p align="right">(<a href="#top">back to top</a>)</p>

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.
<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments
We thank the Center for Information Services and High Performance Computing (ZIH) at TU Dresden for generous allocations of computer time. We acknowledge the support of the following projects: ScaDS (01IS18026A), SPEAKER (FKZ 01MK20011A), JOSEPH (Fraunhofer Zukunftsstiftung), ML2R (FKZ 01 15 18038 A/B/C), MLwin (01IS18050 D/F), TAILOR (GA 952215).
<p align="right">(<a href="#top">back to top</a>)</p>
