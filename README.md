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
      <a href="#further-material">Further Material</a>
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

We have implemented a RASA-based dialogue system prototype, which we have integrated into the JupyterLab environment. Using this natural-language interface, data analyses can be specified at a high abstraction level, which are then automatically mapped to executable program instructions that are generated in a Jupyter Notebook

<div align="center">
<img src="https://user-images.githubusercontent.com/2452384/169987155-1f5d3f52-dbba-4ff8-ae17-173930704919.png" alt="Implementation overview" width="885"></img>
</div>

We performed the evaluation on two tasks, namely semantic parsing and code generation using the ZIH HPC cluster. In both cases, the goal is to generate formal meaning representations from natural language input, that is, lambda-calculus expressions or Python code.

Experimental results show that on one benchmark the tree encoding performs better than the sequential encoding used by the original Transformer architecture.
To test whether the tree-encoded Transformer learns to predict the AST structure correctly, we looked at the exact match accuracy and token- and sequence-level precision and recall. The takeaway from the analysis of correctly predicted prefixes is that string literals have a significant impact on the quality of the prediction and that longer sequences are more difficult to predict. We also found that tree encoding gives an improvement of up to 3.0% when excluding string literals over sequential encoding.

<video src="https://user-images.githubusercontent.com/2452384/169987182-51bb52bc-9d56-4eac-9f90-2567252b1fc3.mp4" controls="controls"></video>

## Installation

### Prerequisites
* Python 3.8 (earlier versions may work as well)
* Conda (Miniconda3 is sufficient)

### Rasa Backend
1. Create and activate Python environment
   ```sh
   conda create -n pyenv_rasa python=3.8
   conda activate pyenv_rasa
   ```
   Alternatively, you can also use virtualenv (e.g. if you would like to use system site_packages during code inference)
   ```sh
   conda deactivate
   python3 -m virtualenv ~/pyenv_rasa
   source ~/pyenv_rasa/bin/activate # use this everywhere instead of "conda activate pyenv_rasa" below
   ```
   (if virtualenv is not installed: `pip3 install --user virtualenv` after `conda deactivate`)
2. Install Rasa
   <!-- Rasa 2.2.10 without fixed Tensorflow version causes numpy version conflicts when installing nl2codemodel -->
   ```sh
   pip3 install rasa==2.2.8 rasa-sdk==2.2.0 tensorflow==2.3.4
   ```
3. Clone repository (if not done yet)
   ```sh
   git clone https://github.com/SmartDataAnalytics/codeCAI.git
   cd codeCAI
   ```
4. Install nl2code model
   ```sh
   pip3 install -e nl2codemodel/
   ```
5. Install missing and incorrectly versioned dependencies
   ```sh
   pip3 install sentencepiece==0.1.95 torchmetrics==0.5.1
   ```
6. Download Rasa model
   ```sh
   mkdir -p rasa/models
   wget -P rasa/models --content-disposition 'https://cloudstore.zih.tu-dresden.de/index.php/s/c3PcAjpPX6AeZaF/download?path=%2F&files=20210726-155823_kmeans.tar.gz'
   ```
7. Download NL2Code vocabulary, grammar graph and checkpoint
   ```sh
   mkdir -p nl2codemodel/models
   wget -P nl2codemodel/models --content-disposition 'https://cloudstore.zih.tu-dresden.de/index.php/s/c3PcAjpPX6AeZaF/download?path=%2F&files=usecase-nd_vocab_src.model' 'https://cloudstore.zih.tu-dresden.de/index.php/s/c3PcAjpPX6AeZaF/download?path=%2F&files=usecase-nd_grammargraph.gpickle' 'https://cloudstore.zih.tu-dresden.de/index.php/s/c3PcAjpPX6AeZaF/download?path=%2F&files=last.ckpt' 
   ```                       
8. Test Rasa installation
   ```sh
   cd rasa
   export NL2CODE_CONF=$PWD/nl2code.yaml
   rasa run actions -vv
   ```  
   In a separate terminal (under `rasa` working directory):
   ```sh
   conda activate pyenv_rasa
   rasa shell -m models/20210726-155823_kmeans.tar.gz
   ```  
   You can use the examples in the `rasa/examples` directory, e.g.`k_means_clustering.csv` (or `.json`) for conversing with the assistant.

<p align="right">(<a href="#top">back to top</a>)</p>

### Jupyter Plugin
1. Create and activate Python environment using conda
   ```sh
   cd codecai_jupyter_nli
   conda env create -f environment.yml
   conda activate codecai_jupyter_nli
   ```
2. Install dependencies
   ```sh
   pip3 install -e .
   ```
3. Build the extension
   ```sh
   jlpm run build
   ```
   (or `jlpm run watch` for development, to automatically rebuild on changes)

## Usage
### codeCAI Rasa Backend
1. Activate Python environment
   ```sh
   conda activate pyenv_rasa
   ```
2. Change to the Rasa project directory 
   ```sh
   cd rasa
   ```
3. Run Rasa server (adjust port 8888 if taken by another application than JupyterLab)
   ```sh
   rasa run --enable-api --debug -m models/***.tar.gz --cors ["localhost:8888"]
   ```
4. Run Rasa action server (in a separate console window, in the `rasa` directory)
   ```sh
   conda activate pyenv_rasa
   cd rasa
   export NL2CODE_CONF=$PWD/nl2code.yaml
   rasa run actions -vv
   ```
<p align="right">(<a href="#top">back to top</a>)</p>

### codeCAI Jupyter Plugin
1. Activate Python environment using conda
   ```sh
   conda activate codecai_jupyter_nli
   ```
2. Change to Jupyter base directory (creating if necessary)
    ```sh
    mkdir -p ~/jupyter_root
    cd ~/jupyter_root
    ```
3. Run JupyterLab
   ```sh
   jupyter lab
   ```
4. Open dialog assistant
   * Open (or create) a Jupyter notebook
   * Press __Ctrl+Shift+C__ to open the Command Palette 

     (or select View → Activate Command Palette)
   * Type "__nli__"
   * Select "__Show codeCAI NLI__"
5. Use dialog assistant
   * Type an instruction (e.g. "Hi" or "Which ML methods do you know?") in the text field on the bottom of the "codeCAI NLI" side panel on the right.
     Working examples are found  `rasa/examples` directory, e.g.`k_means_clustering.csv` (or `.json`).
   * Press __Return__ or click the "__Send__" button

## Further Material
1. In the ScaDS.AI Living Lab lecture, we presented an overview of state-of-the-art language models for program synthesis, introduced some basic characteristics of these models, and discussed several of their limitations. 
   One possible direction of research that could help alleviate these limitations is the inclusion of structural knowledge - an attempt we have made in this regard and which we briefly introduced:

   <p align="center">
   <a href="https://www.youtube.com/watch?v=mto9XS1Bf1c"><img src="https://user-images.githubusercontent.com/5738212/170039371-f3f2d87d-eef5-4da2-8be4-f5750c98674b.png" alt="Language Models for Code Generation - ScaDS.AI Living Lab Lecture" style="width:538px;height:303px;margin:2ex;"></a>
   </p>

2. [codeCAI Poster - Generating Code from Natural Language](https://github.com/SmartDataAnalytics/codeCAI/files/8762855/19_Poster_Generating_Code_from_Natural_Language.pdf)


<p align="right">(<a href="#top">back to top</a>)</p>

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.
<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments
We thank the Center for Information Services and High Performance Computing (ZIH) at TU Dresden for generous allocations of computer time. We acknowledge the support of the following projects: ScaDS (01IS18026A), SPEAKER (FKZ 01MK20011A), JOSEPH (Fraunhofer Zukunftsstiftung), ML2R (FKZ 01 15 18038 A/B/C), MLwin (01IS18050 D/F), TAILOR (GA 952215).
<p align="right">(<a href="#top">back to top</a>)</p>
