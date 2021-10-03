import os
from typing import Any, Text, Dict, List

import torch
import yaml
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from scads_cai_prototype.beamsearch import BeamSearch
from scads_cai_prototype.generator.codegenerator import PythonCodeGenerator
from scads_cai_prototype.grammar.grammargraphloader import GrammarGraphLoader
from scads_cai_prototype.preprocessing.preprocinf import Preprocinf
from scads_cai_prototype.transformerinf import Transformer


class ActionNl2Code(Action):

    def __init__(self):
        nl2code_conf = os.getenv('NL2CODE_CONF')
        nl2code_yaml = open(nl2code_conf)
        parsed_nl2code_yaml = yaml.load(nl2code_yaml, Loader=yaml.FullLoader)

        self.preprocinf = Preprocinf(parsed_nl2code_yaml["preproc"]["vocabsrc-file"])

        self.model = Transformer.load_from_checkpoint(parsed_nl2code_yaml["model_conf"]["test-model-path"])
        grammargraph = GrammarGraphLoader(parsed_nl2code_yaml["model_conf"]["grammar-graph-file"]).load_graph()
        codegenerator = PythonCodeGenerator(grammargraph)

        self.beamsearch = BeamSearch(
            model=self.model,
            grammargraph=grammargraph,
            codegenerator=codegenerator,
            num_beams=parsed_nl2code_yaml["model_conf"]["num-beams"],
            disable_constraint_mask=parsed_nl2code_yaml["model_conf"]["disable-constraint-mask"],
            max_beam_length=parsed_nl2code_yaml["model_conf"]["max-beam-length"],
            max_num_predicted_results=parsed_nl2code_yaml["model_conf"]["max-num-predicted-results"],
            mode=parsed_nl2code_yaml["model_conf"]["beam-search-mode"],
            keep_invalid_results=parsed_nl2code_yaml["model_conf"]["keep-invalid-beamsearch-results"])

        print('action nl2code')

    def name(self) -> Text:
        return "action_nl2code"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        nl_input = tracker.latest_message.get('text')
        print('message', nl_input)

        nl_seq = self.preprocinf.preproc(nl_input)
        print('nl_seq', nl_seq)

        nl_seq = torch.tensor(nl_seq)
        nl_seq = nl_seq.view(-1, 1)

        if not self.model.withcharemb:
            char_seqs = None
        else:
            char_seqs = self.preprocinf.preproc_char_seq(nl_input)
            char_seq_pad = self.preprocinf.pad_char_seq(char_seqs, self.model.max_char_seq_len)
            char_seqs = char_seq_pad.view(-1, 1, self.model.max_char_seq_len)

        prediction = self.beamsearch.perform_beam_search(nl_seq, char_seqs, None)

        if prediction:
            _, _, code_snippet = prediction[0]
            print('code_snippet', code_snippet)
            dispatcher.utter_message(json_message={"nl_input": nl_input, "code_snippet": code_snippet})
        else:
            response = "Could not translate to code. Please try reformulating your question."
            print(response)
            dispatcher.utter_message(text=response)

        return []


class ActionOrganizeKMeansImports(Action):

    def __init__(self):
        print('action organize kmeans imports')

    def name(self) -> Text:
        return "action_organize_kmeans_imports"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        imports = "from sklearn.datasets import make_blobs\n" \
                  "import pandas as pd\n" \
                  "import numpy as np\n" \
                  "import matplotlib.pyplot as plt\n" \
                  "from sklearn.cluster import KMeans\n"

        dispatcher.utter_message(json_message={"nl_input": "Organize kmeans imports", "code_snippet": imports})

        return []


class ActionOrganizeDecisionTreeImports(Action):

    def __init__(self):
        print('action organize decision tree imports')

    def name(self) -> Text:
        return "action_organize_decision_tree_imports"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        imports = "from sklearn.datasets import load_iris\n" \
                  "from sklearn.tree import DecisionTreeClassifier\n" \
                  "from sklearn.tree import export_text"

        dispatcher.utter_message(
            json_message={"nl_input": "Organize decision tree imports", "code_snippet": imports})

        return []


# class ActionDebug(Action):
#
#     def __init__(self):
#         print('action debug')
#
#     def name(self) -> Text:
#         return "action_debug"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
