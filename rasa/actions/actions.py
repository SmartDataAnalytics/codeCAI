from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import torch
import yaml

import os

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

         model = Transformer.load_from_checkpoint(parsed_nl2code_yaml["model_conf"]["test-model-path"])
         grammargraph = GrammarGraphLoader(parsed_nl2code_yaml["model_conf"]["grammar-graph-file"]).load_graph()
         codegenerator = PythonCodeGenerator(grammargraph)

         self.beamsearch = BeamSearch(
             model=model,
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
         print('message',nl_input)

         nl_seq = self.preprocinf.preproc(nl_input)
         print('nl_seq', nl_seq)

         nl_seq = torch.tensor(nl_seq)
         nl_seq = nl_seq.view(-1, 1)

         prediction = self.beamsearch.perform_beam_search(nl_seq, None)
         _, _, code_snippet = prediction[0]
         print('code_snippet', code_snippet)

         dispatcher.utter_message(text=code_snippet)

         return []


class ActionOrganizeKMeansImports(Action):

    def __init__(self):
        print('action organize kmeans imports')

    def name(self) -> Text:
        return "action_organize_kmeans_imports"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="from sklearn.datasets import make_blobs\n"
                                      "import pandas as pd\n"
                                      "import numpy as np\n"
                                      "import matplotlib.pyplot as plt\n"
                                      "from sklearn.cluster import KMeans\n")

        return []

    class ActionOrganizeDecisionTreeImports(Action):

        def __init__(self):
            print('action organize decision tree imports')

        def name(self) -> Text:
            return "action_organize_decision_tree_imports"

        def run(self, dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            dispatcher.utter_message(text="from sklearn.datasets import load_iris\n"
                                          "from sklearn.tree import DecisionTreeClassifier\n"
                                          "from sklearn.tree import export_text")

            return []

    class ActionDebug(Action):

        def __init__(self):
            print('action debug')

        def name(self) -> Text:
            return "action_debug"

        def run(self, dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            dispatcher.utter_message(text="Hello World!")

            return []