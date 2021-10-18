import ast
import copy
import os

import pandas as pd

from asdl.lang.lambda_dcs.logical_form import parse_lambda_expr, logical_form_to_ast
from nltocode.grammar.grammargraph import PythonASTSequenceCreator, AsdlASTSequenceCreator


class PreprocDataProvider:
    def __init__(self, data_path, ast_seq_creator=None):
        self.data_path = data_path
        self.data = None
        self.ast_seq_creator = ast_seq_creator

    def provide_data(self, columns) -> pd.DataFrame:
        if self.data is None:
            self.data = self.load_data(self.data_path)
            self.prepare_data(self.data)

        return self.data[columns].copy(deep=False)

    def load_data(self, path):
        is_jsonl = path.lower().endswith('.jsonl')
        if not os.path.isfile(path):
            raise ValueError("File '%s' does not exist" % path)
        return pd.read_json(path, lines=is_jsonl, dtype=False)

    def prepare_ast(self, data):
        return data['snippet'].apply(lambda row: self.ast_parse(row))

    def prepare_nl(self, data):
        return data['intent'] if 'rewritten_intent' not in data.columns else data.apply(
            lambda row: row['intent'] if pd.isna(row['rewritten_intent']) else row['rewritten_intent'],
            axis=1)

    def prepare_id(self, data: pd.DataFrame):
        if 'id' in data.columns:
            return data['id']
        else:
            return data.reset_index().index

    def prepare_ast_seq_list(self, data):
        return self.ast_seq_creator.create_sequence(data['ast'])

    def prepare_ast_seq_list_no_str(self, ast_seq_list):
        if ast_seq_list is None:
            return None
        return ast_seq_list.apply(lambda ast_seq: [el for el in ast_seq if not el.endswith('#strliteral')])

    def prepare_ast_seq_list_str(self, ast_seq_list):
        if ast_seq_list is None:
            return None
        # return ast_seq_list.apply(lambda ast_seq: [el[:-len("#strliteral")] for el in ast_seq if el.endswith('#strliteral')])
        return ast_seq_list.apply(
            lambda ast_seq: [eval(el[:-len("#strliteral")]) for el in ast_seq if el.endswith('#strliteral')])

    def prepare_ast_seq(self, ast_seq_list):
        return ast_seq_list.apply(lambda ast_seq: " ".join(ast_seq))


class PreprocTrainDataProvider(PreprocDataProvider):
    def __init__(self, data_path, ast_seq_creator):
        super(PreprocTrainDataProvider, self).__init__(data_path, ast_seq_creator=ast_seq_creator)

    def prepare_data(self, data):
        data['id'] = self.prepare_id(data)
        data['nl'] = self.prepare_nl(data)
        data['ast'] = self.prepare_ast(data)
        data.dropna(inplace=True, subset=['id', 'nl', 'ast'])
        data['ast_seq_list'] = self.prepare_ast_seq_list(data)
        data['ast_seq_list_no_str'] = self.prepare_ast_seq_list_no_str(data['ast_seq_list'])
        data['ast_seq_list_str'] = self.prepare_ast_seq_list_str(data['ast_seq_list'])
        data['ast_seq'] = self.prepare_ast_seq(data['ast_seq_list'])
        data.dropna(inplace=True, subset=['ast_seq_list', 'ast_seq_list_no_str', 'ast_seq_list_str', 'ast_seq'])


class PreprocTrainPythonDataProvider(PreprocTrainDataProvider):
    def __init__(self, data_path):
        super(PreprocTrainPythonDataProvider, self).__init__(data_path, ast_seq_creator=PythonASTSequenceCreator())

    def ast_parse(self, snippet):
        return ast_parse_python(snippet)


class PreprocTrainLambdaDCSDataProvider(PreprocTrainDataProvider):
    def __init__(self, data_path, grammar, reorder_predicates=True):
        super(PreprocTrainLambdaDCSDataProvider, self).__init__(data_path,
                                                                ast_seq_creator=AsdlASTSequenceCreator(grammar=grammar))
        self.grammar = grammar
        self.reorder_predicates = reorder_predicates

    def ast_parse(self, snippet):
        return ast_parse_lambda_dcs(self.grammar, self.reorder_predicates, snippet)


class PreprocTestDataProvider(PreprocDataProvider):
    def __init__(self, data_path, ast_seq_creator=None):
        super(PreprocTestDataProvider, self).__init__(data_path, ast_seq_creator=ast_seq_creator)

    def prepare_data(self, data):
        data['id'] = self.prepare_id(data)
        data['nl'] = self.prepare_nl(data)
        data['ast'] = self.prepare_ast(data)
        data['ast_seq_list'] = self.prepare_ast_seq_list(data)
        data['ast_seq'] = self.prepare_ast_seq(data['ast_seq_list'])


class PreprocTestPythonDataProvider(PreprocTestDataProvider):
    def __init__(self, data_path):
        super(PreprocTestPythonDataProvider, self).__init__(data_path, ast_seq_creator=PythonASTSequenceCreator())

    def ast_parse(self, snippet):
        return ast_parse_python(snippet)


class PreprocTestLambdaDCSDataProvider(PreprocTestDataProvider):
    def __init__(self, data_path, grammar, reorder_predicates=True):
        super(PreprocTestLambdaDCSDataProvider, self).__init__(data_path,
                                                               ast_seq_creator=AsdlASTSequenceCreator(grammar=grammar))
        self.grammar = grammar
        self.reorder_predicates = reorder_predicates

    def ast_parse(self, snippet):
        return ast_parse_lambda_dcs(self.grammar, self.reorder_predicates, snippet)


def ast_parse_python(snippet):
    try:
        return ast.parse(snippet)
    except:
        print('Parse error in snippet %s' % ascii(snippet))
        # traceback.print_exc(file=sys.stdout)
        return None


def get_canonical_order_of_logical_form_lambda_dcs(lf):
    lf_copy = copy.deepcopy(lf)

    def _order(_lf):
        if _lf.name in ('and', 'or'):
            child_list_1 = sorted([x for x in _lf.children if x.name in ['flight', 'from', 'to']],
                                  key=lambda x: x.name)
            child_list_2 = sorted([x for x in _lf.children if x.name not in ['flight', 'from', 'to']],
                                  key=lambda x: x.name)
            _lf.children = child_list_1 + child_list_2

        for child in _lf.children:
            _order(child)

    _order(lf_copy)

    return lf_copy


def ast_parse_lambda_dcs(grammar, reorder_predicates, snippet):
    lf = parse_lambda_expr(snippet)
    assert lf.to_string() == snippet
    if reorder_predicates:
        ordered_lf = get_canonical_order_of_logical_form_lambda_dcs(lf)
        assert ordered_lf == lf
        lf = ordered_lf
    return logical_form_to_ast(grammar, lf)
