import _ast
import ast
import logging as log

from networkx.classes import DiGraph
from networkx.readwrite import gpickle
from sentencepiece import SentencePieceProcessor


class GrammarGraphCreator(ast.NodeVisitor):
    def __init__(self, vocab_tgt: SentencePieceProcessor, vocab_src: SentencePieceProcessor, list_end_name='</l>',
                 root_edge_name='root', end_edge_name='end', all_strliteral_edges=False,
                 vocabsrc_add_dummy_prefix=True):
        self.vocab_src = vocab_src
        self.vocab_src_size = self.vocab_src.vocab_size()

        self.vocab_tgt = vocab_tgt
        self.vocab_tgt_size = self.vocab_tgt.vocab_size()

        self.sentence_start_name = self.vocab_tgt.IdToPiece(self.vocab_tgt.bos_id()).lstrip('\u2581')
        self.sentence_end_name = self.vocab_tgt.IdToPiece(self.vocab_tgt.eos_id()).lstrip('\u2581')
        self.pad_name = self.vocab_tgt.IdToPiece(self.vocab_tgt.pad_id()).lstrip('\u2581')
        self.list_end_name = list_end_name

        self.sentence_start_id = self.vocab_tgt.bos_id()
        self.sentence_end_id = self.vocab_tgt.eos_id()
        self.pad_id = self.vocab_tgt.pad_id()
        self.list_end_id = self.get_astnode_vocab_tgt_id(self.list_end_name)
        assert self.list_end_id != self.vocab_tgt.unk_id()

        self.root_edge_name = root_edge_name
        self.end_edge_name = end_edge_name

        self.all_strliteral_edges = all_strliteral_edges

        self.root_edge_id = self.sentence_start_name + "_" + self.root_edge_name
        self.end_edge_id = self.sentence_start_name + "_" + self.end_edge_name

        self.graph = DiGraph(
            sentence_start_id=self.sentence_start_id,
            sentence_end_id=self.sentence_end_id,
            pad_id=self.pad_id,
            list_end_id=self.list_end_id,
            root_edge_id=self.root_edge_id,
            end_edge_id=self.end_edge_id,
            vocabsrc_add_dummy_prefix=vocabsrc_add_dummy_prefix,
        )

        self.graph.add_node(self.sentence_start_id, label=self.sentence_start_name, type='special')
        self.graph.add_node(self.sentence_end_id, label=self.sentence_end_name, type='special')
        self.graph.add_node(self.list_end_id, label=self.list_end_name, type='special')

        self.graph.add_node(self.root_edge_id, label=self.root_edge_name, type='singleton', order=0)
        self.graph.add_node(self.end_edge_id, label=self.end_edge_name, type='singleton', order=1)

        self.graph.add_edge(self.sentence_start_id, self.root_edge_id)
        self.graph.add_edge(self.sentence_start_id, self.end_edge_id)
        self.graph.add_edge(self.end_edge_id, self.sentence_end_id)

        for vocab_id in range(1, len(self.vocab_src)):
            value_name = self.vocab_src.IdToPiece(vocab_id) + '#strliteral'
            self.graph.add_node(vocab_id + self.vocab_tgt_size, label=value_name, type='strliteral')

    def add_edgenode_objectnode_edge(self, edge_id, target_node):
        if isinstance(target_node, ast.AST):
            value_type = 'node'
            value_name = type(target_node).__name__
            value_id = self.get_astnode_vocab_tgt_id(value_name)
            self.add_edge_edgenode_objectnode(edge_id, value_id, value_name, value_type)
            self.visit(target_node)
        else:
            self.add_edgenode_literalnode_edge(edge_id, target_node)

    def add_edgenode_literalnode_edge(self, edge_id, target_node):
        representation = ascii(target_node)
        try:
            representation.encode("UTF-8")
        except:
            log.info("ERROR: Literal cannot be encoded as UTF-8: %s", representation)
            return
        if isinstance(target_node, (str, bytes)):
            value_type = 'strliteral'
            value_ids = self.get_astnode_vocab_src_id(target_node)
            for value_id in value_ids:
                if value_id == self.vocab_tgt_size:
                    representation = ''
                else:
                    representation = self.get_astnode_vocab_src_token(value_id)
                value_name = representation + "#" + value_type
                self.add_edge_edgenode_objectnode(edge_id, value_id, value_name, value_type)
        else:
            value_type = 'literal'
            value_name = representation + "#" + value_type
            value_id = self.get_astnode_vocab_tgt_id(value_name)
            self.add_edge_edgenode_objectnode(edge_id, value_id, value_name, value_type)

    def add_all_strliteral_edges(self):
        all_strliterals = [node for node in self.graph.nodes if self.graph.nodes[node]['type'] == 'strliteral']

        for node in self.graph.nodes:
            if self.graph.nodes[node]['type'] == 'list' and not self.graph.nodes[node]['is_valid_list']:
                for strliteral in all_strliterals:
                    if not self.graph.has_edge(node, strliteral):
                        self.graph.add_edge(node, strliteral)

    def add_edge_edgenode_objectnode(self, edge_id, value_id, value_name, value_type):
        if not self.graph.has_node(value_id):
            self.graph.add_node(value_id, label=value_name, type=value_type)

        if not self.graph.has_edge(edge_id, value_id):
            self.graph.add_edge(edge_id, value_id)

    def get_astnode_vocab_tgt_id(self, node_name):
        vocab_ids = self.vocab_tgt.Encode(node_name)

        if len(vocab_ids) != 1:
            parts = [self.vocab_tgt.DecodeIds([vocab_id]) for vocab_id in vocab_ids]
            log.info(
                "ERROR: Encountered token that cannot be encoded as one target vocab id: %s, Vocab-IDs: %s, Decoded: %s",
                node_name, vocab_ids, parts)
            raise ValueError

        vocab_id = vocab_ids[0]

        if vocab_id == self.vocab_tgt.unk_id():
            parts = [self.vocab_tgt.DecodeIds([vocab_id]) for vocab_id in vocab_ids]
            log.info("ERROR: Tgt vocabulary doesn't contain token %s, ID: %d, pieces: %s", node_name, vocab_id, parts)
            raise ValueError

        return vocab_id

    def get_astnode_vocab_src_id(self, node_name):
        if node_name == '':
            return [self.vocab_tgt_size]
        else:
            vocab_ids = self.vocab_src.Encode(node_name)
            if self.vocab_src.unk_id() in vocab_ids:
                log.info("Warning: Encountered unknown token while encoding literal %s [%s]", repr(node_name),
                         self.vocab_src.Decode(vocab_ids))
            return [vocab_id + self.vocab_tgt_size for vocab_id in vocab_ids if vocab_id != self.vocab_src.unk_id()]

    def get_astnode_vocab_src_token(self, value_id):
        vocab_id = value_id - self.vocab_tgt_size
        return self.vocab_src.IdToPiece(vocab_id)

    def add_str_token_node_and_field(self, edge_id, literal):
        node_name = '<strtoken>'
        node_type = 'node'
        node_id = self.get_astnode_vocab_tgt_id(node_name=node_name)

        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, label=node_name, type=node_type)

        if not self.graph.has_edge(edge_id, node_id):
            self.graph.add_edge(edge_id, node_id)

        field_name = '<t>'
        edge_id = node_name + "_" + field_name

        if not self.graph.has_node(edge_id):
            order = len(list(self.graph.successors(node_id)))
            self.graph.add_node(edge_id, label=field_name, type='list', order=order, is_valid_list=True)
            self.graph.add_edge(node_id, edge_id)

        self.add_edgenode_objectnode_edge(edge_id, literal)

        if not self.graph.has_edge(edge_id, self.list_end_id):
            self.graph.add_edge(edge_id, self.list_end_id)

    def save_graph(self, graph_path):
        gpickle.write_gpickle(self.graph, graph_path)


class PythonGrammarGraphCreator(GrammarGraphCreator):
    def __init__(self, vocab_tgt: SentencePieceProcessor, vocab_src: SentencePieceProcessor, list_end_name='</l>',
                 root_edge_name='root', end_edge_name='end', all_strliteral_edges=False,
                 vocabsrc_add_dummy_prefix=True):
        super(PythonGrammarGraphCreator, self).__init__(
            vocab_tgt,
            vocab_src,
            list_end_name=list_end_name,
            root_edge_name=root_edge_name,
            end_edge_name=end_edge_name,
            all_strliteral_edges=all_strliteral_edges,
            vocabsrc_add_dummy_prefix=vocabsrc_add_dummy_prefix,
        )

    def create_graph(self, data):
        error_indexes = []
        error_ids = []

        for index, row in data.iterrows():
            ast_obj = row['ast']
            ast_root_name = type(ast_obj).__name__
            ast_root_type = 'node'
            ast_root_id = self.get_astnode_vocab_tgt_id(ast_root_name)
            self.graph.add_node(ast_root_id, label=ast_root_name, type=ast_root_type)
            self.graph.add_edge(self.root_edge_id, ast_root_id)

            try:
                self.generic_visit(ast_obj)
            except:
                log.info("Error in row id %s with index %d, Snippet: %s, NL: %s", row['id'], index,
                         ascii(row['snippet']), ascii(row['nl']))
                error_indexes.append(index)
                error_ids.append(row['id'])

        if self.all_strliteral_edges:
            self.add_all_strliteral_edges()

        # log.info('Created Grammar Graph - Errors in lines: %s', error_indexes)
        log.info('Created Grammar Graph - Errors in snippets (id): %s', error_ids)

        log.debug('Node data: %s', self.graph.nodes.data())
        log.debug('Edge data: %s', self.graph.edges.data())

        return self.graph

    def generic_visit(self, node):
        node_name = type(node).__name__
        node_type = 'node'
        node_id = self.get_astnode_vocab_tgt_id(node_name)

        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, label=node_name, type=node_type)

        for field_name, value in ast.iter_fields(node):
            if isinstance(value, _ast.expr_context):
                continue

            edge_id = node_name + "_" + field_name

            if not self.graph.has_node(edge_id):
                order = len(list(self.graph.successors(node_id)))
                self.graph.add_node(edge_id, label=field_name, order=order)
                self.graph.add_edge(node_id, edge_id)

            if isinstance(value, list):
                self.graph.nodes[edge_id]['type'] = 'list'
                self.graph.nodes[edge_id]['is_valid_list'] = True
                for item in value:
                    if isinstance(item, (str, bytes)):
                        self.add_str_token_node_and_field(edge_id, item)
                    else:
                        self.add_edgenode_objectnode_edge(edge_id, item)

                if not self.graph.has_edge(edge_id, self.list_end_id):
                    self.graph.add_edge(edge_id, self.list_end_id)
            else:
                if isinstance(value, (str, bytes)):
                    self.graph.nodes[edge_id]['type'] = 'list'
                    self.graph.nodes[edge_id]['is_valid_list'] = False
                    self.add_edgenode_objectnode_edge(edge_id, value)
                    if not self.graph.has_edge(edge_id, self.list_end_id):
                        self.graph.add_edge(edge_id, self.list_end_id)
                else:
                    if 'type' not in self.graph.nodes[edge_id]:
                        self.graph.nodes[edge_id]['type'] = 'singleton'
                    self.add_edgenode_objectnode_edge(edge_id, value)


class AsdlGrammarGraphCreator(GrammarGraphCreator):
    def __init__(self, vocab_tgt: SentencePieceProcessor, vocab_src: SentencePieceProcessor, grammar,
                 list_end_name='</l>', root_edge_name='root', end_edge_name='end', all_strliteral_edges=False,
                 vocabsrc_add_dummy_prefix=True):
        super(AsdlGrammarGraphCreator, self).__init__(
            vocab_tgt,
            vocab_src,
            list_end_name=list_end_name,
            root_edge_name=root_edge_name,
            end_edge_name=end_edge_name,
            all_strliteral_edges=all_strliteral_edges,
            vocabsrc_add_dummy_prefix=vocabsrc_add_dummy_prefix,
        )
        self.grammar = grammar

    def create_graph(self, data):
        error_indexes = []
        error_ids = []

        for index, row in data.iterrows():
            ast_obj = row['ast']
            ast_root_name = ast_obj.production.constructor.name
            ast_root_type = 'node'
            ast_root_id = self.get_astnode_vocab_tgt_id(ast_root_name)

            self.graph.add_node(ast_root_id, label=ast_root_name, type=ast_root_type)
            self.graph.add_edge(self.root_edge_id, ast_root_id)

            try:
                self.generic_visit(ast_obj)
            except:
                log.info("Error in row id %s with index %d, Snippet: %s, NL: %s", row['id'], index,
                         ascii(row['snippet']), ascii(row['nl']))
                error_indexes.append(index)
                error_ids.append(row['id'])

        if self.all_strliteral_edges:
            self.add_all_strliteral_edges()

        # log.info('Created Grammar Graph - Errors in lines: %s', error_indexes)
        log.info('Created Grammar Graph - Errors in snippets (id): %s', error_ids)

        log.debug('Node data: %s', self.graph.nodes.data())
        log.debug('Edge data: %s', self.graph.edges.data())

        return self.graph

    def generic_visit(self, asdl_ast):
        node_name = asdl_ast.production.constructor.name
        node_type = 'node'
        node_id = self.get_astnode_vocab_tgt_id(node_name)

        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, label=node_name, type=node_type)

        for field in asdl_ast.fields:
            edge_label = field.type.name + ('*-' if field.cardinality == 'multiple' else '-') + field.name
            edge_id = node_name + "_" + edge_label

            if not self.graph.has_node(edge_id):
                order = len(list(self.graph.successors(node_id)))
                self.graph.add_node(edge_id, label=edge_label, order=order)
                self.graph.add_edge(node_id, edge_id)

            if self.grammar.is_composite_type(field.type):
                if field.cardinality in ['single', 'optional']:
                    self.graph.nodes[edge_id]['type'] = 'singleton'
                    self.generic_visit(field.value)
                    tgt_id = self.get_astnode_vocab_tgt_id(field.value.production.constructor.name)
                    if not self.graph.has_edge(edge_id, tgt_id):
                        self.graph.add_edge(edge_id, tgt_id)
                else:
                    if field.cardinality == 'multiple':
                        self.graph.nodes[edge_id]['type'] = 'list'
                        self.graph.nodes[edge_id]['is_valid_list'] = True
                        children = field.value
                        for val in children:
                            self.generic_visit(val)
                            child_node_id = self.get_astnode_vocab_tgt_id(val.production.constructor.name)
                            if not self.graph.has_edge(edge_id, child_node_id):
                                self.graph.add_edge(edge_id, child_node_id)

                        if not self.graph.has_edge(edge_id, self.list_end_id):
                            self.graph.add_edge(edge_id, self.list_end_id)
            else:
                if isinstance(field.value, (str, bytes)):
                    self.graph.nodes[edge_id]['type'] = 'list'
                    self.graph.nodes[edge_id]['is_valid_list'] = False
                    self.add_edgenode_literalnode_edge(edge_id, field.value)
                    if not self.graph.has_edge(edge_id, self.list_end_id):
                        self.graph.add_edge(edge_id, self.list_end_id)
                else:
                    if 'type' not in self.graph.nodes[edge_id]:
                        self.graph.nodes[edge_id]['type'] = 'singleton'
                        self.add_edgenode_literalnode_edge(edge_id, field.value)


class PythonASTNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.ast_sequence = []

    def generic_visit(self, node):
        node_name = type(node).__name__
        self.ast_sequence.append(node_name)

        for field, value in ast.iter_fields(node):
            if isinstance(value, _ast.expr_context):
                continue

            if isinstance(value, list):
                for item in value:
                    if isinstance(item, (str, bytes)):
                        self.ast_sequence.append("<strtoken>")
                    self.handle_fieldvalue(item)
                self.ast_sequence.append('</l>')
            else:
                self.handle_fieldvalue(value)

    def handle_fieldvalue(self, value):
        if isinstance(value, ast.AST):
            self.visit(value)
        else:
            representation = ascii(value)
            if isinstance(value, (str, bytes)):
                suffix = "strliteral"
            else:
                suffix = "literal"

            value_name = representation + "#" + suffix

            try:
                value_name.encode("UTF-8")
            except:
                raise ValueError("Literal cannot be encoded as UTF-8: %s" % value_name)

            self.ast_sequence.append(value_name)

            if isinstance(value, (str, bytes)):
                self.ast_sequence.append('</l>')


class AsdlASTNodeVisitor:
    def __init__(self, grammar):
        self.ast_sequence = []
        self.grammar = grammar

    def generic_visit(self, asdl_ast):
        # print('AST NODE:',asdl_ast.production.constructor.name)
        self.ast_sequence.append(asdl_ast.production.constructor.name)

        for field in asdl_ast.fields:
            if self.grammar.is_composite_type(field.type):
                if field.cardinality == 'single':
                    # print('EDGE NODE:', field.type.name,'-',field.name)
                    self.generic_visit(field.value)
                else:
                    if field.value is not None:
                        if field.cardinality == 'multiple':
                            # print('EDGE NODE:', field.type.name,'*-',field.name)
                            for val in field.value:
                                self.generic_visit(val)
                            # print('</li>')
                            self.ast_sequence.append('</l>')
                        elif field.cardinality == 'optional':
                            # print('EDGE NODE:', field.type.name,'-',field.name)
                            self.generic_visit(field.value)
                        else:
                            pass
                            # print('EDGE NODE:', field.type.name,'-',field.name)
                    else:
                        pass
                        # print('EDGE NODE:', field.type.name,'-',field.name)
            else:
                # print('EDGE NODE:', field.type.name,'-',field.name)
                # print('STR LITERAL',field.value)

                if isinstance(field.value, (str, bytes)):
                    self.ast_sequence.append(ascii(field.value) + '#strliteral')
                    self.ast_sequence.append('</l>')
                else:
                    self.ast_sequence.append(ascii(field.value) + '#literal')


class AsdlASTSequenceCreator:
    def __init__(self, grammar):
        self.grammar = grammar

    def visit_asdl_nodes(self, ast_):
        try:
            asdl_node_visitor = AsdlASTNodeVisitor(self.grammar)
            asdl_node_visitor.generic_visit(ast_)
            ast_sequence = asdl_node_visitor.ast_sequence
            return ast_sequence
        except:
            return None

    def create_sequence(self, data):
        return data.apply(lambda ast: self.visit_asdl_nodes(ast))


class PythonASTSequenceCreator:
    def __init__(self):
        pass

    def visit_ast_nodes(self, ast_):
        try:
            ast_node_visitor = PythonASTNodeVisitor()
            ast_node_visitor.generic_visit(ast_)
            return ast_node_visitor.ast_sequence
        except:
            return None

    def create_sequence(self, data):
        return data.apply(lambda ast: self.visit_ast_nodes(ast))
