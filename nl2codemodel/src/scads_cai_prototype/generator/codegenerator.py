import ast
import logging as log
import sys
import traceback

import astor

from scads_cai_prototype.grammar.grammargraphvisitor import AbstractGrammarGraphVisitor


class PythonCodeGenerator(AbstractGrammarGraphVisitor):
    def __init__(self, grammargraph, validate_parsability=True):
        super(PythonCodeGenerator, self).__init__(grammargraph)
        self.validate_parsability = validate_parsability

    def generate_code(self, seq):
        seq_copy = list(seq)
        try:
            ast_dump = self.visit_graph(seq_copy)
        except:
            log.info("Illegal AST sequence (encountered while serializing): %s", seq, exc_info=True)
            return None
        try:
            ast_obj = eval(ast_dump, {}, ast.__dict__)
        except:
            log.info("Illegal AST dump: %s", ast_dump)
            log.info("Original AST sequence: %s", seq)
            log.info("Modified AST sequence: %s", seq_copy, exc_info=True)
            traceback.print_exc(file=sys.stdout)
            return None

        if not isinstance(ast_obj, ast.AST):
            log.info("AST dump evaluation result is not an AST: %s", ast_obj)
            log.info("AST dump: %s", ast_dump)
            log.info("Original AST sequence: %s", seq)
            log.info("Modified AST sequence: %s", seq_copy)
            return None

        try:
            code = astor.to_source(ast_obj)
        except:
            log.info("Illegal AST object: %s", ast.dump(ast_obj))
            log.info("Original AST sequence: %s", seq)
            log.info("Modified AST sequence: %s", seq_copy)
            log.info("AST dump %s:", ast_dump, exc_info=True)
            return None

        if self.validate_parsability:
            try:
                ast.parse(code)
            except:
                log.info('Generated code not parsable: """')
                log.info(code)
                log.info('"""')
                log.info("AST object: %s", ast.dump(ast_obj))
                log.info("Original AST sequence: %s", seq)
                log.info("Modified AST sequence: %s", seq_copy)
                log.info("AST dump %s:", ast_dump, exc_info=True)
                return None
        return code

    def visit_graph(self, seq):
        path = []
        ast_seq_rep = ''

        i = 0
        while i < len(seq):
            node = seq[i]
            node_type = self.get_node_type(node)
            node_label = self.get_node_label(node)

            if node_label != '<strtoken>':
                path.append(node)

            if node_type == 'literal':
                literal = node_label
                ast_seq_rep += self.get_literal_repr(literal)
            if node_type == 'strliteral':
                node = seq.pop(i)
                literals = ''
                while node != self.list_end_id:
                    strliteral = self.get_node_label(node)
                    # ast_seq_rep += get_strliteral_repr(strliteral)
                    literals += self.get_strliteral_repr(strliteral)
                    node = seq.pop(i)
                decoded_string = self.decode_string(literals)
                ast_seq_rep += ascii(decoded_string)
                i -= 1
            elif node_type == 'node':
                if node_label != '<strtoken>':  # TODO <strtoken>
                    ast_seq_rep += node_label + '('
                    # print("Node:", get_node_label(node), "(")

            successor_edge_nodes = sorted(self.graph.successors(node), key=(self.get_node_order))

            if len(successor_edge_nodes) > 0:
                next_edge = successor_edge_nodes[0]
                if self.get_node_label(next_edge) != '<t>':  # TODO <t>
                    path.append(next_edge)
                if self.get_node_label(next_edge) not in ['root', 'end', '<t>']:  # TODO <t>
                    ast_seq_rep += self.get_node_label(next_edge) + '='
                    # print("NEXT EDGE:", get_node_label(next_edge), '=')
                    if self.get_node_type(next_edge) == 'list':
                        if self.is_valid_list(next_edge):
                            ast_seq_rep += '['
                            # print("NEXT EDGE IS LIST:", '[')
                        # elif seq[i + 1] == self.list_end_id:
                        # seq.insert(i + 1, self.empty_strliteral)
            else:
                current_node = path.pop()
                # print('CURRENT NODE', get_node_label(current_node))
                # Gt / Add etc.
                if (self.get_node_type(current_node) == 'node') and (self.get_node_label(current_node) != '<strtoken>'):
                    # print("POP", get_node_label(current_node), ')')
                    ast_seq_rep += ')'

                while len(path) > 0:
                    current_edge = path[-1]
                    # print('PATH[-1] current edge', current_edge)
                    current_edge_type = self.get_node_type(current_edge)

                    if (current_edge_type == "list") and (self.is_valid_list(current_edge)):
                        if current_node == self.list_end_id:
                            path.pop()
                            ast_seq_rep += ']'
                            # print("LIST END", get_node_label(current_node), ']')
                            current_edge_order = self.get_node_order(current_edge)
                            parent_node = path[-1]
                            parent_node_edges = sorted(self.graph.successors(parent_node), key=(self.get_node_order))
                            if len(parent_node_edges) > current_edge_order + 1:
                                next_edge = parent_node_edges[current_edge_order + 1]
                                path.append(next_edge)
                                ast_seq_rep += ',' + self.get_node_label(next_edge) + '='
                                # print("NEXT EDGE:", ',', get_node_label(next_edge), '=')
                                if self.get_node_type(next_edge) == 'list':
                                    ast_seq_rep += '['
                                    # print("NEXT EDGE IS LIST:", '[')
                                break
                            else:
                                current_node = path.pop()
                                ast_seq_rep += ')'
                                # print('POP PARENT NODE AFTER REACHING LIST END ', get_node_label(current_node), ')')
                        else:
                            if i + 1 < len(seq) and seq[i + 1] != self.list_end_id:
                                ast_seq_rep += ', '
                                # print("LIST NEXT EL", ",")
                            break
                    else:
                        path.pop()
                        current_edge_order = self.get_node_order(current_edge)
                        parent_node = path[-1]
                        parent_node_edges = sorted(self.graph.successors(parent_node), key=(self.get_node_order))
                        if len(parent_node_edges) > current_edge_order + 1:
                            next_edge = parent_node_edges[current_edge_order + 1]
                            path.append(next_edge)
                            if self.get_node_label(next_edge) not in ['end']:
                                ast_seq_rep += ',' + self.get_node_label(next_edge) + '='
                                # print("NEXT EDGE:", ',', get_node_label(next_edge), '=')
                                if self.get_node_type(next_edge) == 'list':
                                    if self.is_valid_list(next_edge):
                                        ast_seq_rep += '['
                                        # print("NEXT EDGE IS LIST:", '[')
                                    elif seq[i + 1] == self.list_end_id:
                                        ast_seq_rep += ascii('')
                            break
                        else:
                            current_node = path.pop()
                            # FIXME: Sollte hier bei Literalen eine schließende Klammer ausgegeben werden?
                            if self.get_node_type(current_node) != 'special':
                                ast_seq_rep += ')'
                                # print("SINGLETON END", get_node_label(current_node), ')')
            i += 1

        return ast_seq_rep
