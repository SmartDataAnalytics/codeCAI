import logging as log

from nltocode.grammar.grammargraphvisitor import AbstractGrammarGraphVisitor


class AstSeqCodeGenerator(AbstractGrammarGraphVisitor):
    def __init__(self, grammargraph):
        super(AstSeqCodeGenerator, self).__init__(grammargraph)

    def generate_code(self, seq):
        seq_copy = list(seq)
        try:
            ast_seq_list = self.visit_graph(seq_copy)
        except:
            log.info("Illegal AST sequence (encountered while serializing): %s", seq, exc_info=True)
            return None

        return ' '.join(ast_seq_list)

    def visit_graph(self, seq):
        path = []
        ast_seq_list = []

        i = 0
        while i < len(seq):
            node = seq[i]
            node_type = self.get_node_type(node)
            node_label = self.get_node_label(node)

            if node_type == 'strliteral':
                node = seq.pop(i)
                literals = ''
                while node != self.list_end_id:
                    strliteral = self.get_node_label(node)
                    literals += self.get_strliteral_repr(strliteral)
                    node = seq.pop(i)
                if literals.startswith('▁'):
                    literals = literals[1:]
                decoded_string = self.decode_string(literals)
                ast_seq_list.append(ascii(decoded_string) + '#strliteral')
                node_label = self.get_node_label(node)
                i -= 1

            if node not in (self.eos, self.sos):
                ast_seq_list.append(node_label)

            successor_edge_nodes = sorted(self.graph.successors(node), key=(self.get_node_order))

            if node_label != '<strtoken>':
                path.append(node)

            if len(successor_edge_nodes) > 0:
                next_edge = successor_edge_nodes[0]
                if self.get_node_label(next_edge) != '<t>':
                    path.append(next_edge)
            else:
                current_node = path.pop()

                while len(path) > 0:
                    current_edge = path[-1]
                    current_edge_type = self.get_node_type(current_edge)

                    if current_edge_type == "list" and self.is_valid_list(
                            current_edge) and current_node != self.list_end_id:
                        break
                    else:
                        path.pop()
                        current_edge_order = self.get_node_order(current_edge)
                        parent_node = path[-1]
                        parent_node_edges = sorted(self.graph.successors(parent_node), key=(self.get_node_order))
                        if len(parent_node_edges) > current_edge_order + 1:
                            next_edge = parent_node_edges[current_edge_order + 1]
                            path.append(next_edge)
                            break
                        else:
                            current_node = path.pop()
            i += 1

        return ast_seq_list
