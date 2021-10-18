import logging as log

from networkx import DiGraph


class AbstractGrammarGraphVisitor:
    def __init__(self, graph: DiGraph):
        self.graph = graph
        graph_attr = self.graph.graph
        self.list_end_id = graph_attr['list_end_id']
        self.sos = graph_attr['sentence_start_id']
        self.eos = graph_attr['sentence_end_id']

        if 'vocabsrc_add_dummy_prefix' in graph_attr:
            self.vocabsrc_add_dummy_prefix = graph_attr['vocabsrc_add_dummy_prefix']
        else:
            self.vocabsrc_add_dummy_prefix = True

        self.empty_strliteral = self.get_empty_strliteral()

    def get_node_order(self, node):
        return self.graph.nodes[node]['order']

    def get_node_label(self, node):
        try:
            return self.graph.nodes[node]['label']
        except:
            raise

    def get_node_type(self, node):
        try:
            return self.graph.nodes[node]['type']
        except:
            raise

    def get_empty_strliteral(self):
        for node in self.graph.nodes:
            if self.is_empty_strliteral(node):
                return node
        return None

    def is_empty_strliteral(self, node):
        if self.vocabsrc_add_dummy_prefix:
            return self.graph.nodes[node]['label'] == '▁#strliteral'
        else:
            return self.graph.nodes[node]['label'] == '#strliteral'

    def is_valid_list(self, node):
        try:
            return self.graph.nodes[node]['is_valid_list']
        except:
            raise

    def get_literal_repr(self, literal):
        try:
            return literal[:literal.rfind('#literal')]
        except:
            raise

    def get_strliteral_repr(self, strliteral):
        try:
            return strliteral[:strliteral.rfind('#strliteral')]
        except:
            raise

    def compute_allowed_tokens(self, edgenode, preceding_node):
        if edgenode is not None:
            next_edge_successors = self.graph.successors(edgenode)

            if self.get_node_type(preceding_node) == 'strliteral':
                next_edge_successors = (n for n in next_edge_successors if self.get_node_label(n) != "None#literal")

            allowed_tokens = sorted(next_edge_successors)

            if 0 in allowed_tokens:
                log.info("Warning: Encountered <unk> as child of edge %s", edgenode)

            if self.graph.nodes[edgenode]['type'] == 'list':
                if not self.list_end_id in allowed_tokens:
                    log.info("Warning: Missing list end as child of edge %s", edgenode)
        else:
            allowed_tokens = []
        return allowed_tokens

    def decode_string(self, encoded_string):
        if self.vocabsrc_add_dummy_prefix and encoded_string.startswith('▁'):
            encoded_string = encoded_string[1:]
        return encoded_string.replace('▁', ' ')


class GrammarGraphVisitor(AbstractGrammarGraphVisitor):
    def __init__(self, graph: DiGraph):
        super(GrammarGraphVisitor, self).__init__(graph)

    def visit_graph_edge_order_path(self, seq, sample_id):
        path = []
        edge_order_seq = []
        edge_order_path = []
        allowed_tokens = []

        # Stats
        edge_counts = {}
        max_depth = 0

        # log.debug('ID: %s', sample_id)

        for node in seq:
            node_label = self.get_node_label(node)
            node_type = self.get_node_type(node)

            # Sanity check
            if len(path) > 0 and node not in allowed_tokens[-1]:
                parent_edge_label = self.get_node_label(path[-1])
                grandparent_node_label = self.get_node_label(path[-2])
                log.info("Warning: Target node %s is not in list of allowed nodes of %s.%s in sample %s!",
                         node_label, grandparent_node_label, parent_edge_label, sample_id)

            if len(path) == 0:
                seg = [1]
            else:
                # Current Edge Node
                cen = path[-1]
                cen_seg = edge_order_path[-1]
                cen_type = self.get_node_type(cen)

                if cen_type == "list" and node_label != "None#literal":
                    last_list_el = next((seg for seg, node_type in reversed(edge_order_seq) if seg[1:] == cen_seg),
                                        None)
                    edge_counter = 1 if last_list_el is None else last_list_el[0] + 1
                else:
                    edge_counter = 1

                # AST Edge Nodes
                seg = [edge_counter] + cen_seg
                edge_counts[cen_seg.__str__()] = (cen, cen_type, edge_counter)

            if max_depth < len(seg):
                max_depth = len(seg)

            edge_order_seq.append((seg, node_type))
            edge_order_path.append(seg)

            # print((len(path) * ' ') + 'Path:', [node_label for node in path])
            # print((len(path) * ' ') + 'Edge Path *:', edge_order_path)
            # print((len(path) * ' ') + 'node:', self.get_node_label(node), "Path seg:", seg)
            path.append(node)

            # AST Nodes
            successor_edge_nodes = sorted(self.graph.successors(node), key=self.get_node_order)
            edge_counter = len(successor_edge_nodes)
            edge_counts[seg.__str__()] = (node_label, node_type, edge_counter)

            if len(successor_edge_nodes) > 0:
                next_edge = successor_edge_nodes[0]
                next_edge_type = self.get_node_type(next_edge)

                seg = [1] + edge_order_path[-1]
                edge_order_seq.append((seg, next_edge_type))
                edge_order_path.append(seg)

                if max_depth < len(seg):
                    max_depth = len(seg)
                # print((len(path) * ' ') + 'first edge (singleton, first list element or empty list end) (A):',self.get_node_label(next_edge), "Path seg:", seg)
                path.append(next_edge)
                current_edge = None
            else:
                current_node = path.pop()
                seg = edge_order_path.pop()
                # print((len(path) * ' ') + 'finished node (A):', self.get_node_label(current_node), "Path seg:", seg)

                while len(path) > 0:
                    current_edge = path[-1]
                    current_edge_type = self.get_node_type(current_edge)

                    if current_edge_type == "list" and current_node != self.list_end_id and self.get_node_label(
                            current_node) != "None#literal":
                        next_edge = current_edge
                        next_edge_type = edge_order_path[-1]
                        # print((len(path) * ' ') + 'same edge (additional list element or list end):', self.get_node_label(current_edge), "Path seg:", seg)
                        break
                    else:
                        current_edge = path.pop()
                        seg = edge_order_path.pop()
                        # print((len(path) * ' ') + 'finished edge (singleton or list):', self.get_node_label(current_edge), "Path seg:", seg)
                        current_edge_order = self.get_node_order(current_edge)
                        parent_node = path[-1]

                        parent_node_edges = sorted(self.graph.successors(parent_node), key=self.get_node_order)
                        if len(parent_node_edges) > current_edge_order + 1:
                            next_edge_order = current_edge_order + 1
                            next_edge = parent_node_edges[next_edge_order]
                            next_edge_type = self.get_node_type(next_edge)
                            seg = [next_edge_order + 1] + edge_order_path[-1]
                            edge_order_seq.append((seg, next_edge_type))
                            edge_order_path.append(seg)

                            if max_depth < len(seg):
                                max_depth = len(seg)

                            # print((len(path) * ' ') + 'additional edge (singleton, first list element or empty list end) (B):',self.get_node_label(next_edge), "Path seg:", seg)
                            path.append(next_edge)
                            break
                        else:
                            current_node = path.pop()
                            seg = edge_order_path.pop()
                            # print((len(path) * ' ') + 'finished node (B):', self.get_node_label(current_node),"Path seg:", seg)

            if len(path) == 0:
                next_edge = None
            next_edge_allowed_tokens = self.compute_allowed_tokens(next_edge, node)
            allowed_tokens.append(next_edge_allowed_tokens)

        # print((len(path) * ' ') + 'Path: ', [self.get_node_label(node) for node in path])

        max_depth = max_depth - 1
        edge_counts = list(edge_counts.values())

        # Stats
        # print((len(path) * ' ') + 'Max depth: ', max_depth)
        # print((len(path) * ' ') + 'Edge counts: ', edge_counts)

        node_paths_seq = self.compute_node_paths_seq(edge_order_seq)
        # print('node_paths_seq ',' len: ',len(node_paths_seq),node_paths_seq)
        return allowed_tokens, node_paths_seq, edge_counts, max_depth

    def compute_node_paths_seq(self, edge_order_seq):
        return [path for path, node_type in edge_order_seq if node_type in ('node', 'strliteral', 'literal', 'special')]

    def visit_graph_edge_order_path_beam_search(self, node, path, edge_order_seq, edge_order_path):
        node_label = self.get_node_label(node)
        node_type = self.get_node_type(node)

        if len(path) == 0:
            seg = [1]
        else:
            cen = path[-1]
            cen_seg = edge_order_path[-1]
            cen_type = self.get_node_type(cen)

            if cen_type == "list" and node_label != "None#literal":
                last_list_el = next((seg for seg, node_type in reversed(edge_order_seq) if seg[1:] == cen_seg), None)
                edge_counter = 1 if last_list_el is None else last_list_el[0] + 1

            else:
                edge_counter = 1

            # AST Edge Nodes
            seg = [edge_counter] + cen_seg

        edge_order_seq.append((seg, node_type))
        edge_order_path.append(seg)

        path.append(node)

        # AST Nodes
        successor_edge_nodes = sorted(self.graph.successors(node), key=self.get_node_order)

        if len(successor_edge_nodes) > 0:
            next_edge = successor_edge_nodes[0]
            next_edge_type = self.get_node_type(next_edge)

            seg = [1] + edge_order_path[-1]
            edge_order_seq.append((seg, next_edge_type))
            edge_order_path.append(seg)

            path.append(next_edge)
            current_edge = None
        else:
            current_node = path.pop()
            seg = edge_order_path.pop()

            while len(path) > 0:
                current_edge = path[-1]
                current_edge_type = self.get_node_type(current_edge)

                if current_edge_type == "list" and current_node != self.list_end_id and self.get_node_label(
                        current_node) != "None#literal":
                    next_edge = current_edge
                    next_edge_type = edge_order_path[-1]
                    break
                else:
                    current_edge = path.pop()
                    seg = edge_order_path.pop()
                    current_edge_order = self.get_node_order(current_edge)
                    parent_node = path[-1]

                    parent_node_edges = sorted(self.graph.successors(parent_node), key=self.get_node_order)
                    if len(parent_node_edges) > current_edge_order + 1:
                        next_edge_order = current_edge_order + 1
                        next_edge = parent_node_edges[next_edge_order]
                        next_edge_type = self.get_node_type(next_edge)
                        seg = [next_edge_order + 1] + edge_order_path[-1]
                        edge_order_seq.append((seg, next_edge_type))
                        edge_order_path.append(seg)

                        path.append(next_edge)
                        break
                    else:
                        current_node = path.pop()
                        seg = edge_order_path.pop()
            if len(path) == 0:
                next_edge = None

        token_child_nodes = self.compute_allowed_tokens(next_edge, node)
        return token_child_nodes, path, edge_order_seq, edge_order_path
