from networkx.readwrite import gpickle


class GrammarGraphLoader():
    def __init__(self, graph_path):
        self.graph_path = graph_path

    def load_graph(self):
        graph = gpickle.read_gpickle(self.graph_path)

        #print('Loading Grammar Graph ', self.graph_path)
        #print('Node data: ', graph.nodes.data())
        #print('Edge data: ', graph.edges.data())

        return graph
