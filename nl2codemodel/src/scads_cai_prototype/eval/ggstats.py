import argparse
import json
import logging as log

from scads_cai_prototype.grammar.grammargraphloader import GrammarGraphLoader


def ggstats(gg_train_path, gg_test_path, gg_stats_file):
    gg_train = load_ggraph(gg_train_path)
    n_subg_train, lit_subg_train, slit_subg_train = edges_by_nodetype(gg_train)

    n_subg_train_attr_nodes = {attr for attr, node in n_subg_train}

    gg_test = load_ggraph(gg_test_path)
    n_subg_test, lit_subg_test, slit_subg_test = edges_by_nodetype(gg_test)

    n_subg_test_attr_nodes = {attr for attr, node in n_subg_test}

    n_subres_test = subtract(n_subg_test, n_subg_train)
    lit_subres_test = subtract(lit_subg_test, lit_subg_train)
    slit_subres_test = subtract(slit_subg_test, slit_subg_train)

    gg_stats = {
        "no_attr_obj_edge-type_node-train": len(n_subg_train),
        "no_attr_obj_edge-type_lit-train": len(lit_subg_train),
        "no_attr_obj_edge-type_strlit-train": len(slit_subg_train),
        "no_attr_nodes_train_childtype_node-train": len(n_subg_train_attr_nodes),

        "no_attr_obj_edge-type_node-test": len(n_subg_test),
        "no_attr_obj_edge-type_lit-test": len(lit_subg_test),
        "no_attr_obj_edge-type_strlit-test": len(slit_subg_test),
        "no_attr_nodes_train_childtype_node-test": len(n_subg_test_attr_nodes),

        "avg_attr_obj_node_connections_type_node-train": len(n_subg_train) / len(n_subg_train_attr_nodes),
        "avg_attr_obj_node_connections_type_node-test": len(n_subg_test) / len(n_subg_test_attr_nodes),

        "substr_no_attr_obj_edge-type_node-train": len(n_subres_test),
        "substr_no_attr_obj_edge-type_lit-train": len(lit_subres_test),
        "substr_no_attr_obj_edge-type_strlit-train": len(slit_subres_test),

        "substr_attr_obj_edge-type_node-train": n_subres_test,
        "substr_attr_obj_edge-type_lit-train": lit_subres_test,
        "substr_attr_obj_edge-type_strlit-train": slit_subres_test,
    }

    # print(gg_stats)

    with open(gg_stats_file, "w") as json_file:
        json.dump(gg_stats, json_file)


def subtract(list_a, list_b):
    intersection = list(set(list_a) & set(list_b))

    return [item for item in list_a if item not in intersection]


def edges_by_nodetype(ggraph):
    n_subg = []
    lit_subg = []
    slit_subg = []

    for u, v, a in ggraph.edges(data=True):
        v_type = ggraph.nodes[v]['type']
        v_label = ggraph.nodes[v]['label']

        if v_type == 'node':
            n_subg.append((u, v_label))
            # print('ATTR.-OBJ. EDGE:', u,' -> ', v_label)

        if v_type == 'literal':
            lit_subg.append((u, v_label[:-len('#literal')]))
            # print('ATTR.-LIT.OBJ. EDGE:', u, ' -> ', v_label_)

        if v_type == 'strliteral':
            slit_subg.append((u, v_label[:-len('#strliteral')]))
            # print('ATTR.-STR.LIT.OBJ. EDGE:', u, ' -> ', v_label_)

    # print('Edge data - Type Node: ', n_subg)
    # print('Edge data - Type Literal: ', lit_subg)
    # print('Edge data - Type StrLiteral: ', slit_subg)

    return n_subg, lit_subg, slit_subg


def load_ggraph(gg_path):
    gg = GrammarGraphLoader(gg_path).load_graph()
    # print('Loading Grammar Graph for Test Data...', gg_path)
    # print('Node data: ', gg.nodes.data())
    # print('Edge data: ', gg.edges.data())

    return gg


def get_args():
    parser = argparse.ArgumentParser("Compute Grammar Graph stats", fromfile_prefix_chars='@')

    parser.add_argument("--gg-train-data", type=str)
    parser.add_argument("--gg-test-data", type=str)
    parser.add_argument("--gg-stats-file", type=str)

    return parser.parse_args()


def main():
    args = get_args()
    # print("Grammar Graph stats args:", vars(args))
    log.basicConfig(level='DEBUG')

    ggstats(args.gg_train_data, args.gg_test_data, args.gg_stats_file)


if __name__ == '__main__':
    main()
