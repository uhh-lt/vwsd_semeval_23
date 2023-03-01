import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_visualsem_bnids(visualsem_nodes_path, visualsem_images_path=None):
    x = json.load(open(visualsem_nodes_path, 'r'))
    ims = []
    bn_to_ims = defaultdict(list)

    def get_full_img_name(im):
        """ Closure to give full path to image given image name. """
        if not visualsem_images_path is None:
            fname = os.path.join(visualsem_images_path, im[:2], im + ".jpg")
        else:
            fname = os.path.join(im[:2], im + ".jpg")
        return fname

    for bid, v in x.items():
        for im in v['ims']:
            bn_to_ims[bid].append(get_full_img_name(im))

    # sort entries by BabelNet ID
    full_bnids_to_ims = {bid: ims for bid, ims in sorted(bn_to_ims.items(), key=lambda kv: kv[0])}
    print(
        "Total number of BabelNet IDs in VisualSem: %i.\nTotal number of image-node associations: %i.\nMaximum number of images linked to a node: %i." % (
            len(full_bnids_to_ims),
            sum([len(v) for (k, v) in full_bnids_to_ims.items()]),
            max([len(v) for (k, v) in full_bnids_to_ims.items()])
        ))
    # print("First 5 BabelNet IDs: ", [bnid for (bnid,ims) in full_bnids_to_ims[:5]], "...")

    return full_bnids_to_ims


class VisualSemNodesDataset(torch.utils.data.Dataset):
    """
        Dataset class that can be used to iterate all nodes in VisualSem (linking all data available in a node).
        Nodes are associated to images, multilingual glosses, and tuples
        (i.e. its tuples include all nodes with which it is a tail in VisualSem).
    """

    def __init__(self, path_to_nodes, path_to_glosses, path_to_tuples, path_to_images=None):
        """
            path_to_nodes(str):         Path to JSON file containing VisualSem nodes.
            path_to_glosses(str):       Path to JSON file containing VisualSem glosses.
            path_to_tuples(str):        Path to JSON file containing VisualSem tuples.
            path_to_images(str):        Path to directory containing VisualSem images. (Optional)
        """
        super(VisualSemNodesDataset).__init__()

        assert Path(path_to_nodes).exists(), f"File not found: {path_to_nodes}"
        assert Path(path_to_tuples).exists(), f"File not found: {path_to_tuples}"
        assert Path(path_to_glosses).exists(), f"File not found: {path_to_glosses}"
        assert path_to_images is None or Path(path_to_images).exists(), f"File not found: {path_to_images}"

        self.root_path = Path(path_to_nodes).parent

        full_bnids_to_ims = load_visualsem_bnids(path_to_nodes, path_to_images)
        self.full_bnids_to_ims = full_bnids_to_ims
        self.bnids = list(full_bnids_to_ims.keys())

        self.nodes = {}
        with open(path_to_nodes, 'r') as fh:
            nodes_json = json.load(fh)
            for node_key, node_value in nodes_json.items():
                # initialize node
                self.nodes[node_key] = {
                    "ms": node_value['ms'],
                    "se": node_value['se'],
                    "images": node_value["ims"],
                }

        with open(path_to_tuples, 'r') as fh:
            tuples_json = json.load(fh)
            for tuple_key, tuple_value in tuples_json.items():
                if not "incoming_nodes" in self.nodes[tuple_key]:
                    self.nodes[tuple_key]["incoming_nodes"] = []

                for entry in tuple_value:
                    self.nodes[tuple_key]["incoming_nodes"].append({
                        "head": entry['s'],
                        "tail": tuple_key,
                        "relation": entry['r'],
                        "tuple_id": entry['r_id']
                    })

        with open(path_to_glosses, 'r') as fh:
            glosses_json = json.load(fh)
            for gloss_entry in glosses_json[1:]:
                assert (len(gloss_entry) == 1)
                key = list(gloss_entry.keys())[0]
                self.nodes[key]["glosses"] = gloss_entry[key]

    def __getitem__(self, index):
        node = dict(self.nodes[self.bnids[index]])
        node.update({'bnid': self.bnids[index]})
        return node

    def get_node_by_bnid(self, bnid):
        node = dict(self.nodes[bnid])
        node.update({'bnid': bnid})
        return node

    def get_node_images_by_bnid(self, bnid):
        return self.full_bnids_to_ims[bnid]

    def __len__(self):
        return len(self.full_bnids_to_ims)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    nodes_json = os.path.join(dir_path, "dataset", "nodes.v2.json")
    glosses_json = os.path.join(dir_path, "dataset", "gloss_files", "nodes.glosses.json")
    tuples_json = os.path.join(dir_path, "dataset", "tuples.v2.json")
    # testing node dataset
    print("Testing node dataset...")
    vs = VisualSemNodesDataset(nodes_json, glosses_json, tuples_json)
    print("len(vs): ", len(vs))
    print(vs[0])
