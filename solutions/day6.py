from typing import List, Dict, Union
import csv


def build_tree(list_of_links: List[str]) -> Dict[str, str]:
    tree = {}
    for link in list_of_links:
        parent, child = link.split(')')
        tree[child] = parent
    return tree


def get_parent(tree, child: str) -> Union[str, None]:
    return tree.get(child, None)


def get_num_direct_links(tree: Dict[str, str]) -> int:
    return len(list(tree.keys()))


def get_num_ancestors(tree: Dict[str, str], node: str) -> int:
    parent = get_parent(tree, node)
    if parent is None:
        return 0
    else:
        return 1 + get_num_ancestors(tree, parent)


def get_ancestors(tree: Dict[str, str], node: str) -> List[str]:
    parent = get_parent(tree, node)
    if parent is None:
        return []
    else:
        return [parent] + get_ancestors(tree, parent)


def get_num_indirect_links(tree: Dict[str, str]) -> int:
    total = 0
    for _, v in tree.items():
        total += get_num_ancestors(tree, v)   # direct parent link does not count
    return total


def readfile(inputfile: str) -> List[str]:
    result = []
    with open(inputfile) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            result.append(line[-1])
    return result


def get_ancestor_distance(tree: Dict[str, str], node1: str, node2: str) -> int:
    path = get_ancestors(tree, node1)
    if node2 in path:
        return path.index(node2) + 1

    raise RuntimeError('Not an ancestor')


def get_first_common_ancestor(path1: List[str], path2: List[str]) -> Union[str, None]:
    for node in path1:
        if node in path2:
            return node
    return None


def get_distance(tree: Dict[str, str], node1: str, node2: str) -> int:
    path1 = get_ancestors(tree, node1)
    path2 = get_ancestors(tree, node2)

    # find first common ancestor
    common = get_first_common_ancestor(path1 , path2)
    assert common is not None, 'No common ancestors'

    dist = get_ancestor_distance(tree, node1, common) + \
           get_ancestor_distance(tree, node2, common) \
           - 2  # don't count links to parents
    return dist


if __name__ == '__main__':
    def _main():
        links = readfile('../inputs/day6.txt')
        tree = build_tree(links)
        print(get_num_direct_links(tree) + get_num_indirect_links(tree))
        print(get_distance(tree, 'YOU', 'SAN'))

    _main()
