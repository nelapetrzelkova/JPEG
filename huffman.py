from queue import PriorityQueue
from collections import Counter


class HuffmanTree:
    class Node:
        def __init__(self, val, freq, left_child, right_child):
            self.value = val
            self.freq = freq
            self.left_child = left_child
            self.right_child = right_child

        def __eq__(self, other):
            stup = self.value, self.freq, self.left_child, self.right_child
            otup = other.value, other.freq, other.left_child, other.right_child
            return stup == otup

        def __nq__(self, other):
            return not (self == other)

        def __lt__(self, other):
            return self.freq < other.freq

        def __le__(self, other):
            return self.freq < other.freq or self.freq == other.freq

        def __gt__(self, other):
            return not (self <= other)

        def __ge__(self, other):
            return not (self < other)

    def __init__(self, arr):
        self.tree = self.create_tree(arr)
        self.root = self.tree.get()
        self.value_to_bitstring = dict()

    def init_queue(self, arr):
        q = PriorityQueue()
        c = Counter(arr)
        for val, freq in c.items():  # define leaves nodes
            q.put(self.Node(val, freq, None, None))
        return q

    def create_tree(self, arr):
        q = self.init_queue(arr)
        while q.qsize() > 1:
            u = q.get()
            v = q.get()
            freq = u.freq + v.freq
            q.put(self.Node(None, freq, u, v))
        return q

    def value_to_bitstring_table(self):
        if len(self.value_to_bitstring.keys()) == 0:
            self.create_huffman_table()
        return self.value_to_bitstring

    def create_huffman_table(self):
        def tree_traverse(current_node, bitstring=''):
            if current_node is None:
                return
            if current_node.value is not None:  # is a leaf
                self.value_to_bitstring[current_node.value] = bitstring
                return
            tree_traverse(current_node.left_child, bitstring + '0')
            tree_traverse(current_node.right_child, bitstring + '1')

        tree_traverse(self.root)

