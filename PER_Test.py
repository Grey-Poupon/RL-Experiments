import unittest
from PER_Memory import PEReplayMemory, SumTree, Item
import random
import numpy as np
class PER_TEST(unittest.TestCase):
    def setUp(self):
        pass

    def test_TD_order(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)]
        sumTree = SumTree(10, items)
        curr = None
        for leaf in sumTree.get_leaves():
            if curr is None:
                curr = abs(leaf.TD_error)

            self.assertFalse(abs(leaf.TD_error) < curr)
            curr = abs(leaf.TD_error)


    def test_priority_order(self):
        sumTree = SumTree(10, [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),
                               Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)])
        curr = None

        for leaf in sumTree.get_leaves():
            if curr is None:
                curr = leaf.get_priority()
                continue

            self.assertTrue(leaf.get_priority() > curr)
            curr = leaf.get_priority()

    def test_sum_priority(self):
        sumTree = SumTree(10, [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),
                               Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)])
        p = 0

        for leaf in sumTree.get_leaves():
                p += leaf.get_priority()

        self.assertTrue(p == sumTree.get_sum_priority())

    def test_num_leaves(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),
                               Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)]
        sumTree = SumTree(10, items)

        self.assertTrue(sumTree.get_num_leaves() == len(items))

    def test_change_priority(self):
        sumTree = SumTree(10, [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),
                               Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)])
        p = 0

        for leaf in sumTree.get_leaves():
                p += leaf.get_priority()

        for leaf in sumTree.get_leaves():
            p1 = leaf.get_priority()
            sumTree.update_leaf_priority(leaf.idx,p1+1)

        self.assertTrue(p+sumTree.get_num_leaves() == sumTree.get_sum_priority())

    def test_leaf_internal_idx(self):
        sumTree = SumTree(10, [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),
                               Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)])

        for leaf in sumTree.get_leaves():
            leaf.update_TD_error(random.randrange(10))
        sumTree.sort_tree()

        l = sumTree.get_left_most_node(sumTree.layers)
        r = sumTree.get_right_most_node(sumTree.layers)
        for idx in range(l,r+1):
            self.assertTrue(sumTree.tree[idx].idx == idx)

    def test_remove_add_items(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),  Item("A", 5)]
        sumTree = SumTree(7, items)

        total = sumTree.get_sum_priority()

        x,y,z = Item("X", 99), Item("Y", 98), Item("Z", 97)
        sumTree.add_item(x)
        sumTree.add_item(y)
        sumTree.add_item(z)

        self.assertTrue(sumTree.get_leaves()[-3:] == [x,y,z])


    def test_changing_priority(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),  Item("A", 5)]
        sumTree = SumTree(7, items)

        total = sumTree.get_sum_priority()

        x,y,z = Item("X", 99), Item("Y", 98), Item("Z", 97)
        sumTree.add_item(x)
        sumTree.add_item(y)
        sumTree.add_item(z)
        curr = None
        for leaf in sumTree.get_leaves():
            if curr is None:
                curr = leaf.get_priority()
                continue
            print(leaf.get_priority())
            self.assertTrue(leaf.get_priority() > curr)
            curr = leaf.get_priority()

    def test_adding_priority(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),  Item("A", 5)]
        sumTree = SumTree(7, [])

        for x in items:
          sumTree.add_item(x)

        curr = None
        for leaf in sumTree.get_leaves():
            if curr is None:
                curr = leaf.get_priority()
                print(curr)
                continue
            print(leaf.get_priority())
            self.assertTrue(leaf.get_priority() > curr)
            curr = leaf.get_priority()


if __name__ == '__main__':
    unittest.main()
