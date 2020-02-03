import unittest
from collections import deque

from PER_Memory import PEReplayMemory, SumTree, Item
import random
import time
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
                continue

            self.assertGreaterEqual(curr, abs(leaf.TD_error))
            curr = abs(leaf.TD_error)

    def test_priority_order(self):
        sumTree = SumTree(10, [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),
                               Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)])
        curr = None

        for leaf in sumTree.get_leaves():
            if curr is None:
                curr = leaf.get_priority()
                continue

            self.assertGreaterEqual(curr, leaf.get_priority())
            curr = leaf.get_priority()

    def test_sum_priority(self):
        sumTree = SumTree(10, [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),
                               Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)])
        p = 0

        for leaf in sumTree.get_leaves():
                p += leaf.get_priority()

        self.assertAlmostEqual(p, sumTree.get_sum_priority(), places=8)

    def test_num_leaves(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9),
                 Item("A", 8), Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)]
        sumTree = SumTree(10, items)

        self.assertEqual(sumTree.get_num_leaves(), len(items))

    def test_get_leaves(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9),
                 Item("A", 8), Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)]
        sumTree = SumTree(10, items)

        self.assertEqual(sumTree.get_leaves(), deque(items, maxlen=10))

    def test_leaf_internal_idx(self):
        sumTree = SumTree(10, [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),
                               Item("A", 5), Item("A", 8), Item("A", 7), Item("A", -1)])

        for idx in range(sumTree.get_num_leaves()):
            self.assertEqual(sumTree.tree[idx].idx, idx - sumTree.idx_shift)

    def test_remove_add_items(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8),  Item("A", 5)]
        sumTree = SumTree(7, items)

        total = sumTree.get_sum_priority()

        x,y,z = Item("X", 99), Item("Y", 98), Item("Z", 97)
        sumTree.add_item(x)
        sumTree.add_item(y)
        sumTree.add_item(z)

        self.assertListEqual(list(sumTree.get_leaves())[-3:], [x,y,z])

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

            self.assertGreaterEqual(curr, leaf.get_priority())
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
                continue

            self.assertGreaterEqual(curr, leaf.get_priority())
            curr = leaf.get_priority()

    def test_adding_speed(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5)]
        sumTree = SumTree(1000000, items)

        print("Adding Speed")
        for x in items:
            start = time.perf_counter()
            sumTree.add_item(x)
            end = time.perf_counter()
            elapsed_time = end - start

            print("Time:", elapsed_time)
            self.assertLessEqual(elapsed_time, 0.3)
        print("\n")

    def test_minibatch_speed(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5)]
        sumTree = SumTree(1000000, items)
        # start = time.perf_counter()
        # sumTree.get_minibatch(32)
        # end = time.perf_counter()

        start2 = time.perf_counter()
        sumTree.get_minibatch2(32)
        end2 = time.perf_counter()

 ##       print("MinB  ", end-start)
        print("Minibatch Speed")
        print("Time: ", end2-start2)
        print("\n")
        self.assertLessEqual(end2-start2, 0.003)

    def test_tree_sort(self):
        items = [Item("A", 6), Item("A", 3), Item("A", 999), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 67), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 68), Item("A", 5),
                 Item("A", 16), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 89), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 33), Item("A", 1), Item("A", 70), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", -89), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 68), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 15),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 5),
                 Item("A", 6), Item("A", 3), Item("A", 1), Item("A", 0), Item("A", 9), Item("A", 8), Item("A", 15)]
        sumTree = SumTree(1000000, items)

        sumTree.sort_tree()

        curr = None
        for leaf in sumTree.get_leaves():
            if curr is None:
                curr = abs(leaf.TD_error)
                continue

            self.assertGreaterEqual(curr, abs(leaf.TD_error))
            curr = abs(leaf.TD_error)

if __name__ == '__main__':
    unittest.main()
