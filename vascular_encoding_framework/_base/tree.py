from __future__ import annotations

from copy import deepcopy
from typing import Generic, Type, TypeVar

from ..messages import warning_message
from .node import Node
from .value_typed_dict import ValueTypedDict

_TT = TypeVar("_TT")


class Tree(ValueTypedDict[_TT], Generic[_TT]):
    """
    Abstract class for trees.

    It inherits from dictionary structure so its easier to get-set items.
    """

    def __init__(self, _node_type: Type[_TT] = Node) -> None:
        """Tree constructor."""
        super().__init__()
        # This way we allow more than one tree to be hold. Actually more like a
        # forest...

        if not issubclass(_node_type, Node):
            raise ValueError(
                f"Expecting a subclass of Node. Provided {_node_type.__name__} is not."
            )

        self._node_type: Type[_TT] = _node_type

        return

    def __str__(self):
        outstr = ""
        ind = " "

        def append_str(nid, outstr, l=0):
            strout = "\n".join([ind * 4 * l + s for s in self[nid].__str__().split("\n")]) + "\n\n"
            for cid in self[nid].children:
                strout += append_str(cid, outstr, l=l + 1)

            return strout

        for rid in self.roots:
            outstr += append_str(rid, outstr=outstr)

        return outstr

    def pop(self, key: str) -> _TT:
        nd = self[key]
        self.remove(key)
        return nd

    def __or__(self, *args, **kwargs) -> None:
        raise AttributeError(f"{self.__class__.__name__} object has no attribute __or__.")

    def __ior__(self, *args, **kwargs) -> None:
        raise AttributeError(f"{self.__class__.__name__} object has no attribute __ior__.")

    @property
    def roots(self) -> set[str]:
        """Uptade the attribute roots to contain only nodes with None as parent."""
        return {k for k, v in self.items() if v.parent is None}

    def enumerate(self):
        """Get a list with the id of stored items."""
        return list(self.keys())

    def __setitem__(self, __key, nd: Node) -> None:
        if not isinstance(nd, self._node_type):
            raise TypeError(
                f"Aborted insertion element: {__key}. Object of type {type(nd).__name__} given, "
                + f"only {self._node_type.__name__} elements are accepted."
            )

        if not isinstance(__key, str):
            raise TypeError(f"Provided {__key} must be a string.")

        if __key != nd.id:
            raise ValueError(f"Provided key ({__key}) does not match the id attribute ({nd.id}).")

        super().__setitem__(__key, nd)

        if nd.parent is not None:
            self[nd.parent].add_child(__key)

    def graft(self, tr, gr_id=None):
        """
        Merge another tree. If gr_id is a node id of this tree,
        root nodes are grafted on self[gr_id], otherwise they are
        grafted as roots.

        Parameters
        ----------
        tr : Tree
            The tree to merge into self.
        gr_id : node id
            The id of a node in this tree where tr will be grafted.
        """

        def add_child(nid):
            self[nid] = tr[nid]
            for cid in tr[nid].children:
                add_child(nid=cid)

        for rid in tr.roots:
            add_child(rid)
            if gr_id in self:
                self[rid].parent = gr_id
                self[gr_id].add_child(rid)

    def remove(self, k):
        """
        Remove node by key. Note this operation does not remove
        its children. See prune method to remove a subtree. Using
        this method will make children belong to roots set.

        Returns
        -------
            The removed node, as pop method does in dictionaries.
        """

        # Children are now roots
        for child in self[k].children:
            self[child].parent = None

        # If has a parent remove it from parent children set.
        if self[k].parent is not None:
            self[self[k].parent].remove_child(k)

        return super().pop(key=k)

    def prune(self, k):
        """
        Remove all the subtree rooted at node k, included.

        Parameters
        ----------
        k : any id
            id of the node from which to prune.
        """

        def rm_child(nid):
            for cid in self[nid].children:
                rm_child(nid=cid)
            super().pop(__key=nid)

        pid = self[k].parent
        if pid is not None:
            self[pid].remove_child(k)

        rm_child(k)

    def copy(self):
        new_tree = self.__class__()

        def copy_and_insert(nid):
            new_node = deepcopy(self[nid])
            new_tree[nid] = new_node
            for cid in new_node.children:
                copy_and_insert(cid)

        for rid in self.roots:
            copy_and_insert(rid)
        return new_tree

    def set_data_to_nodes(self, data: dict[str, dict]):
        """
        Broadcast set_data method on the nodes of the tree using its id.

        The data argument is expected to be a dictionary of dictionaries containing the data
        for each Node, i.e.
        data = {
                 'id1' : {'center' : [x,y,z], 'normal' :[x1, y1, z1] }
                 'id2' : {'normal' :[x2, y2, z2] }
                 'id3' : {'center' : [x3,y3,z3]}
        }.

        """

        for nid, ndata in data.items():
            self[nid].set_data(**ndata)

        self.is_consistent()

    def is_consistent(self):
        """
        Check if the parent - children attributes of the nodes are consistent among them.
        If not, report unconsistencies.

        Returns
        -------
        out : bool
            True if parent-child attributes are not in contradiction among nodes, False otherwise.
        """
        out = True
        for nid, node in self.items():
            if node.parent is not None:
                if nid not in self[node.parent].children:
                    warning_message(
                        f"Inconsistency found: {nid} has {node.parent} as parent, but it is not in "
                        + "its children set."
                    )
                    out = False
        return out

    def has_non_roots(self):
        """
        Check if has non-root elements. This implies that there is an actual hierarchy, otherwise this is pretty much a dict.

        Empty Tree objects will return False.


        Returns
        -------
        out : bool
            True if has non-root nodes, False otherwise.

        See Also
        --------
        is_consistent
        """

        out = False
        nonroots = set(self.enumerate()) - self.roots
        if nonroots:
            out = True
        return out

    def change_node_id(self, old_id, new_id):
        """
        Change the id of a Node of the Tree and update all its relatives.

        Parameters
        ----------
            old_id, new_id : str
                The current id and the desired new one.
        """

        if new_id in self:
            raise ValueError(f"{new_id} is already present. Cant rename {old_id} to {new_id}.")

        self[old_id].id = new_id
        self[new_id] = self.pop(old_id)

        if self[new_id].parent in self:
            self[self[new_id].parent].remove_child(old_id)
            self[self[new_id].parent].add_child(new_id)

        for cid in self[new_id].children:
            self[cid].parent = new_id
