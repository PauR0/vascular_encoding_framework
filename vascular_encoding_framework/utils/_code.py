

from copy import deepcopy

import numpy as np

from ..messages import *


class Node:
    """
    Abstract class for tree node.
    """

    def __init__(self, nd=None) -> None:

        self.id = None
        self.parent = None
        self.children: set = set()

        if nd is not None:
            self.set_data_from_other_node(nd=nd)
    #

    def __str__(self):
        long_atts = ['points', 'faces']
        strout = '\n'.join([f'{k}'.ljust(
            20, '.') + f': {v}' for k, v in self.__dict__.items() if k not in long_atts])
        for att in long_atts:
            if att in self.__dict__:
                val = getattr(self, att)
                if val is not None:
                    n = len(val)
                    if att == 'faces':
                        n /= 4
                    strout += f'\nn_{att}'.ljust(20, '.') + f': {n}'

        return strout
    #

    def set_data(self, to_numpy=True, **kwargs):
        """
        Method to set attributes by means of kwargs.
        If to_numpy lists containig floats or lists of floats
        will be tried to turn into numpy arrays.

        E.g.
            a = Node()
            a.set_data(center=np.zeros((3,)))

        Arguments:
        ------------

            to_numpy : bool, opt
                Default True. Whether to try to cast numerical sequences to
                numpy arrays.
        """

        if 'children' in kwargs:
            self.children = set(kwargs['children'])
            kwargs.pop('children')

        if to_numpy:
            kwargs_np = deepcopy(kwargs)
            for k, v in kwargs.items():
                if v is not None:
                    if is_arrayable(v):
                        kwargs_np[k] = k = np.array(v)
            kwargs = kwargs_np

        attribute_setter(self, **kwargs)
    #

    def set_data_from_other_node(self, nd):
        """
        Copy the Node attribute from other Node object into this.
        Note that only the default Node attributes defined in the
        class constructor will be copied.

        Arguments:
        ----------

            nd : Node
                The node from which attributes will be copied.
        """

        self.set_data(**{k: getattr(nd, k) for k in Node().__dict__})
    #

    def add_child(self, c):
        """
        Add a child to this branch.
        """
        self.children.add(c)
    #

    def remove_child(self, c):
        """
        Remove child. If does not exists, nothing happens.
        """
        self.children.discard(c)
    #
#


class Tree(dict):
    """
    Abstract class for trees. It inherits from dictionary structure so its
    easier to get-set items.
    """

    def __init__(self) -> None:
        """
        Tree constructor.
        """
        super().__init__()
        # This way we allow more than one tree to be hold. Actually more like a
        # forest...
        self.roots: set = set()
        return
    #

    def __str__(self):
        outstr = ''
        ind = ' '

        def append_str(nid, outstr, l=0):
            strout = '\n'.join(
                [ind * 4 * l + s for s in self[nid].__str__().split('\n')]) + '\n\n'
            for cid in self[nid].children:
                strout += append_str(cid, outstr, l=l + 1)

            return strout

        for rid in self.roots:
            outstr += append_str(rid, outstr=outstr)

        return outstr
    #

    def enumerate(self):
        """
        Get a list with the id of stored items.
        """
        return list(self.keys())
    #

    def __setitem__(self, __key, nd: Node) -> None:

        # Checking it has parent and children attributes. Since Nones are
        # admited, attribute_checker is not well suited.
        for att in ['parent', 'children']:
            if not hasattr(nd, att):
                error_message(
                    f'Aborted insertion of node with id: {__key}. It has no {att} attribute.')
                return

        if nd.parent is not None and nd.parent not in self.keys():
            error_message(
                f'Aborted insertion of node with id: {__key}. Its parent {nd.parent} does not belong to the tree.')
            return

        if not isinstance(__key, str):
            warning_message(
                f'node {__key} has been set with a non-string key. This may turn in troubles...')
        if __key != nd.id:
            warning_message(
                f'node id attribute is {nd.id} and node id in tree has been set as {__key}.')

        super().__setitem__(__key, nd)

        if nd.parent is None:
            self.roots.add(__key)
        else:
            self[nd.parent].add_child(__key)
    #

    def graft(self, tr, gr_id=None):
        """
        Merge another tree. If gr_id is a node id of this tree,
        root nodes are grafted on self[gr_id], otherwise they are
        grafted as roots.

        Arguments:
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
    #

    def remove(self, k):
        """
        Remove node by key. Note this operation does not remove
        its children. See prune method to remove a subtree. Using
        this method will make children belong to roots set.

        Returns:
        ----------
            The removed node, as pop method does in dictionaries.
        """

        # Children are now roots
        for child in self[k].children:
            self[child].parent = None
            self.roots.add(child)

        # If k is a root remove from roots set, otherwise remove it from parent
        # children set.
        if self[k].parent is None:
            self.roots.discard(k)
        else:
            self[self[k].parent].remove_child(k)

        return super().pop(__key=k)
    #

    def prune(self, k):
        """
        Remove all the subtree rooted at node k, included.

        Arguments:
        ------------

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
    #

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
    #

    def set_data_to_nodes(self, data):
        """
        This method allows to use the set_data method on the nodes of the tree using its id.
        The data argument is expected to be a dictionary of dictionaries containing the data
        for each Node, i.e.
        data = {
                 'id1' : {'center' : [x,y,z], 'normal' :[x1, y1, z1] }
                 'id2' : {'normal' :[x2, y2, z2] }
                 'id3' : {'center' : [x3,y3,z3]}
        }

        """

        for nid, ndata in data.items():
            self[nid].set_data(**ndata)
            if nid in self.roots and self[nid].parent is not None:
                self.roots.remove(nid)
        self.is_consistent()
    #

    def is_consistent(self):
        """
        Check if the parent - children attributes of the nodes are consisten among them.
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
                        f'Inconsistency found: {nid} has {node.parent} as parent, but it is not in its children set.')
                    out = False
        return out
    #

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
        :py:meth:`is_consistent`
        """

        out = False
        nonroots = set(self.enumerate()) - self.roots
        if nonroots:
            out = True
        return out
    #

    def change_node_id(self, old_id, new_id):
        """
        Change the id of a Node of the Tree and update all its relatives.

        Arguments
        ---------

            old_id, new_id : str
                The current id and the desired new one.
        """

        if new_id in self:
            error_message(
                f'{new_id} is already present. Cant rename {old_id} to {new_id}.')
            return

        self[old_id].id = new_id
        self[new_id] = self.pop(old_id)

        if self[new_id].parent in self:
            self[self[new_id].parent].remove_child(old_id)
            self[self[new_id].parent].add_child(new_id)

        elif self[new_id].parent is None:
            self.roots.remove(old_id)
            self.roots.add(new_id)

        for cid in self[new_id].children:
            self[cid].parent = new_id
    #

    @staticmethod
    def from_hierarchy_dict(hierarchy):
        """
        Build a tree object infering the hierarchy from a dictionary.
        The dictionary must contain the tree nodes as dictionaries themselves.
        Each node-dict must have the pairs 'id': id, 'parent' : parent_id,
        children : [child_id1, child_id2,....], the following dict is an exemple
        node-dict. Note that children must be an iterable of 'ids' that will be
        turned into a set, duplications of ids are disregarded.

        {
            '1' : {'id'       : '1',
                   'parent'   : None,
                   'children' : {}
                  }
        }

        In the following exemple, a Boundaries object is created with a root node
        whose id is '1', with a child node '2', and whose center is at (x1,y1,z1). The
        node '2', has a child '0', its parent is '1', and its center is (x2,y2,z2).
        Finally, node '0', has no children, its parent is '2' and its center is (x0,y0,z0).

        hierarchy = {"1" : {"id"       : "1"
                            "parent"   : None,
                            "center"   : [ x1, y1, z1],
                            "children" : {"2"}
                           }
                     "2" : {"id"       : "2"
                            "parent"   : '1',
                            "center"   : [ x2, y2, z2],
                            "children" : {"0"}
                           }
                     "0" : {"id"       : "0",
                            "parent"   : '2',
                            "center"   : [ x0, y0, z0],
                            "children" : {}
                           }
                    }

        Arguments:
        -----------
            hierarchy : dict
                The dictionary with the hierarchy.

        """

        tree = Tree()

        roots = [nid for nid, node in hierarchy.items() if node['parent'] in [
            None, 'None']]

        def add_node(nid):  # , children, parent=None, **kwargs):

            for k in Node().__dict__:
                if k not in hierarchy[nid]:
                    error_message(
                        f'cant build hierarchy base on dict. Node {nid} has no entry for {k}')
                    return False

            n = Node()
            n.id = nid
            n.set_data(**hierarchy[nid])
            tree[n.id] = n
            for cid in n.children:
                add_node(nid=cid)
            return True

        for rid in roots:
            add_node(nid=rid)

        return tree
    #
#


def check_specific(params, nid, arg, default):
    """
    This function checks if the params dict contains params[nid][arg].
    Then if it exists it is return, otherwise default argument is returned.

    This function is meant to be used to filter out node specific parameters in
    in functions applied to a Tree.

    Arguments
    ---------

        params : dict
            The Node specific parameter dict.

        arg : str
            The agument name to check for.

        nid : str
            The node id to check for.

        default : Any
            The default value the param arg must have.

    Returns
    -------
        : Any
            Either params[nid][arg] or default.
    """

    try:
        return params[nid][arg]
    except KeyError:
        return default
#


def is_sequence(obj):
    """
    Check wether an object is a sequence.

    Arguments:
    ----------

        obj : any
            The object to be checked.

    Returns:
    ---------
        True or False.
    """
    if isinstance(obj, str) and len(obj) < 2:
        return False

    return hasattr(obj, '__iter__') and callable(getattr(obj, '__iter__'))
#


def is_numeric(obj):
    """
    Check whether a object is numeric.

    Arguments:
    ----------

        seq : iterable object.
            The sequence to test.

    Returns:
    ----------
        True or False
    """

    numeric = (int, float)
    if not isinstance(obj, numeric):
        return False
    return True
#


def is_arrayable(seq):
    """
    Check whether a sequence is all numeric and safe to be casted it to a numpy array.
    This function is used to parse float list as numpy arrays but preventing strings and
    other actual arrayable functions to be a numpy array.

    Arguments:
    -----------

        seq : any
            The object to be tested.

    Returns:
    ----------
        True or False.
    """

    if not is_sequence(seq):
        return False

    for elmnt in seq:
        if is_sequence(elmnt):
            if not is_arrayable(elmnt):
                return False
        else:
            if not is_numeric(elmnt):
                return False

    return True
#


def attribute_checker(obj, atts, info=None, opts=None):
    """
    Function to check if attribute has been set and print error message.

    Arguments:
    ------------

        obj : any,
            The object the attributes of which will be checked.

        atts : list[str]
            The names of the attributes to be checked for.

        info : str, opt
            An information string to be added to error message before
            'Attribute {att} is None....'. If None, no message is printed.

        opts : List[Any], optional.
            Default None. A list containing accepted values for attribute.

    Returns:
    --------
        True if all the attributes are different to None or in provided options.
        False otherwise.
    """

    if opts is None:
        for att in atts:
            if getattr(obj, att) is None:
                if info is not None:
                    error_message(
                        info=f'{info}. Attribute {att} is {getattr(obj, att)}....')
                return False

    else:
        for att, opt in zip(atts, opts):
            if getattr(obj, att) not in opt:
                if info is not None:
                    error_message(
                        info=f'{info}. Attribute {att} is {getattr(obj, att)}, and it must be in [{opt}]....')
                return False

    return True
#


def attribute_setter(obj, **kwargs):
    """
    Function to set attributes passed in a dict-way.
    """
    for k, v in kwargs.items():
        setattr(obj, k, v)
#
