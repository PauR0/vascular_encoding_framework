
from copy import copy, deepcopy

import messages as msg


class Node:
    """
    Abstract class for tree node.
    """

    def __init__(self, nd=None) -> None:

        self.id             = None
        self.parent         = None
        self.children : set = set()

        if nd is not None:
            self.set_data_from_other_node(nd=nd)
    #

    def __str__(self):
        strout="\n".join([f"{k}".ljust(10, '.')+f": {v}" for k, v in self.__dict__.items() if k!='points'])
        if 'points' in self.__dict__:
            strout += "\npoints shape".ljust(10, '.')+f"{self.points.shape}"

        return strout
    #

    def set_data(self, **kwargs):
        """
        Method to set attributes by means of kwargs.
        E.g.
            a = Node()
            a.set_data(center=np.zeros((3,)))

        """

        if 'children' in kwargs:
            self.children = set(kwargs['children'])
            kwargs.pop('children')

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

        self.set_data(**{k : getattr(nd, k) for k in Node().__dict__})
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
        #This way we allow more than one tree to be hold. Actually more like a forest...
        self.roots : set = set()
        return
    #

    def __str__(self):
        outstr = ""
        ind = " "

        def append_str(nid, outstr, l=0):
            strout = "\n".join([ind*4*l+s for s in self[nid].__str__().split("\n")]) + "\n\n"
            for cid in self[nid].children:
                strout += append_str(cid, outstr, l=l+1)

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

        #Checking it has parent and children attributes. Since Nones are admited, attribute_checker is not well suited.
        for att in ['parent', 'children']:
            if not hasattr(nd, att):
                msg.error_message(f"Aborted insertion of node with id: {__key}. It has no {att} attribute.")
                return

        if nd.parent is not None and nd.parent not in self.keys():
            msg.error_message(f"Aborted insertion of node with id: {__key}. Its parent {nd.parent} does not belong to the tree.")
            return

        if not isinstance(__key, str):
            msg.warning_message(f"node {__key} has been set with a non-string key. This may turn in troubles...")
        if __key != nd.id:
            msg.warning_message(f"node id attribute is {nd.id} and node id in tree has been set as {__key}.")

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

        #Children are now roots
        for child in self[k].children:
            self[child].parent = None
            self.roots.add(child)

        #If k is a root remove from roots set, otherwise remove it from parent children set.
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

    def copy(self, deep=True):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)
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

        roots = [nid for nid, node in hierarchy.items() if node['parent'] in [None, 'None']]

        def add_node(nid):#, children, parent=None, **kwargs):

            for k in Node().__dict__:
                if k not in hierarchy[nid]:
                    msg.error_message(f"cant build hierarchy base on dict. Node {nid} has no entry for {k}")
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



def attribute_checker(obj, atts, extra_info='', opts=None):
    """
    Function to check if attribute has been set and print error message.

    Arguments:
    ------------

        obj : any,
            The object the attributes of which will be checked.

        atts : list[str]
            The names of the attributes to be checked for.

        extra_info : str, opt
            An extra information string to be added to error message befor
            'Attribute {att} is None....'

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
                msg.error_message(info=f"{extra_info}. Attribute {att} is {getattr(obj, att)}....")
                return False

    else:
        for att, opt in zip(atts, opts):
            if getattr(obj, att) not in opt:
                msg.error_message(info=f"{extra_info}. Attribute {att} is {getattr(obj, att)}, and it must be in {opt}....")
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
