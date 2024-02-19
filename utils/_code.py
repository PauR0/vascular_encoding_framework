
from copy import copy

import messages as msg


class Node:
    """
    Abstract class for tree node.
    """

    def __init__(self, parent=None) -> None:

        self.parent = None
        if parent is not None:
            self.parent = parent

        self.children : set = set()
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

        super().__setitem__(__key, nd)

        if nd.parent is None:
            self.roots.add(__key)
        else:
            self[nd.parent].set_children(__key)
    #

    def graft(self, tr, r=None):
        """
        Merge another tree. If r is a node id of this tree,
        root nodes are rooted on self[r], otherwise they are
        kept as roots.

        Arguments:
            tr : Tree
                The tree to merge in this.

            r : node id
                The id of a node in this tree.
        """

        new_nodes = []
        for nid in tr.roots:
            n = tr[nid]
            if r is not None:
                n = copy(tr[nid])
                n.parent = r
            new_nodes.append(n)

        while new_nodes:
            nid = new_nodes.pop(0)
            new_nodes += list(tr[nid].children)
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

        children = [k]
        while children:
            #Adding all the children of the last node...
            children += list(self[children[0]].children)
            self.remove(k=children[0])
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

    check = lambda at: getattr(obj, at) is None
    if opts is not None:
        check = lambda at: getattr(obj, at) not in opts

    for att in atts:
        if check(att):
            msg.error_message(info=f"{extra_info}. Attribute {att} is None....")
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
