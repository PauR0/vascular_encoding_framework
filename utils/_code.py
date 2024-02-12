
import messages as msg

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
        if check(att) is None:
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
