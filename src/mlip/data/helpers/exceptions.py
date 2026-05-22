class DatasetsHaveNotBeenProcessedError(Exception):
    """Exception to be raised if dataset info is not available yet."""


class SplitProportionsInvalidError(Exception):
    """Exception to be raised if data split proportions don't sum up to one."""


class GroupIDNotInSplitError(Exception):
    """Exception to be raised if group ID not found in the given splits."""


class GraphsDiscardedError(Exception):
    """Exception to be raised if some graphs are invalid due to the given parameters
    for batch size, max. number of nodes, and max. number of edges.
    """
