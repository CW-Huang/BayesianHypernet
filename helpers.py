
def get_task(task_name):
    """
    returns:
        X, Y, output_function, loss_function, {other}
    """
    pass


def get_mushrooms():
    from mushroom_data import X,Y
    from lasagne.objectives import squared_error
    return X, Y, None, squared_error

def get_mnist():
    pass


