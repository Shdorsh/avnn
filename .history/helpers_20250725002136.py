def flatten_avnn(tensor, num):
    return tensor.view(tensor.size(0), -1, num - 1)
