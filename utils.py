import time
import autograd.numpy as np
from protos import distributedML_pb2

def log_info(value):
    print(str(time.time()) + ' ' + str(value))


def server_address(local_id):
    if local_id is not None:
        server_addresses = list_server_addresses(local_id)
        return server_addresses[local_id-1]
    else:
        # TODO: Fill this to find remote server address
        pass


def list_server_addresses(local_id):
    if local_id is None:
        # TODO: Fill this with remote server instances
        pass
    if local_id is not None:
        return ['[::]:50052', '[::]:50053', '[::]:50054']


def convert_bytes_to_array(param_bytes):
    params = np.fromstring(param_bytes, dtype=np.float32)
    return params


def convert_array_to_bytes(params):
    if (params.dtype == np.float64):
        params = params.astype(np.float32)
    param_bytes = params.tostring()
    return param_bytes


def convert_tensor_iter(tensor_bytes, data_indx):
    CHUNK_SIZE = 524228
    tensor_bytes_len = len(tensor_bytes)
    tensor_chunk_count = 0
    while len(tensor_bytes):
        tensor_chunk_count += 1
        tensor_content = tensor_bytes[:CHUNK_SIZE]
        tensor_bytes = tensor_bytes[CHUNK_SIZE:]
        yield distributedML_pb2.SubTensor(tensor_len=tensor_bytes_len, tensor_chunk=tensor_chunk_count,
                                     tensor_content=tensor_content, data_indx=data_indx)
