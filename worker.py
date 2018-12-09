# ------------------------------------------------------------
# Implements a worker that runs backpropagation on batches
# provided by the server. If no server exists, then Paxos
# is called to generate a server.
# ------------------------------------------------------------

import time
import sys
import argparse
import traceback

import grpc
from protos import distributedML_pb2
from protos import distributedML_pb2_grpc


import autograd.numpy as np
from autograd import grad

import utils
import ann

import param_server
import paxos

# Loads in a really small version of the data that could fit in Github.
# It will train extremely quickly as a result.
images_fname = 'data/images(16).npy'
labels_fname = 'data/output_labels(16).npy'

_TIMEOUT_SECONDS = 20
TENSOR_TIMEOUT_SECONDS = 60
SERVER_PORT = 50051


# Loops through all possible addresses that are part of the instance
# group if this is launched on a remote server. Loops through all possible
# addresses that are part of the local server as well.
# Determines whether or not a server exists by trying to connect with the
# a predefined port on the server
def find_server(local_id=None):
    TOT_ATTEMPTS = 1
    for i in range(TOT_ATTEMPTS):
        # Generates local address information
        local_address = utils.server_address(local_id)
        server_addresses = utils.list_server_addresses(local_id)
        server_addresses.remove(local_address)

        # Loops through all the servers and tries to makes the server stub
        for server_address in server_addresses:
            if local_id is not None:
                channel = grpc.insecure_channel('%s:%d' % ('localhost', SERVER_PORT))
            else:
                # This is for remote server instances
                channel = grpc.insecure_channel('%s:%d' % (server_address, SERVER_PORT))
            stub = distributedML_pb2_grpc.ParamFeederStub(channel)
            try:
                # Attempts to ping the server to see if the port is open
                response = stub.ping(distributedML_pb2.empty(), _TIMEOUT_SECONDS)

                # If the PING succeeds, then it is the server
                return server_address

            except Exception as e:
                # Log any network or expiration errors we run into
                if ('ExpirationError' in str(e) or 'NetworkError' in str(e)):
                    utils.log_info(str(e))
                    continue
                else:
                    # More severe error, should log and crash
                    traceback.print_exc()
                    sys.exit(1)
        time.sleep(1 * TOT_ATTEMPTS)
    return ''


# After determining the correct server, generate the stub for it
def server_stub(server_addr, local_id):
    if local_id is not None:
        channel = grpc.insecure_channel('%s:%d' % ('localhost', SERVER_PORT), options=[('grpc.min_reconnect_backoff_ms', 100)])
    else:
        # TODO: channel for remote connection
        channel = grpc.insecure_channel('%s:%d' % (server_addr, SERVER_PORT), options=[('grpc.min_reconnect_backoff_ms', 100)])
    stub = distributedML_pb2_grpc.ParamFeederStub(channel)
    return stub


# Main function of the worker that loops forever. Receieves parameters and
# batch information from the server. Calculates gradients and sends them
# to the server
def run(local_id=None):
    # Load and process Caltech data
    train_images, train_labels, test_images, test_labels = ann.load_caltech100(images_fname, labels_fname)
    image_input_d = train_images.shape[1]

    # Network parameters
    layer_sizes = [image_input_d, 800, 600, 400, 350, 250, 101]

    L2_reg = 1.0

    # Training parameters
    param_scale = 0.1
    momentum = 0.9
    batch_size = 256
    num_epochs = 50

    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = ann.make_nn_funs(layer_sizes, L2_reg)
    loss_grad = grad(loss_fun)

    # Train with sgd
    batch_idxs = ann.make_batches(train_images.shape[0], batch_size)
    cur_dir = np.zeros(N_weights)

    # Previous batch for the purpose of timing
    prev_data_indx = -1

    # Number of consective expirations, used to detect server failure
    consec_expiration = 0

    # Determine the server address by running Paxos or pinging all addresses
    server_addr = ''
    while server_addr == '':
        server_addr = paxos.run_paxos(local_id)
        if server_addr == '':
            server_addr = find_server(local_id)
    utils.log_info('Server address is ' + server_addr)

    # If this worker is selected to be server, then transform into a server
    if server_addr == utils.server_address(local_id):
        utils.log_info('Transforming into the server')
        try:
            param_server.serve(server_addr, None, prev_data_indx, local_id)
        except KeyboardInterrupt as e:
            utils.log_info('interrupted')
            sys.exit(0)
        return

    # Generates the server stub and connects with it
    stub = server_stub(server_addr, local_id)
    worker_id = 0

    utils.log_info('Data loaded and connected to server:')

    while True:
        try:
            # Gets the next batch that it should run
            response = stub.SendNextBatch(distributedML_pb2.PrevBatch(worker_id=worker_id, prev_data_indx=prev_data_indx),
                                          _TIMEOUT_SECONDS)
            while response.data_indx != -2:
                worker_id = response.worker_id
                # If this fails, it keeps on trying to get your first batch
                while response.data_indx == -1:
                    time.sleep(5)
                    utils.log_info('Waiting for server to send next batch')
                    response = stub.SendNextBatch(
                        distributedML_pb2.PrevBatch(worker_id=worker_id, prev_data_indx=prev_data_indx), _TIMEOUT_SECONDS)
                utils.log_info('Processing parameters in batch %d!' % response.data_indx)

                # Generates the W matrix
                get_parameters_time = time.time()
                W_bytes = ''
                W_subtensors_iter = stub.SendParams(distributedML_pb2.WorkerInfo(worker_id=worker_id), TENSOR_TIMEOUT_SECONDS)
                for W_subtensor_pb in W_subtensors_iter:
                    W_bytes = W_bytes + W_subtensor_pb.tensor_content
                W = utils.convert_bytes_to_array(W_bytes)
                utils.log_info('Received parameters in {0:.2f}s'.format(time.time() - get_parameters_time))

                # Calculate the gradients
                grad_start = time.time()
                grad_W = loss_grad(W, train_images[batch_idxs[response.data_indx]],
                                   train_labels[batch_idxs[response.data_indx]])
                utils.log_info('Done calculating gradients in {0:.2f}s'.format(time.time() - grad_start))

                # Serialize the gradients
                tensor_compress_start = time.time()
                tensor_bytes = utils.convert_array_to_bytes(grad_W)
                tensor_iterator = utils.convert_tensor_iter(tensor_bytes, response.data_indx)
                utils.log_info('Done compressing gradients in {0:.2f}s'.format(time.time() - tensor_compress_start))

                # Send the gradients
                send_grad_start = time.time()
                stub.GetUpdates(tensor_iterator, _TIMEOUT_SECONDS)
                utils.log_info('Done sending gradients through in {0:.2f}s'.format(time.time() - send_grad_start))

                # Get the next batch to process
                prev_data_indx = response.data_indx
                response = stub.SendNextBatch(distributedML_pb2.PrevBatch(worker_id=worker_id, prev_data_indx=prev_data_indx),
                                              _TIMEOUT_SECONDS)

                consec_expiration = 0
        except KeyboardInterrupt as e:
            sys.exit(1)
        except Exception as e:
            if ('ExpirationError' in str(e) or 'NetworkError' in str(e)):
                SERVER_CONSEC_FAILURE = 2
                # Count the failures of the server
                consec_expiration += 1

                # If consecutive failures exceed a predefined value, then we look for
                # the server by pinging available instances or by restarting Paxos
                if consec_expiration == SERVER_CONSEC_FAILURE:
                    utils.log_info('Failure to connect to server_stub. Starting Paxos')
                    # Launches paxos and then looks for the server
                    while server_addr == '':
                        server_addr = paxos.run_paxos(local_id)
                        if server_addr == '':
                            server_addr = find_server(local_id)
                    # Generates the server if it is chosen to be the server
                    if server_addr == utils.server_address(local_id):
                        param_server.serve(server_addr, W, prev_data_indx, local_id)
                        return
                    # Connects to the server
                    stub = server_stub(server_addr, local_id)
            else:
                utils.log_info(traceback.print_exc())
                if ('UNAVAILABLE' in str(e)):
                    print('Traceback contains UNAVAILABLE error, retrying.')
                #sys.exit(0)
    sys.exit(1)



if __name__ == '__main__':
    utils.log_info('Starting worker')
    parser = argparse.ArgumentParser()
    parser.add_argument('--id')
    args = parser.parse_args()

    # Local id is only used if running the worker on localhost
    local_id = args.id
    if local_id is not None:
        local_id = int(local_id)
        assert (local_id > 0)
    while True:
        run(local_id)
