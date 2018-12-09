# ------------------------------------------------------------
# Implements a Paxos server and runs Paxos with this server. 
# This function is called through run_paxos if the client_server
# has gone down.
# ------------------------------------------------------------

import grpc
import time
import sys
from threading import Thread
from concurrent import futures

from protos import paxos_pb2
from protos import paxos_pb2_grpc
import argparse
import traceback

import utils

import random


_TIMEOUT_SECONDS = 4
PAXOS_PORT_STR = 50052


# Actual implementation of the PaxosServer that is used to communicate between the clients.
# Paxos is called to determine the future main server from amongst many different clients.
class PaxosServer(paxos_pb2_grpc.PaxosNodeServicer):
    def __init__(self, hostname):
        # Initial consensus value is none, this will be the server
        self.new_server = ''
        self.consensus_value = None
        self.consensus_reached = False

        # Values for paxos
        self.seq_no = random.random()
        self.prop_seq_no = 0
        self.value = ''
        self.seq_no_v = 0

        # Exponential backoff to prevent spamming other servers
        # Randomness is introduced to help Paxos converge quicker
        self.backoff = (1 * random.gauss(1, 0.25))
        if self.backoff < 0:
            self.backoff = 1

        # Saves the server's address as well
        self.address = hostname

    # Runs the prepare phase of the Paxos algorithm
    def prepare(self, request, context):
        # Update the highest seen proposal
        if request.seq_no > self.prop_seq_no:
            self.prop_seq_no = request.seq_no
        # Returns an acknowledgement containing highest accepted proposal
        return paxos_pb2.ack(seq_no=self.seq_no, value=self.value, seq_no_v=self.seq_no_v)

    # Accepts the proposal if it is higher than
    def accept(self, request, context):
        if request.seq_no >= self.prop_seq_no:
            self.seq_no_v = request.seq_no
            self.value = request.value
            self.seq_no = request.seq_no
            return paxos_pb2.acceptance(accept_bool=True)
        else:
            return paxos_pb2.acceptance(accept_bool=False)

    # Notifies the server that consensus has been reached
    def accepted(self, request, context):
        self.consensus_reached = True
        self.new_server = request.value
        return paxos_pb2.blank()

    # Ping function to allow confirmation between PaxosServer that they
    # are still running
    def ping(self, request, context):
        return paxos_pb2.blank()

# Runs the PaxosServer. Checks periodically to see if a consensus has
# been reached.
def run_server(server, paxos_server):
    server.start()
    while True:
        time.sleep(0.1)
        try:
            if paxos_server.consensus_reached:
                if paxos_server.new_server != '':
                    utils.log_info('Consensus reached, server shutting down')
                # Wait briefly for the consensus message to propogate out
                time.sleep(5)
                server.stop(0)
                break
            time.sleep(1)
        except KeyboardInterrupt:
            server.stop(0)


# Actually instantiates the Paxos Server according to a defined port
def create_server(hostname, local_id):
    # Allow argument that allows this parameter to be changed
    paxos_server = PaxosServer(hostname)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=None))# thread_pool=None)
    paxos_pb2_grpc.add_PaxosNodeServicer_to_server(paxos_server, server)
    if local_id is None:
        # Remote connection
        server.add_insecure_port(hostname + ':' + str(PAXOS_PORT_STR))
    else:
        server.add_insecure_port(hostname)
        utils.log_info('server created ' + hostname)
    return paxos_server, server


# Attempts to send proposals to all the other servers
def send_proposals(server_stubs, self_paxos_server):
    # Increments the proposal number from the previous one that it sends out
    self_paxos_server.seq_no = self_paxos_server.seq_no * (1 + random.random())
    self_paxos_server.value = self_paxos_server.address
    seq_no_proposal = self_paxos_server.seq_no
    value = self_paxos_server.address
    utils.log_info('Making a proposal from {0} for n = {1} '.format(self_paxos_server.address, seq_no_proposal))

    # Track the failures of the proposals
    seq_no_so_far = 0
    failed = False
    responded = 0

    for server_stub in server_stubs:
        # Makes the connection to the server
        try:
            # gRPC call to other Paxos Servers to see if they acceept the proposal
            response = server_stub.prepare(paxos_pb2.proposal(seq_no=seq_no_proposal), _TIMEOUT_SECONDS)

            # Sees a higher n value then it's current value and immediately stops the process
            if response.seq_no >= seq_no_proposal:
                failed = True
                utils.log_info('Proposal ' + str(seq_no_proposal) + ' failed')
                break
            else:
                # If the response is positive, then it notes the positive response
                if response.seq_no_v > seq_no_so_far:
                    seq_no_so_far = response.seq_no
                    value = response.value
                responded += 1
        except Exception as e:
            if ('ExpirationError' in str(e)):
                utils.log_info('Failure to connect to server_stub')
                continue
            else:
                # More severe error, should log and crash
                traceback.print_exc()
                sys.exit(1)

    # No proposals have been sent so far, suggests its own IP
    if value is None:
        value = self_paxos_server.address

    # If it does not have a majority of responses, Paxos fails
    if responded < len(server_stubs) / 2.0:
        failed = True

    return (failed, seq_no_proposal, value)


# Requests that the other Paxos Server accepts the proposal
def request_accept(server_stubs, self_paxos_server, seq_no_proposal, value):
    accepted = 0
    for stub in server_stubs:
        try:
            response = stub.accept(paxos_pb2.request_acceptance(seq_no=seq_no_proposal, value=value), _TIMEOUT_SECONDS)
        except Exception as e:
            traceback.print_exc()
            return False
        if response.accept_bool:
            accepted += 1

    # If the majority accept the proposal, then it passes
    if accepted > len(server_stubs) / 2.0:
        utils.log_info('Proposal accepted')
        return True
    else:
        utils.log_info('Proposal {0} rejected with value {1}'.format(seq_no_proposal, value))
        return False


# Checks to ensure that all the stubs are currently available by pinging them
# If more than half of them are available, it begins Paxos. Otherwise, it waits.
def check_stubs_up(stubs):
    responses = 0
    for stub in stubs:
        try:
            response = stub.ping(paxos_pb2.blank(), _TIMEOUT_SECONDS)
            responses += 1
        except Exception as e:
            if ('ExpirationError' in str(e)):
                utils.log_info('Failure to connect to server_stub during startup')
                continue
            else:
                traceback.print_exc()
                sys.exit(1)
    if responses < len(stubs) / 2:
        return False
    else:
        return True


# Make sure that all machines are aware that the Paxos algorithm is finishing
# Not all machines are aware that the server has failed at the same time. Could
# be in the middle of calculating gradients or waiting to be timed out.
def gen_server_stubs(self_paxos_server, local_id):
    TOT_ATTEMPTS = 3
    for i in range(TOT_ATTEMPTS):
        server_addresses = utils.list_server_addresses(local_id)
        server_addresses.remove(self_paxos_server.address)
        stubs = []
        for server_address in server_addresses:
            if not self_paxos_server.consensus_reached:
                if local_id is not None:
                    server_port = int(server_address[-5:])
                    channel = grpc.insecure_channel('%s:%d' % ('localhost', server_port))
                else:
                    # TODO: Remote connection
                    channel = grpc.insecure_channel('%s:%d' % (server_address, PAXOS_PORT_STR))

                stub = paxos_pb2_grpc.PaxosNodeStub(channel)
                stubs.append(stub)
        all_stubs_responsive = check_stubs_up(stubs)
        if all_stubs_responsive:
            return stubs
        time.sleep(1 * TOT_ATTEMPTS)
    return None


# Sends to all servers that consensus was reached and a server was chosen.
def broadcast_consensus(server_stubs, self_paxos_server, value):
    for stub in server_stubs:
        response = stub.accepted(paxos_pb2.consensus(seq_no=self_paxos_server.seq_no, value=value), 2 * _TIMEOUT_SECONDS)


# Begins the Paxos protocol
def start_paxos(server_stubs, self_paxos_server):
    proposal_failed, seq_no_proposal, value = send_proposals(server_stubs, self_paxos_server)
    if not proposal_failed and not self_paxos_server.consensus_reached:
        # Have everyone accept the proposal
        accepted = request_accept(server_stubs, self_paxos_server, seq_no_proposal, value)
        if accepted and not self_paxos_server.consensus_reached:
            # If accepted, let everyone know that the server has been chosen
            broadcast_consensus(server_stubs, self_paxos_server, value)
            self_paxos_server.new_server = value
            self_paxos_server.consensus_reached = True
            return True

    # If proposal failed, backoff to try again later
    self_paxos_server.backoff = self_paxos_server.backoff * (1 + 10 * random.random())
    return False


# Client loops and runs the paxos algorithm every few seconds
def paxos_loop(self_paxos_server, local_id):
    time_slept = 0
    send_proposal_time = self_paxos_server.backoff

    while not self_paxos_server.consensus_reached:
        time.sleep(0.1)
        time_slept += 0.1

        # Send a proposal at allocated time
        if time_slept > send_proposal_time and not self_paxos_server.consensus_reached:
            time.sleep(random.random())
            server_stubs = gen_server_stubs(self_paxos_server, local_id)
            if server_stubs is None:
                self_paxos_server.new_server = ''
                break
            start_paxos(server_stubs, self_paxos_server)
            send_proposal_time = (random.gauss(1, 0.25) * self_paxos_server.backoff)
            time_slept = 0

        # If proposal fails, revert to checking for a server
        if send_proposal_time > 60:
            self_paxos_server.consensus_reached = True
            self_paxos_server.consensus_value = ''
            break


# This is the final function that exterior functions like client.py will call
def run_paxos(local_id=None):
    # Generates the host name
    hostname = utils.server_address(local_id)
    utils.log_info(hostname + ' called to run Paxos for determining the server')

    # Generates the server
    paxos_server, server = create_server(hostname, local_id)
    try:
        # Launch the server on a separate thread
        Thread(target=run_server, args=(server, paxos_server,)).start()
        start_paxos = time.time()

        # Begin to run Paxos
        paxos_loop(paxos_server, local_id)
        if paxos_server.new_server != '':
            utils.log_info('Done, new server is: {0} finished paxos in {1:2}s'.format(paxos_server.new_server,
                                                                                time.time() - start_paxos))
        else:
            # New server is empty only when a suitable server was not found after a predefined amount of time
            utils.log_info('Failure to connect to other allocated instances. Stopping paxos.')
    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        paxos_server.consensus_reached = True
        server.stop(0)
    return paxos_server.new_server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id')
    args = parser.parse_args()
    local_id = args.id
    if local_id is not None:
        local_id = int(local_id)
        assert (local_id > 0)
    utils.log_info(run_paxos(local_id))
