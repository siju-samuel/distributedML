# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/paxos.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/paxos.proto',
  package='distributedML',
  syntax='proto3',
  serialized_pb=_b('\n\x12protos/paxos.proto\x12\rdistributedML\"\x1a\n\x08proposal\x12\x0e\n\x06seq_no\x18\x01 \x01(\x02\"6\n\x03\x61\x63k\x12\x0e\n\x06seq_no\x18\x01 \x01(\x02\x12\r\n\x05value\x18\x02 \x01(\t\x12\x10\n\x08seq_no_v\x18\x03 \x01(\x02\"3\n\x12request_acceptance\x12\x0e\n\x06seq_no\x18\x01 \x01(\x02\x12\r\n\x05value\x18\x02 \x01(\t\"!\n\nacceptance\x12\x13\n\x0b\x61\x63\x63\x65pt_bool\x18\x01 \x01(\x08\"*\n\tconsensus\x12\x0e\n\x06seq_no\x18\x01 \x01(\x02\x12\r\n\x05value\x18\x02 \x01(\t\"\x07\n\x05\x62lank2\x83\x02\n\tPaxosNode\x12\x38\n\x07prepare\x12\x17.distributedML.proposal\x1a\x12.distributedML.ack\"\x00\x12H\n\x06\x61\x63\x63\x65pt\x12!.distributedML.request_acceptance\x1a\x19.distributedML.acceptance\"\x00\x12<\n\x08\x61\x63\x63\x65pted\x12\x18.distributedML.consensus\x1a\x14.distributedML.blank\"\x00\x12\x34\n\x04ping\x12\x14.distributedML.blank\x1a\x14.distributedML.blank\"\x00\x62\x06proto3')
)




_PROPOSAL = _descriptor.Descriptor(
  name='proposal',
  full_name='distributedML.proposal',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='seq_no', full_name='distributedML.proposal.seq_no', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=37,
  serialized_end=63,
)


_ACK = _descriptor.Descriptor(
  name='ack',
  full_name='distributedML.ack',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='seq_no', full_name='distributedML.ack.seq_no', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='distributedML.ack.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='seq_no_v', full_name='distributedML.ack.seq_no_v', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=65,
  serialized_end=119,
)


_REQUEST_ACCEPTANCE = _descriptor.Descriptor(
  name='request_acceptance',
  full_name='distributedML.request_acceptance',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='seq_no', full_name='distributedML.request_acceptance.seq_no', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='distributedML.request_acceptance.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=121,
  serialized_end=172,
)


_ACCEPTANCE = _descriptor.Descriptor(
  name='acceptance',
  full_name='distributedML.acceptance',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='accept_bool', full_name='distributedML.acceptance.accept_bool', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=174,
  serialized_end=207,
)


_CONSENSUS = _descriptor.Descriptor(
  name='consensus',
  full_name='distributedML.consensus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='seq_no', full_name='distributedML.consensus.seq_no', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='distributedML.consensus.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=209,
  serialized_end=251,
)


_BLANK = _descriptor.Descriptor(
  name='blank',
  full_name='distributedML.blank',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=253,
  serialized_end=260,
)

DESCRIPTOR.message_types_by_name['proposal'] = _PROPOSAL
DESCRIPTOR.message_types_by_name['ack'] = _ACK
DESCRIPTOR.message_types_by_name['request_acceptance'] = _REQUEST_ACCEPTANCE
DESCRIPTOR.message_types_by_name['acceptance'] = _ACCEPTANCE
DESCRIPTOR.message_types_by_name['consensus'] = _CONSENSUS
DESCRIPTOR.message_types_by_name['blank'] = _BLANK
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

proposal = _reflection.GeneratedProtocolMessageType('proposal', (_message.Message,), dict(
  DESCRIPTOR = _PROPOSAL,
  __module__ = 'protos.paxos_pb2'
  # @@protoc_insertion_point(class_scope:distributedML.proposal)
  ))
_sym_db.RegisterMessage(proposal)

ack = _reflection.GeneratedProtocolMessageType('ack', (_message.Message,), dict(
  DESCRIPTOR = _ACK,
  __module__ = 'protos.paxos_pb2'
  # @@protoc_insertion_point(class_scope:distributedML.ack)
  ))
_sym_db.RegisterMessage(ack)

request_acceptance = _reflection.GeneratedProtocolMessageType('request_acceptance', (_message.Message,), dict(
  DESCRIPTOR = _REQUEST_ACCEPTANCE,
  __module__ = 'protos.paxos_pb2'
  # @@protoc_insertion_point(class_scope:distributedML.request_acceptance)
  ))
_sym_db.RegisterMessage(request_acceptance)

acceptance = _reflection.GeneratedProtocolMessageType('acceptance', (_message.Message,), dict(
  DESCRIPTOR = _ACCEPTANCE,
  __module__ = 'protos.paxos_pb2'
  # @@protoc_insertion_point(class_scope:distributedML.acceptance)
  ))
_sym_db.RegisterMessage(acceptance)

consensus = _reflection.GeneratedProtocolMessageType('consensus', (_message.Message,), dict(
  DESCRIPTOR = _CONSENSUS,
  __module__ = 'protos.paxos_pb2'
  # @@protoc_insertion_point(class_scope:distributedML.consensus)
  ))
_sym_db.RegisterMessage(consensus)

blank = _reflection.GeneratedProtocolMessageType('blank', (_message.Message,), dict(
  DESCRIPTOR = _BLANK,
  __module__ = 'protos.paxos_pb2'
  # @@protoc_insertion_point(class_scope:distributedML.blank)
  ))
_sym_db.RegisterMessage(blank)



_PAXOSNODE = _descriptor.ServiceDescriptor(
  name='PaxosNode',
  full_name='distributedML.PaxosNode',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=263,
  serialized_end=522,
  methods=[
  _descriptor.MethodDescriptor(
    name='prepare',
    full_name='distributedML.PaxosNode.prepare',
    index=0,
    containing_service=None,
    input_type=_PROPOSAL,
    output_type=_ACK,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='accept',
    full_name='distributedML.PaxosNode.accept',
    index=1,
    containing_service=None,
    input_type=_REQUEST_ACCEPTANCE,
    output_type=_ACCEPTANCE,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='accepted',
    full_name='distributedML.PaxosNode.accepted',
    index=2,
    containing_service=None,
    input_type=_CONSENSUS,
    output_type=_BLANK,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ping',
    full_name='distributedML.PaxosNode.ping',
    index=3,
    containing_service=None,
    input_type=_BLANK,
    output_type=_BLANK,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_PAXOSNODE)

DESCRIPTOR.services_by_name['PaxosNode'] = _PAXOSNODE

# @@protoc_insertion_point(module_scope)
