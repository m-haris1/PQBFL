import requests
import gzip, io
import os, sys, time, ast
import tarfile
import pickle
import tenseal as ts
import torch

from eth_account.messages import *
from eth_keys import keys
from eth_account._utils.legacy_transactions import serializable_unsigned_transaction_from_dict
from eth_account._utils.signing import to_standard_v
from eth_account.datastructures import SignedMessage

from Crypto.Util.number import *
from Crypto.Hash import SHAKE128, SHA384
from Crypto.Cipher import AES
from collections import defaultdict
import numpy as np
import hashlib
import struct

# -----------------------------
# IPFS CONFIG
# -----------------------------
IPFS_API = "http://127.0.0.1:5001/api/v0"

def ipfs_version():
    """Check IPFS version"""
    try:
        r = requests.post(f"{IPFS_API}/version")
        return r.json()
    except Exception as e:
        print(f"⚠️ Could not connect to IPFS daemon: {e}")
        return None

def upload_to_Ipfs(wrapped_data, ETH_address=None):
    """Upload compressed data to IPFS, return CID"""
    try:
        compressed_data = gzip.compress(wrapped_data)
        files = {"file": ("data.bin", compressed_data)}
        r = requests.post(f"{IPFS_API}/add", files=files)
        r.raise_for_status()
        return r.json()["Hash"]
    except Exception as e:
        print(f"⚠️ Upload to IPFS failed: {e}")
        return None

def get_from_Ipfs(Ipfs_id, client_address=None):
    """Fetch data back from IPFS CID"""
    try:
        r = requests.post(f"{IPFS_API}/cat", params={"arg": Ipfs_id})
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"⚠️ Fetch from IPFS failed: {e}")
        return None

# -----------------------------
# CRYPTO HELPERS
# -----------------------------
def kdf(x):
    return SHAKE128.new(x).read(32)

def wrapfiles(*files):   # ('A.bin', A), ('B.enc', B)
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
        for file_name, file_data in files:
            file_info = tarfile.TarInfo(name=file_name)
            file_info.size = len(file_data)
            tar.addfile(file_info, io.BytesIO(file_data))
    return tar_buffer.getvalue()

def unwrap_files(tar_data):
    extracted_files = {}
    tar_buffer = io.BytesIO(tar_data)
    with tarfile.open(fileobj=tar_buffer, mode='r') as tar:
        for member in tar.getmembers():
            file = tar.extractfile(member)
            if file is not None:
                extracted_files[member.name] = file.read()
    return extracted_files

def unzip(gzip_data):
    with gzip.GzipFile(fileobj=io.BytesIO(gzip_data)) as gz_file:
        return gz_file.read()

def serialize_data(encrypted_model, metadata, HE_algorithm):
    data_package = {'weights': {}, 'metadata': metadata, 'algorithm': HE_algorithm}
    for name, enc_weight in encrypted_model.items():
        data_package['weights'][name] = enc_weight.serialize()
    buffer = io.BytesIO()
    pickle.dump(data_package, buffer)
    return buffer.getvalue()

def deserialize_data(serialized_data, context):
    data_package = pickle.loads(serialized_data)
    serialized_weights = data_package['weights']
    metadata = data_package['metadata']
    algorithm = data_package['algorithm']
    deserialized_weights = {}
    for name, weight_bytes in serialized_weights.items():
        if algorithm == 'BFV':
            deserialized_weights[name] = ts.bfv_vector_from(context, weight_bytes)
        else:
            deserialized_weights[name] = ts.ckks_vector_from(context, weight_bytes)
    return deserialized_weights, metadata

# -----------------------------
# HOMOMORPHIC ENCRYPTION HELPERS
# -----------------------------
def HE_encrypt_model(model, context, HE_algorithm):
    def get_adaptive_scale(tensor):
        abs_max = np.abs(tensor).max()
        return min(1e6, max(1e4, 10 ** (np.ceil(np.log10(1 / (abs_max + 1e-9))) + 2))) 

    scaling_factors = {'conv': defaultdict(lambda: 1e5),
                       'fc': defaultdict(lambda: 1e4),
                       'bn': defaultdict(lambda: 1e3)}

    if HE_algorithm == "CKKS":
        context.global_scale = 2 ** 40
        context.auto_relin_size = 20
    context.generate_galois_keys()
    context.generate_relin_keys()

    encrypted_weights = {}
    metadata = {'scaling_factors': {}, 'norms': {}, 'num_clients': 1}

    for name, param in model.named_parameters():
        param_data = param.detach().cpu().numpy()
        layer_type = 'conv' if 'conv' in name else 'fc' if 'fc' in name else 'bn'

        if HE_algorithm == "BFV":
            norm = np.linalg.norm(param_data)
            if norm != 0:
                param_data = param_data / norm
            metadata['norms'][name] = norm
            scale = get_adaptive_scale(param_data)
            if "bias" in name:
                scale *= 0.1
            metadata['scaling_factors'][name] = scale
            param_data = np.round(param_data * scale)
            param_data = param_data.flatten().astype(int).tolist()
            encrypted_weights[name] = ts.bfv_vector(context, param_data)
        elif HE_algorithm == "CKKS":
            scale = scaling_factors[layer_type][name]
            metadata['scaling_factors'][name] = scale
            param_data = param_data.flatten().tolist()
            encrypted_weights[name] = ts.ckks_vector(context, param_data, scale=context.global_scale)

    return encrypted_weights, metadata

def HE_decrypt_model(encrypted_weights, model, context, HE_algorithm, metadata):
    context.generate_galois_keys()
    context.generate_relin_keys()
    decrypted_weights = {}
    state_dict = model.state_dict()
    num_clients = metadata.get('num_clients', 1)

    for name, encrypted_weight in encrypted_weights.items():
        try:
            decrypted_weight = encrypted_weight.decrypt()
            if HE_algorithm == "BFV":
                decrypted_weight = [x / num_clients for x in decrypted_weight]
                scale = metadata['scaling_factors'].get(name, 1.0)
                norm = metadata['norms'].get(name, 1.0)
                decrypted_weight = [x / scale for x in decrypted_weight]
                if norm not in (0, 1.0):
                    decrypted_weight = [x * norm for x in decrypted_weight]
                decrypted_weight = [x if abs(x) > 1e-6 else 0 for x in decrypted_weight]
            elif HE_algorithm == "CKKS":
                scale = metadata['scaling_factors'].get(name, 1.0)
                decrypted_weight = [x / scale for x in decrypted_weight]
            decrypted_weights[name] = decrypted_weight
        except Exception as e:
            print(f"Decryption failed for {name}: {str(e)}")
            continue

    for name, decrypted_weight in decrypted_weights.items():
        tensor_weight = torch.tensor(decrypted_weight, dtype=state_dict[name].dtype).view(state_dict[name].shape)
        if torch.isnan(tensor_weight).any() or torch.isinf(tensor_weight).any():
            print(f"Warning: Invalid values detected in {name}. Skipping update for this layer.")
            continue
        state_dict[name].copy_(tensor_weight)
    return model

# -----------------------------
# MISC CRYPTO HELPERS
# -----------------------------
def pubKey_from_tx(tx_hash, web3):
    tx = web3.eth.get_transaction(tx_hash)
    v = tx['v']
    r = int(tx['r'].hex(), 16)
    s = int(tx['s'].hex(), 16)
    unsigned_tx = serializable_unsigned_transaction_from_dict({
        'nonce': tx['nonce'],
        'gasPrice': tx['gasPrice'],
        'gas': tx['gas'],
        'to': tx['to'],
        'value': tx['value'],
        'data': tx['input']
    })
    tx_hash_bytes = unsigned_tx.hash()
    standard_v = to_standard_v(v)
    signature = keys.Signature(vrs=(standard_v, r, s))
    public_key = signature.recover_public_key_from_msg_hash(tx_hash_bytes)
    return public_key

def sign_data(msg, Eth_private_key, web3):
    if isinstance(msg, bytes):
        msg_hex = msg.hex()
    else:
        msg_hex = msg
    encoded_ct = encode_defunct(text=msg_hex)
    signed_ct = web3.eth.account.sign_message(encoded_ct, private_key=Eth_private_key)
    message_hash = signed_ct.messageHash
    r_bytes = long_to_bytes(signed_ct.r, 32)
    s_bytes = long_to_bytes(signed_ct.s, 32)
    v_bytes = long_to_bytes(signed_ct.v, 1)
    sign_bytes = signed_ct.signature
    return message_hash + r_bytes + s_bytes + v_bytes + sign_bytes

def verify_sign(signed_data, msg, pubkey):
    msg_hash = signed_data[:32]
    r_sign = bytes_to_long(signed_data[32:64])
    s_sign = bytes_to_long(signed_data[64:96])
    v_sign = bytes_to_long(signed_data[96:97])
    sign_bytes = signed_data[97:]
    signature = SignedMessage(messageHash=msg_hash, r=r_sign, s=s_sign, v=v_sign, signature=sign_bytes)
    if not signature:
        raise ValueError("Invalid signed message data structure.")
    return True

def AES_encrypt_data(key, msg):
    nonce = os.urandom(8)
    crypto = AES.new(key, AES.MODE_CTR, nonce=nonce)
    model_ct = crypto.encrypt(msg)
    return nonce + model_ct

def AES_decrypt_data(key, cipher):
    nonce = cipher[:8]
    crypto = AES.new(key, AES.MODE_CTR, nonce=nonce)
    return crypto.decrypt(cipher[8:])

def hash_data(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def receive_Model(sock):
    raw_size = sock.recv(4)
    if not raw_size:
        return None
    data_size = struct.unpack('!I', raw_size)[0]
    data = b''
    while len(data) < data_size:
        chunk = sock.recv(min(data_size - len(data), 4096))
        if not chunk:
            raise ConnectionError("Connection lost while receiving data")
        data += chunk
    return data

def send_model(sock, data):
    data_size = len(data)
    sock.sendall(struct.pack('!I', data_size))
    sock.sendall(data)
