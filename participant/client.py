from web3 import Web3
from eth_account import Account
from eth_account.messages import *
from eth_keys import keys

from pqcrypto.kem import ml_kem_768

from Crypto.Protocol.DH import key_agreement
from Crypto.Protocol.KDF import HKDF
from Crypto.PublicKey import ECC
from Crypto.Hash import SHA384
from Crypto.Util.number import *  # Imports bytes_to_long, long_to_bytes

import tenseal as ts
import socket
import pickle
import json
import os
import sys
import time
import train_model

# --- FIX: Path manipulation must occur before external imports to resolve ModuleNotFoundError ---
pqbfl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.abspath(pqbfl_path))

from utils import (
    wrapfiles, unwrap_files, receive_Model, send_model,
    AES_encrypt_data, AES_decrypt_data, sign_data, verify_sign,
    hash_data, HE_encrypt_model, serialize_data, deserialize_data,
    HE_decrypt_model, kdf, pubKey_from_tx
)
# --- END FIX ---

# Global variables will be defined in __main__
w3 = None
contract = None
contract_address = None
ETH_address = None
Eth_private_key = None
registered_id_p = 0  # To be set after registration

# --- RESTORED HELPER FUNCTIONS ---

# def register_client(hash_epk, Project_id):
#     global w3, contract, ETH_address, registered_id_p
#     print(f"DEBUG: Entering register_client. hash_epk type: {type(hash_epk)} (Expected: str/hex hash)")
#     try:
#         Call_reg = contract.functions.registerClient(hash_epk, int(Project_id)).transact({'from': ETH_address})  # Send a registration transaction
#         receipt = w3.eth.wait_for_transaction_receipt(Call_reg)
#         gas_used = receipt['gasUsed']
#         tx_registration = receipt['transactionHash'].hex()
#         logs = receipt['logs']
#         log_data_bytes = logs[0]['data']
#         project_id_bytes = log_data_bytes[32:64]
#         project_id = int.from_bytes(project_id_bytes[-1:], byteorder='big', signed=True)
#         initial_score_bytes = log_data_bytes[64:96]
#         initial_score = int.from_bytes(initial_score_bytes[-1:], byteorder='big', signed=True)
#         offset = int.from_bytes(log_data_bytes[96:128], byteorder='big')
#         epk_len = int.from_bytes(log_data_bytes[offset:offset + 32], byteorder='big')  # Length of the publicKey
#         epk_bytes = log_data_bytes[offset + 32:offset + 32 + epk_len]  # Extract publicKey using its length
#         onchain_epk = epk_bytes.decode('utf-8')
#         assert onchain_epk == hash_epk, 'epk placed on the chain is not same as generated epk !!'
#         print('registration in Project : successfully')
#         print(f'    Project ID: {project_id}')
#         print(f'    Tx: {tx_registration}')
#         print(f'    Your Address: {ETH_address}')
#         print(f'    Gas: {gas_used} Wei')
#         print(f'    Initial Score: {initial_score}')
#         print(f'    PublicKey: {onchain_epk}')
#         print('-' * 75)
#     except Exception as e:
#         if "Registration completed" in str(e):
#             print("The project has reached its limit for client registrations. No more registrations are accepted.")
#             sys.exit()
#         else:
#             print(f"An unexpected error occurred: {e}")
#             sys.exit()
#     registered_id_p = project_id
#     return initial_score, tx_registration, project_id



# def register_client(hash_epk, Project_id):
#     """
#     Register client on-chain.
#     Accepts:
#       - hash_epk: either '0x...' hex string OR bytes OR plain hex string without 0x
#     Returns: initial_score, tx_registration, project_id
#     """
#     global w3, contract, ETH_address, registered_id_p
#     # Normalize: produce a plain hex string WITHOUT 0x for local comparisons
#     if isinstance(hash_epk, (bytes, bytearray)):
#         hash_epk_hex = hash_epk.hex()
#         abi_value = hash_epk  # pass bytes directly to contract
#     else:
#         # it's a string
#         s = str(hash_epk)
#         if s.startswith("0x") or s.startswith("0X"):
#             hash_epk_hex = s[2:]
#             abi_value = s  # pass 0x-prefixed string to contract
#         else:
#             hash_epk_hex = s
#             abi_value = "0x" + s

#     print(f"DEBUG: Entering register_client. normalized hash_epk_hex type: {type(hash_epk_hex)} value start: {hash_epk_hex[:10]}...")

#     try:
#         # Use abi_value when calling the contract so web3/eth-abi encodes correctly
#         Call_reg = contract.functions.registerClient(abi_value, int(Project_id)).transact({'from': ETH_address})
#         receipt = w3.eth.wait_for_transaction_receipt(Call_reg)
#         gas_used = receipt['gasUsed']
#         tx_registration = receipt['transactionHash'].hex()

#         logs = receipt.get('logs', [])
#         if not logs:
#             # no logs — return basic info
#             print("Warning: no logs found in registration receipt.")
#             registered_id_p = int(Project_id)
#             return 0, tx_registration, registered_id_p

#         log_data_bytes = logs[0]['data']
#         # parse same way as before (defensive)
#         project_id_bytes = log_data_bytes[32:64]
#         project_id = int.from_bytes(project_id_bytes[-1:], byteorder='big', signed=True)
#         initial_score_bytes = log_data_bytes[64:96]
#         initial_score = int.from_bytes(initial_score_bytes[-1:], byteorder='big', signed=True)

#         # Extract on-chain public key/hash if present (may be encoded with a length prefix)
#         try:
#             offset = int.from_bytes(log_data_bytes[96:128], byteorder='big')
#             epk_len = int.from_bytes(log_data_bytes[offset:offset + 32], byteorder='big')
#             epk_bytes = log_data_bytes[offset + 32:offset + 32 + epk_len]
#             onchain_epk = epk_bytes.decode('utf-8')
#         except Exception:
#             onchain_epk = None

#         # If contract placed a value on-chain (string), check it against normalized local hex
#         if onchain_epk is not None:
#             if onchain_epk.startswith("0x") or onchain_epk.startswith("0X"):
#                 onchain_hex = onchain_epk[2:]
#             else:
#                 onchain_hex = onchain_epk
#             assert onchain_hex.lower() == hash_epk_hex.lower(), "epk placed on the chain is not same as generated epk !!"

#         print('registration in Project : successfully')
#         print(f'    Project ID: {project_id}')
#         print(f'    Tx: {tx_registration}')
#         print(f'    Your Address: {ETH_address}')
#         print(f'    Gas: {gas_used} Wei')
#         print(f'    Initial Score: {initial_score}')
#         if onchain_epk:
#             print(f'    PublicKey: {onchain_epk}')
#         print('-' * 75)

#     except Exception as e:
#         # Show raw exception so we can see the exact cause if anything else fails
#         print(f"An unexpected error occurred during registration: {e}")
#         sys.exit(1)

#     registered_id_p = project_id
#     return initial_score, tx_registration, project_id


def register_client(hash_epk, project_id):
    """
    Register client on-chain.
    Contract signature: registerClient(string,uint256)

    Args:
        hash_epk (str or bytes): Hex string (with or without '0x') or raw bytes
        project_id (int): The project ID

    Returns:
        tuple: (initial_score, tx_registration, project_id)
    """
    global w3, contract, ETH_address, registered_id_p

    # Always normalize to hex string with '0x' prefix
    if isinstance(hash_epk, (bytes, bytearray)):
        abi_value = "0x" + hash_epk.hex()
    else:
        s = str(hash_epk)
        if not (s.startswith("0x") or s.startswith("0X")):
            abi_value = "0x" + s
        else:
            abi_value = s

    # Debug print
    print(f"DEBUG: Entering register_client. abi_value type={type(abi_value)} value start={abi_value[:10]}...")

    try:
        # Call the contract with a string (hex with 0x prefix)
        Call_reg = contract.functions.registerClient(abi_value, int(project_id)).transact({'from': ETH_address})
        receipt = w3.eth.wait_for_transaction_receipt(Call_reg)
        gas_used = receipt['gasUsed']
        tx_registration = receipt['transactionHash'].hex()

        # No need to manually decode logs if ABI events are set up, but leave defensive
        logs = receipt.get('logs', [])
        if not logs:
            print("Warning: no logs found in registration receipt.")
            registered_id_p = int(project_id)
            return 0, tx_registration, registered_id_p

        # Simplify: just print receipt info
        print("✔ Registration successful")
        print(f"    Project ID: {project_id}")
        print(f"    Tx: {tx_registration}")
        print(f"    Your Address: {ETH_address}")
        print(f"    Gas: {gas_used} Wei")
        print('-' * 75)

        registered_id_p = int(project_id)
        return 0, tx_registration, registered_id_p

    except Exception as e:
        print(f"An unexpected error occurred during registration: {e}")
        sys.exit(1)






def task_completed(task_id, project_id):
    global contract
    return contract.functions.isTaskDone(task_id, project_id).call()


def listen_for_projcet():
    global contract
    print("Listen for project...")
    while True:
        try:
            task_event_filter = contract.events.ProjectRegistered.create_filter(fromBlock="latest")
            events = task_event_filter.get_all_entries()
            if events:
                ev = events[0]
                project_id = ev['args']['project_id']
                cnt_clients = ev['args']['cnt_clients']
                server_address = ev['args']['serverAddress']
                creation_time = time.gmtime(int(ev['args']['transactionTime']))
                initial_model_hash = ev['args']['hash_init_model']
                server_hash_pubkeys = ev['args']['hash_keys']
                tx_hash = ev['transactionHash']
                print('Received Project Info:')
                print(f'    Poject ID: {project_id}')
                print(f'    Server address: {server_address}')
                print(f'    required client count: {cnt_clients}')
                print(f'    Time: {time.strftime("%Y-%m-%d %H:%M:%S (UTC)", creation_time)}')
                print(f'    Hash_pubkeys: {server_hash_pubkeys}')
                print('-' * 75)
                return tx_hash, project_id, server_address, cnt_clients, initial_model_hash, server_hash_pubkeys
        except Exception as e:
            print(f"Error fetching project events: {e}")
            break
        time.sleep(1)
    return None, None, None, None, None, None


def listen_for_task(timeout):
    global contract, registered_id_p
    print("Listen for task...")
    start_time = time.time()
    Task_id = Hashed_model = round_num = hash_keys = project_id_received = server_address = D_t = 0
    while True:
        try:
            task_event_filter = contract.events.TaskPublished.create_filter(fromBlock="latest")
            events = task_event_filter.get_all_entries()
            if events:
                event = events[0]
                round_num = event['args']['round']
                Task_id = event['args']['taskId']
                server_address = event['args']['serverAddress']
                Hashed_model = event['args']['HashModel']
                hash_keys = event['args']['hash_keys']
                project_id_received = event['args']['project_id']
                tx_hash = event['transactionHash'].hex()
                creation_time = time.gmtime(int(event['args']['creationTime']))
                D_t = time.gmtime(int(event['args']['DeadlineTask']))

                if registered_id_p == project_id_received:
                    print('Published Task Info:')
                    print(f'    Task ID: {Task_id}')
                    print(f'    Project ID: {project_id_received}')
                    print(f'    Server address: {server_address}')
                    print(f'    Transaction Hash: {tx_hash}')
                    print(f'    Time: {time.strftime("%Y-%m-%d %H:%M:%S (UTC)", creation_time)}')
                    print(f'    Deadline: {time.strftime("%Y-%m-%d %H:%M:%S (UTC)", D_t)}')
                    print('-' * 75)
                    return round_num, Task_id, Hashed_model, hash_keys, project_id_received, server_address, D_t
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                break
        except Exception as e:
            print(f"Error while fetching tasks: {e}")
            break
        time.sleep(1)
    return 0, 0, None, None, 0, None, 0


def update_model_Tx(r, Hash_model, hash_ct_epk, Task_id, project_id):
    global w3, ETH_address, Eth_private_key, contract
    print(f"DEBUG: Updating Model Tx. R: {r}, Model Hash: {Hash_model[:10]}..., ct_epk_hash: {hash_ct_epk[:10]}...")
    try:
        nonce = w3.eth.get_transaction_count(ETH_address)  # Fetch the latest nonce for the account
        transaction = contract.functions.updateModel(r, Hash_model, hash_ct_epk, Task_id, project_id).build_transaction({
            'from': ETH_address,
            'nonce': nonce,
            'gas': 2000000,  # Adjust gas limit if necessary
            'gasPrice': w3.to_wei('50', 'gwei')
        })
        signed_tx = w3.eth.account.sign_transaction(transaction, private_key=Eth_private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)  # Wait for transaction receipt
        gas_used = tx_receipt['gasUsed']
        tx_update = tx_receipt['transactionHash'].hex()
        print(' ')
        print('Train completed, model update Info:')
        print(f'    Tx: {tx_update}')
        print(f'    Gas: {gas_used} Wei')
        print('-' * 75)
        return tx_update
    except ValueError as e:
        print(f"Error occurred: {e}")
        if "nonce" in str(e):
            print("Retrying transaction with updated nonce...")
            return update_model_Tx(r, Hash_model, hash_ct_epk, Task_id, project_id)  # Recursive retry
        raise e


def listen_for_feedback(current_round, client_address, blocks_lookback=10):
    global w3, contract
    print(f"DEBUG: Listening for Feedback (R: {current_round})")
    latest_block = w3.eth.block_number
    start_block = max(0, latest_block - blocks_lookback)
    feedback_filter = contract.events.FeedbackProvided.create_filter(fromBlock=start_block)
    while True:
        feedback_events = feedback_filter.get_all_entries()  # Fetch events from the filter
        for feedback in feedback_events:
            event_client_address = feedback['args']['clientAddress']
            event_round = feedback['args']['round']
            if event_client_address == client_address and event_round == current_round:
                accepted = feedback['args']['accepted']
                task_id = feedback['args']['taskId']
                tx_hash = feedback['transactionHash'].hex()
                project_id = feedback['args']['project_id']
                T = feedback['args']['terminate']
                score_change = feedback['args']['scoreChange']
                server_addr = feedback['args']['serverId']
                print('Feedback Info:')
                print(f'    Tx: {tx_hash}')
                print(f'    Status: {accepted}')
                print(f'    Round: {event_round}')
                print(f'    Score: {score_change}')
                print(f'    Time: {time.strftime("%Y-%m-%d %H:%M:%S (UTC)", time.gmtime())}')
                print(f'    Server address: {server_addr}')
                print('-' * 75)
                return project_id, T, score_change
        time.sleep(1)
    return None, False, 0


if __name__ == "__main__":
    print("--- CLIENT START ---")
    # connect to ganache
    try:
        ganache_url = "http://127.0.0.1:7545"
        w3 = Web3(Web3.HTTPProvider(ganache_url))
        print("Client connected to blockchain (Ganache) successfully\n")
    except Exception as e:
        print("Exception connecting to Ganache:", e)
        sys.exit(1)

    # CLI args
    if len(sys.argv) < 6:
        print("Usage: python client.py <private_key> <contract_address> <num_epochs> <dataset_type> <HE_algorithm>")
        sys.exit(1)

    Eth_private_key = sys.argv[1]
    contract_address = sys.argv[2]
    num_epochs = int(sys.argv[3])
    dataset_type = sys.argv[4]
    HE_algorithm = sys.argv[5]
    print(f"DEBUG: Config: Epochs={num_epochs}, Dataset={dataset_type}, HE={HE_algorithm}")

    account = Account.from_key(Eth_private_key)
    ETH_address = account.address
    print(f"DEBUG: Client ETH Address: {ETH_address}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)

    with open(main_dir + "/contract/contract-abi.json", "r") as f:
        contract_abi = json.load(f)
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)

    HE_config_with_key = None
    if HE_algorithm == 'CKKS':
        with open(main_dir + f'/participant/keys/CKKS_with_priv_key.pkl', "rb") as f:
            serialized_with_key = pickle.load(f)
        HE_config_with_key = ts.context_from(serialized_with_key)
    elif HE_algorithm == 'BFV':
        with open(main_dir + f'/participant/keys/BFV_with_priv_key.pkl', "rb") as f:
            serialized_with_key = pickle.load(f)
        HE_config_with_key = ts.context_from(serialized_with_key)
    print(f"DEBUG: HE Config loaded (Context object type: {type(HE_config_with_key)})")

    Tx_r, project_id, server_address, cnt_clients, initial_model_hash, hash_pubkeys = listen_for_projcet()
    print(f"DEBUG: Project ID from event: {project_id}, Server Hash Keys: {hash_pubkeys}")
    if project_id is None:
        print("Failed to find project information. Exiting.")
        sys.exit(1)

    esk_a = ECC.generate(curve='p256')
    epk_a_bytes = bytes(esk_a.public_key().export_key(format='PEM'), 'utf-8')
    epk_a_hex = epk_a_bytes.hex()
    print(f"DEBUG: Client Ephemeral Key (bytes) length: {len(epk_a_bytes)}. Type: {type(epk_a_bytes)}")

    # ini_score, Tx_r, registered_id_p = register_client(hash_data(epk_a_bytes), project_id)
    # hash_epk = hash_data(epk_a_bytes)        # hex string WITHOUT 0x
    # ini_score, Tx_r, registered_id_p = register_client("0x" + hash_epk, project_id)
    hash_epk = hash_data(epk_a_bytes)          # hex string
    # hash_epk_bytes = bytes.fromhex(hash_epk)   # raw 32-byte value
    ini_score, Tx_r, registered_id_p = register_client(hash_epk, project_id)


    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 65432))
    print("DEBUG: Connected to server off-chain.")

    # Hello
    hello_msg = {"msg_type": "Hello!", "Data": ETH_address  }
    client_socket.send(json.dumps(hello_msg).encode('utf-8'))
    print("DEBUG: entering task loop now...")

    session_id = None
    while True:
        print("DEBUG: Waiting to receive from server...")
        try:
            data = client_socket.recv(4096)
            print(f"DEBUG: Raw data received (len={len(data) if data else 0}): {data[:100]}...")
        except Exception as e:
            print(f"❌ ERROR: socket.recv() failed: {e}")
            time.sleep(1)
            continue

        if not data:
            print("DEBUG: No data received, retrying...")
            time.sleep(0.1)
            continue

        if b"You haven't registered" in data:
            print("DEBUG: Server says 'You haven't registered yet'. Retrying...")
            time.sleep(0.5)
            continue

        try:
            decoded = data.decode('utf-8', errors="ignore")
            print(f"DEBUG: Decoded server message: {decoded}")
            session_id = decoded.split(':')[-1].strip()
            print(f"DEBUG: Parsed Session ID: {session_id} (type={type(session_id)})")
            break
        except Exception as e:
            print(f"❌ ERROR parsing session ID from data: {data}. Exception: {e}")
            sys.exit(1)


    pubkeys_req = {"msg_type": "pubkeys please", "Data": session_id}
    client_socket.send(json.dumps(pubkeys_req).encode('utf-8'))

    data = client_socket.recv(4096)
    received_data = json.loads(data.decode('utf-8'))
    epk_b_pem = bytes.fromhex(received_data['epk_b_pem'])
    kpk_b = bytes.fromhex(received_data['kpk_b'])

    assert hash_data(kpk_b + epk_b_pem) == hash_pubkeys, "on-chain and off-chain pub keys mismatch"
    epk_b = ECC.import_key(epk_b_pem)

    ct, ss_k = ml_kem_768.encrypt(kpk_b)
    print(f"DEBUG: Kyber CT length: {len(ct)}, ss_k length: {len(ss_k)}. Type: {type(ct)}")

    hash_ct_epk_a = hash_data(ct + epk_a_bytes)
    print(f"DEBUG: Initial hash_ct_epk_a (sent to server and used in Tx): {hash_ct_epk_a[:10]}... Type: {type(hash_ct_epk_a)}")

    init_send = {"msg_type": "none", "epk_a_pem": epk_a_hex, "ciphertext": ct.hex()}
    client_socket.send(json.dumps(init_send).encode('utf-8'))

    ss_e = key_agreement(eph_priv=esk_a, eph_pub=epk_b, kdf=kdf)
    SS = ss_k + ss_e
    salt_a = salt_s = b'\0' * 32
    Root_key = HKDF(SS, 32, salt_a, SHA384, 1)
    chain_key = Root_key
    chain_key, Model_key = HKDF(Root_key, 32, salt_s, SHA384, 2)
    print(f"DEBUG: Initial Model Key generated. Length: {len(Model_key)}. Type: {type(Model_key)}")

    Local_model_info = {}
    Local_model = None
    timeout = 240

    while True:
        print("--------------------- NEW ROUND START ---------------------")
        r, Task_id, Hash_model_onchain, hash_keys, project_id_received, server_eth_addr, D_t = listen_for_task(timeout)
        if task_completed(Task_id, project_id_received):
            print(f"Server has already terminated Task id: {Task_id}")
            break
        if Task_id == 0:
            print("No new task received within the timeout. Exiting.")
            break
        print(f"Start Round {r}")

        if hash_keys != 'None':
            print("DEBUG: --- ASYMMETRIC RATCHETING TRIGGERED ---")
            esk_a = ECC.generate(curve='p256')
            epk_a_bytes = bytes(esk_a.public_key().export_key(format='PEM'), 'utf-8')
            epk_a_hex = epk_a_bytes.hex()

            client_socket.send(json.dumps({"msg_type": "update pubkeys", "Data": session_id}).encode('utf-8'))
            data = client_socket.recv(4096)
            received = json.loads(data.decode('utf-8'))
            epk_b_pem = bytes.fromhex(received['epk_b_pem'])
            kpk_b = bytes.fromhex(received['kpk_b'])
            epk_b = ECC.import_key(epk_b_pem)
            assert hash_data(kpk_b + epk_b_pem) == hash_keys, "on-chain and off-chain pub keys mismatch"

            ct, ss_k = ml_kem_768.encrypt(kpk_b)
            hash_ct_epk_a = hash_data(ct + epk_a_bytes)
            print(f"DEBUG: New hash_ct_epk_a: {hash_ct_epk_a[:10]}... Type: {type(hash_ct_epk_a)}")
            client_socket.send(json.dumps({"msg_type": "none", "epk_a_pem": epk_a_hex, "ciphertext": ct.hex()}).encode('utf-8'))

            ss_e = key_agreement(eph_priv=esk_a, eph_pub=epk_b, kdf=kdf)
            SS = ss_k + ss_e
            Root_key = HKDF(SS, 32, salt_a, SHA384, 1)
            chain_key = Root_key
            time.sleep(1)

        print("DEBUG: Requesting Global Model from server.")
        client_socket.send(json.dumps({"msg_type": "Global model please", "Data": session_id}).encode('utf-8'))
        x = receive_Model(client_socket)
        if x is None:
            print("Failed to receive global model package")
            break
        unwrapped_msg = unwrap_files(x)
        global_model_ct = unwrapped_msg.get('global_model.enc')
        print(f"DEBUG: Received Encrypted Model. Ciphertext length: {len(global_model_ct)}")

        dec_wrapfile = AES_decrypt_data(Model_key, global_model_ct)
        unwraped = unwrap_files(dec_wrapfile)

        if r != 1 and HE_algorithm != 'None':
            global_HE_model = unwraped.get('global_HE_model.bin')
            encrypted_weights, metadata = deserialize_data(global_HE_model, HE_config_with_key)
            global_model = HE_decrypt_model(encrypted_weights, Local_model, HE_config_with_key, HE_algorithm, metadata)
            print(f"DEBUG: HE Model decrypted. Global Model object type: {type(global_model)}")
        else:
            global_model = unwraped.get('global_model.pth')
            print(f"DEBUG: Standard Model received. Data type: {type(global_model)}. Length: {len(global_model)}")

        print("Start training...")
        Local_model = train_model.train(global_model, num_epochs, dataset_type)
        print(f"DEBUG: Training complete. Local_model object type: {type(Local_model)}")

        if HE_algorithm != 'None':
            HE_enc_model, metadata = HE_encrypt_model(Local_model, HE_config_with_key, HE_algorithm)
            serialized_model = serialize_data(HE_enc_model, metadata, HE_algorithm)
            local_model_bytes = serialized_model
            local_hash = hash_data(local_model_bytes)
            model_filename = f'local_HE_model_{ETH_address}.bin'
        else:
            local_model_bytes = pickle.dumps(Local_model.state_dict())
            local_hash = hash_data(local_model_bytes)
            model_filename = f'local_model_{ETH_address}.pth'

        Local_model_info['Model hash'] = local_hash
        Local_model_info['Round number'] = r
        Local_model_info['Project id'] = project_id_received
        Local_model_info['Task id'] = Task_id

        Tx_u = update_model_Tx(r, local_hash, hash_ct_epk_a, Task_id, project_id_received)
        print(f"DEBUG: Model Update Tx published: {Tx_u}")

        json_info = json.dumps(Local_model_info, indent=4).encode('utf-8')
        wrapped_model_info = wrapfiles(('Local_model_info.json', json_info), (model_filename, local_model_bytes))

        model_ct = AES_encrypt_data(Model_key, wrapped_model_info)
        signed_ct = sign_data(model_ct, Eth_private_key, w3)
        wraped_msg = wrapfiles(('signature.bin', signed_ct), ('Local_model.enc', model_ct))

        client_socket.send(json.dumps({"msg_type": "local model update", "Data": session_id}).encode('utf-8'))
        send_model(client_socket, wraped_msg)

        project_id_fb, T, score = listen_for_feedback(r, ETH_address)

        if T:
            if project_id_fb == project_id_received:
                print(f'Server terminated the project id {project_id_fb}')
                break

        old_chain_key = chain_key.hex()
        chain_key, Model_key = HKDF(chain_key, 32, salt_s, SHA384, 2)
        salt_s = (bytes_to_long(salt_s) + 1).to_bytes(32, 'big')
        salt_a = (bytes_to_long(salt_a) + 1).to_bytes(32, 'big')
        print(f"DEBUG: Symmetric Ratcheting complete. Old Chain Key: {old_chain_key[:10]}..., New Model Key: {Model_key.hex()[:10]}...")
        print("--------------------- ROUND END ---------------------")

    print("Client finished.")
