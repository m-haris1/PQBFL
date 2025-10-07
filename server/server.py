"""
Created on Tue Jan  2 14:18:41 2024

@author: HIGHer
""" 
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account

from collections import namedtuple
from pqcrypto.kem import ml_kem_768 

from Crypto.Protocol.DH import key_agreement
from Crypto.Protocol.KDF import HKDF
from Crypto.PublicKey import ECC
from Crypto.Hash import SHA384
from Crypto.Util.number import *

import socket, pickle
import tenseal as ts
import os, sys, time,json

import aggregate
from threading import *
from queue import Queue
import torch


from simple_cnn_config import SimpleCNN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *


# Global Variables (Initialized in __main__)
w3 = None
contract_address = None
contract_abi = None
Eth_address = None
Eth_private_key = None
ecdh = None
kyber = None
salt_a = None
salt_s = None
clients_dict = {}
wraped_global_model = None
model_info = {}


def generate_keys():
    KeyPair = namedtuple('KeyPair', ['pk', 'sk'])
    ecdh_priv = ECC.generate(curve='p256')  # ECDH private key
    ecdh_pub = bytes(ecdh_priv.public_key().export_key(format='PEM'), 'utf-8')  # ECDH public key
    kyber_pub, kyebr_priv = ml_kem_768.generate_keypair()  # Kyber key pair
    ecdh_keys = KeyPair(pk =ecdh_pub, sk =ecdh_priv)
    kyber_keys = KeyPair(pk=kyber_pub, sk=kyebr_priv)
    print(f"DEBUG SERVER: New ECDH/Kyber keys generated.") # DEBUG
    print(f"DEBUG SERVER: ECDH Pub (PEM bytes) length: {len(ecdh_pub)}") # DEBUG
    return ecdh_keys, kyber_keys

def register_project(project_id, cnt_clients_req, hash_init_model, hash_keys):
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    if not contract.functions.isProjectTerminated(project_id).call():
        print(f"DEBUG SERVER: Registering project {project_id} on-chain. Model Hash: {hash_init_model[:10]}...") # DEBUG
        for attempt in range(3):  # Retry logic
            try:
                nonce = w3.eth.get_transaction_count(Eth_address, 'pending')
                transaction = contract.functions.registerProject(
                    project_id, cnt_clients_req, hash_init_model, hash_keys
                ).build_transaction({
                    'from': Eth_address,
                    'gas': 2000000,
                    'gasPrice': w3.to_wei('50', 'gwei'),
                    'nonce': nonce,
                })
                signed_transaction = w3.eth.account.sign_transaction(transaction, Eth_private_key)
                tx_sent = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
                receipt = w3.eth.wait_for_transaction_receipt(tx_sent)
                gas_used=receipt['gasUsed']
                tx_registration = receipt['transactionHash'].hex()
                print(f'Project Registeration on contract:')
                print(f'    Tx_hash: {tx_registration}')
                print(f'    Gas: {gas_used} Wei')
                print(f'    Project ID: {project_id}')
                print(f'    required client count: {cnt_clients_req}') 
                print(f'    Initial model hash: {hash_init_model}')
                print(f'    Pubic keys hash: {hash_keys}')
                print('-'*75)
                return tx_registration
            except ValueError as e:
                print(f"Error: {e}. Retrying transaction...")
                time.sleep(2)
        raise Exception("Transaction failed after retries.")
    else:
        print(f"Project {project_id} is already completed.")
        sys.exit()


def wait_for_clients(event_queue, stop_event, poll_interval=2):
    print('DEBUG SERVER: Event listener thread started (ClientRegistered events)') # DEBUG
    if geth_poa_middleware not in w3.middleware_onion: 
        # Add PoA middleware for Ganache (if needed)
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    # Create an instance of the contract
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    last_processed_block = w3.eth.block_number  # Keep track of the last processed block
    while not stop_event.is_set():  # Check the stop_event to terminate the loop
        try:
            current_block = w3.eth.block_number  # Get current block number
            if current_block > last_processed_block:
                # Create filter for the specific block range
                event_filter = contract.events.ClientRegistered.create_filter(
                    fromBlock=last_processed_block + 1,
                    toBlock=current_block
                )
                events = event_filter.get_all_entries()  # Get events 
                for event in events:  # Process events
                    event_queue.put(event)
                    print(f"DEBUG SERVER: Client registration event caught at block {event['blockNumber']}: {event['args']['clientAddress']}") # DEBUG
                last_processed_block = current_block  # Update last processed block
                w3.eth.uninstall_filter(event_filter.filter_id)  # Clean up filter

            time.sleep(poll_interval)  # Wait before next poll
        except Exception as e:
            print(f"DEBUG SERVER: Error in registration event listener: {str(e)}") # DEBUG
            time.sleep(poll_interval)  # Wait before retrying


def finish_tash(task_id, project_id):
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    print(f"DEBUG SERVER: Finishing Task {task_id}") # DEBUG
    # ... [function body remains the same] ...
    nonce = w3.eth.get_transaction_count(Eth_address)
    
    # Build the transaction with task_id and project_id
    transaction = contract.functions.finishTask(task_id, project_id).build_transaction({
        'from': Eth_address,
        'gas': 2000000,  # Adjust the gas limit based on your contract's needs
        'gasPrice': w3.to_wei('50', 'gwei'),
        'nonce': nonce,
    })
    # Sign and send the transaction
    signed_transaction = w3.eth.account.sign_transaction(transaction, Eth_private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
    # Wait for the receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    gas_used = receipt['gasUsed']
    tx_publish = receipt['transactionHash'].hex()
    print(f'Task terminated:')
    print(f'    Tx_hash: {tx_publish}')
    print(f'    Gas: {gas_used} Wei')
    print(f'    Task ID: {task_id}')
    print(f'    Project ID: {project_id}')
    print('-' * 75)

def finish_project(project_id):
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    print(f"DEBUG SERVER: Finishing Project {project_id}") # DEBUG
    # ... [function body remains the same] ...
    nonce = w3.eth.get_transaction_count(Eth_address)
    # Build the transaction with project_id
    transaction = contract.functions.finishProject(project_id).build_transaction({
        'from': Eth_address,
        'gas': 2000000,  # Adjust the gas limit based on your contract's needs
        'gasPrice': w3.to_wei('50', 'gwei'),
        'nonce': nonce,
    })
    # Sign and send the transaction
    signed_transaction = w3.eth.account.sign_transaction(transaction, Eth_private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
    # Wait for the receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    gas_used = receipt['gasUsed']
    tx_publish = receipt['transactionHash'].hex()
    print(f'Project terminated:')
    print(f'    Tx_hash: {tx_publish}')
    print(f'    Gas: {gas_used} Wei')
    print(f'    Project ID: {project_id}')
    print('-' * 75)


def publish_task(r, Hash_model, hash_keys, Task_id, project_id, D_t):
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    print(f"DEBUG SERVER: Publishing Task R:{r}. Model Hash: {Hash_model[:10]}..., Keys Hash: {hash_keys}") # DEBUG
    # ... [function body remains the same] ...
    nonce = w3.eth.get_transaction_count(Eth_address)
    transaction = contract.functions.publishTask(r,Hash_model, hash_keys, Task_id, project_id, D_t).build_transaction({
        'from': Eth_address,
        'gas': 2000000,
        'gasPrice': w3.to_wei('50', 'gwei'),
        'nonce': nonce,
    })
    signed_tx = w3.eth.account.sign_transaction(transaction, Eth_private_key)
    tx_sent = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_sent)
    gas_used=receipt['gasUsed']
    tx_publish = receipt['transactionHash'].hex()
    print('')
    print(f'Task published round {r}:')
    print(f'    Tx_hash: {tx_publish}')
    print(f'    Gas: {gas_used} Wei')
    print(f'    Task ID: {Task_id}')
    print('-'*75)
    return tx_publish


def listen_for_updates(event_filter, event_queue):
    print('DEBUG SERVER: ModelUpdate listener thread started') # DEBUG
    # Add PoA middleware for Ganache (if needed)
    if geth_poa_middleware not in w3.middleware_onion:
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    # Create an instance of the contract with the ABI and address
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    event_filter = contract.events.ModelUpdated.create_filter(fromBlock="latest")           # Get events since the last checked block
    # Loop to listen for events
    while True:
        events = event_filter.get_new_entries()
        if events:
            for event in events:
                event_queue.put(event)
                print(f"DEBUG SERVER: ModelUpdate event caught for {event['args']['clientAddress']}") # DEBUG
        time.sleep(1)


def feedback_TX(r, task_id, project_id, client_address, feedback_score, T):
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    print(f"DEBUG SERVER: Sending Feedback (R:{r}) for {client_address}. Score: {feedback_score}") # DEBUG
    # ... [function body remains the same] ...
    for attempt in range(3):  # Retry mechanism
        try:
            nonce = w3.eth.get_transaction_count(Eth_address, 'pending') # Fetch the current pending nonce
            transaction = contract.functions.provideFeedback(r, task_id, project_id, client_address, feedback_score, T   # Add the feedback score
            ).build_transaction({
                'from': Eth_address,
                'gas': 2000000,
                'gasPrice': w3.to_wei('50', 'gwei'),
                'nonce': nonce,
            })
            # Sign the transaction
            signed_transaction = w3.eth.account.sign_transaction(transaction, Eth_private_key)
            tx_sent = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_sent)
            gas_used = receipt['gasUsed']
            tx_feedback = receipt['transactionHash'].hex()
            # Print transaction details
            print(f'Feedback:')
            print(f'      Client address: {client_address}')
            print(f'      Tx_hash: {tx_feedback}')
            print(f'      Gas: {gas_used} Wei')
            print(f'      Task ID: {task_id}')
            print(f'      Score: {feedback_score}')
            print('-'*75)
            return tx_feedback
        except ValueError as e:
            print(f"Transaction failed: {e}. Retrying...")
            time.sleep(2)  # Wait before retrying
    raise Exception("Feedback transaction failed after retries.")

def analyze_model (Local_model,Task_id,project_id_update):
    res=True
    Feedback_score=1
    return res, Feedback_score


def establish_root_key(client_socket,clients_dict,ecdh,kyber,salt_a,session_id):
            matching_addr = [address for address, details in clients_dict.items() if details.get("Session ID") == session_id] # find eth addr based Session ID
            if not matching_addr:
                print('DEBUG SERVER: Client session ID not found in registered clients.') # DEBUG
                client_socket.close()
                return None
            
            client_addr = matching_addr[0]
            print(f"DEBUG SERVER: Key exchange started for {client_addr}") # DEBUG

            msg_keys={'epk_b_pem':(ecdh.pk).hex(), 'kpk_b':(kyber.pk).hex()}
            client_socket.sendall(json.dumps(msg_keys).encode('utf-8'))
            
            data = client_socket.recv(4096).decode('utf-8')  # Receive epk_a_pem and ct from client via off-chain
            if data is None:
                print(f"DEBUG SERVER: Failed to receive client keys/ciphertext.") # DEBUG
                client_socket.close()
                return None
                
            received_data= json.loads(data) # Process the received Json data construct root, chain and model keys
            epk_a_pem = bytes.fromhex(received_data['epk_a_pem'])
            ct = bytes.fromhex(received_data['ciphertext']) 
            print(f"DEBUG SERVER: Received ct len: {len(ct)}, epk_a_pem len: {len(epk_a_pem)}") # DEBUG

            epk_a = ECC.import_key(epk_a_pem)
            ss_e = key_agreement(eph_priv=ecdh.sk, eph_pub=epk_a, kdf=kdf)    # ECDH shared secret 
            ss_k = ml_kem_768.decrypt(kyber.sk, ct)
            SS = ss_k + ss_e           # (ss_k||ss_e) construnct general shared secret 
            Root_key= HKDF(SS, 32, salt_a, SHA384, 1)     #  RK_1 <-- SS + Salt_a  
            
            clients_dict[client_addr]['Hash_ct_epk_a']=hash_data(ct +epk_a_pem) 
            clients_dict[client_addr]['Root key']  = Root_key.hex()
            print(f"DEBUG SERVER: Root Key established for {client_addr}. Hash(ct||epk): {clients_dict[client_addr]['Hash_ct_epk_a'][:10]}...") # DEBUG
            return Root_key


def offchain_listener(server_socket):    #Listen for incoming off-chain client connections.
    print('DEBUG SERVER: Off-chain listener thread started') # DEBUG
    while True:
        try:
            client_socket, client_address = server_socket.accept()
            print(f"DEBUG SERVER: New off-chain connection from {client_address}")
            client_thread = Thread(
                target=handle_offchain_client,
                args=(client_socket,),
                daemon=True
            )
            client_thread.start()
        except Exception as e:
            print(f"DEBUG SERVER: Error accepting connection: {e}") # DEBUG
            time.sleep(1)

    
def handle_offchain_client(client_socket):    # Handle individual client communication on off-chain.
    global salt_a, salt_s, ecdh, kyber, wraped_global_model , model_info
    
    # Placeholder for the client's ETH address for this thread's scope
    eth_address = None 
    
    # NEW: Counter for initial registration check retries
    max_registration_attempts = 10
    
    while True: 
        try:
            data_raw = client_socket.recv(4096)
            if not data_raw:
                print(f"DEBUG SERVER: Client {eth_address or 'Unknown'} disconnected.")
                break
                
            recv_msg = json.loads(data_raw.decode('utf-8'))
            msg_type = recv_msg.get("msg_type")
            
            print(f"DEBUG SERVER: Received message type: {msg_type}") # DEBUG

            if msg_type == 'Hello!':
                eth_address = recv_msg["Data"]
                
                # --- FIX START: Retry logic for registration check to fix race condition ---
                is_registered = False
                registration_check_attempts = 0
                while not is_registered and registration_check_attempts < max_registration_attempts:
                    if eth_address in clients_dict:
                        is_registered = True
                        break
                    
                    print(f"DEBUG SERVER: Client {eth_address} not in clients_dict yet. Retrying in 0.5s... (Attempt {registration_check_attempts + 1}/{max_registration_attempts})")
                    time.sleep(0.5) # Give the main thread time to process the on-chain event
                    registration_check_attempts += 1
                
                if is_registered:
                # --- FIX END ---
                    session_id = clients_dict[eth_address]['Session ID']
                    client_socket.send(('Session ID:' + str(session_id)).encode('utf-8'))
                    print(f"DEBUG SERVER: Sent Session ID {session_id} to {eth_address}") # DEBUG
                    
                    data_raw = client_socket.recv(4096) # Wait for pubkeys request
                    if not data_raw: continue
                    recv_msg = json.loads(data_raw.decode('utf-8'))
                    
                    if recv_msg.get("msg_type") == "pubkeys please":
                        session_id = int(recv_msg["Data"])
                        root_key = establish_root_key(client_socket, clients_dict, ecdh, kyber, salt_a, session_id)
                        if root_key:
                            # Re-derive eth_address after successful key establishment if needed, but it should be known here.
                            chain_key, model_key = HKDF(root_key, 32, salt_s, SHA384, 2)
                            clients_dict[eth_address]['Model key'] = model_key.hex()
                            clients_dict[eth_address]['Chain key'] = chain_key.hex()
                            print(f"DEBUG SERVER: Initial keys derived for {eth_address}. Model Key: {model_key.hex()[:10]}...") # DEBUG
                    
                else:
                    print(f"DEBUG SERVER: Client {eth_address} failed registration check after {max_registration_attempts} attempts. Sending rejection.")
                    client_socket.send("You haven't registered for the project on blockchain.".encode('utf-8'))
            
            elif msg_type == 'update pubkeys':
                session_id = int(recv_msg["Data"])
                root_key = establish_root_key(client_socket, clients_dict, ecdh, kyber, salt_a, session_id)
                if root_key:
                    # Update keys only if establishment was successful (symmetric ratcheting will happen in the main thread)
                    print(f"DEBUG SERVER: Asymmetric Ratcheting successful for session {session_id}") # DEBUG
                    
            elif msg_type == 'Global model please':
                session_id = int(recv_msg["Data"])
                eth_addr = [addr for addr, details in clients_dict.items() if details["Session ID"] == session_id][0]
                Client_Model_key=bytes.fromhex(clients_dict[eth_addr]['Model key'])
                
                # Model Encryption and Signing
                model_ct=AES_encrypt_data(Client_Model_key, wraped_global_model)
                signed_ct=sign_data(model_ct, Eth_private_key, w3)
                global_model_msg=wrapfiles(('signature.bin',signed_ct), ('global_model.enc',model_ct))
                
                send_model(client_socket, global_model_msg)
                print(f"DEBUG SERVER: Sent encrypted global model to {eth_addr}") # DEBUG
                
            elif msg_type == 'local model update':
                session_id = int(recv_msg["Data"])
                Recieved_model=receive_Model(client_socket)
                eth_addr = [addr for addr, details in clients_dict.items() if details["Session ID"] == session_id][0]
                
                model_info[eth_addr]={'model_data': Recieved_model}
                print(f"DEBUG SERVER: Received local model from {eth_addr}. Data stored in model_info.") # DEBUG
                
            else:
                print(f"DEBUG SERVER: Invalid message type received: {msg_type}") # DEBUG
                
        except json.JSONDecodeError:
            print(f"DEBUG SERVER: JSON Decode Error from client.") # DEBUG
            break
        except Exception as e:
            print(f"DEBUG SERVER: Unexpected Error handling client {eth_address}: {e}") # DEBUG
            break
            
    client_socket.close()


if __name__ == "__main__":
    print("--- SERVER START ---") # DEBUG: Execution starts here
    try:  # Connect to the local Ganache blockchain
        onchain_addr = "http://127.0.0.1:7545"   # (on-chain) address anache_url
        w3 = Web3(Web3.HTTPProvider(onchain_addr))
        print("Server connected to blockchain (Ganache) successfully\n")
    except Exception as e:
        print("An exception occurred in connecting to blockchain (Ganache) or offchain:", e)
        exit()
        
    # --- Network Setup ---
    offcahin_addr = ('localhost', 65432)          # server (off-chain) address
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    server_socket.bind(offcahin_addr)
    server_socket.listen()
    print(f"DEBUG SERVER: Off-chain socket bound to {offcahin_addr}") # DEBUG


    # --- CLI Argument Parsing ---
    Eth_private_key=sys.argv[1]    
    contract_address = sys.argv[2]
    project_id=int(sys.argv[3])
    round_count=int(sys.argv[4])
    client_req=int(sys.argv[5])     
    Dataset_type=sys.argv[6]    
    HE_algorithm=sys.argv[7]    

    account = Account.from_key(Eth_private_key)
    Eth_address = account.address   # load the Ethereum account
    print(f"DEBUG SERVER: Server ETH Address: {Eth_address}") # DEBUG

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)  # Get the path to the parent directory of the script
    with open(main_dir+"/contract/contract-abi.json", "r") as abi_file:
        contract_abi = json.load(abi_file)     # Load ABI from file
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)  # Create a contract instance

    # --- Load HE Keys ---
    if HE_algorithm=='CKKS':
        with open(main_dir + f'/server/keys/CKKS_without_priv_key.pkl', "rb") as f:
            serialized_without_key = pickle.load(f)
        HE_config_without_key = ts.context_from(serialized_without_key)
    elif HE_algorithm=='BFV':
        with open(main_dir + f'/server/keys/BFV_without_priv_key.pkl', "rb") as f:
            serialized_without_key = pickle.load(f)
        HE_config_without_key = ts.context_from(serialized_without_key)
    print(f"DEBUG SERVER: HE Config loaded for algorithm: {HE_algorithm}") # DEBUG
    
    # --- Initial Key Generation & Project Registration ---
    ecdh, kyber = generate_keys()    # remember the ecdh public key is in pem format
    hash_pubkeys=hash_data(kyber.pk+ecdh.pk) # string/hex hash
    print(f"DEBUG SERVER: Initial Pub Keys Hash: {hash_pubkeys[:10]}...") # DEBUG
#-------------------------------------------------
    Init_Global_model = SimpleCNN(Dataset_type)
    Init_Global_model = pickle.dumps(Init_Global_model.state_dict()) # bytes
    Hash_model = hash_data(Init_Global_model) # string/hex hash
    print(f"DEBUG SERVER: Initial Model Hash: {Hash_model[:10]}...") # DEBUG
    Tx_r =register_project(project_id, client_req, Hash_model, hash_pubkeys)

    # --- Client Registration & Setup ---
    registered_cnt=0
    salt_a = salt_s = b'\0'*32    # asymmetric (salt_a) and symmetric (salt_s) salt (bytes)
    registration_queue = Queue()
    stop_event = Event()
    print(f"DEBUG SERVER: Starting registration listeners...") # DEBUG

    Thread(target=wait_for_clients, args=(registration_queue, stop_event), daemon=True).start()
    Thread(target=offchain_listener, args=(server_socket,), daemon=True).start()

    while registered_cnt < client_req:
        try:
            event = registration_queue.get(timeout=30)  # Wait for ClientRegistered event
            eth_address = event['args']['clientAddress']                
            session_id = registered_cnt + 1
            clients_dict[eth_address] = {
                'Session ID': session_id,
                'score': event['args']['initialScore'],
                'hash_epk': event['args']['hash_PubKeys'],
                'registration_tx': event['transactionHash'].hex(),
                'block_number': event['blockNumber']
            }
            registered_cnt += 1
            print(f"DEBUG SERVER: Client {registered_cnt}/{client_req} registered: {eth_address}") # DEBUG
        except Exception as e:
            print(f"DEBUG SERVER: Timeout waiting for client registration or error: {str(e)}") # DEBUG
            if registered_cnt == 0:
                print("Exiting due to registration failure.")
                sys.exit(1)
            break

    print("All clients registered or timeout reached.")
    stop_event.set()   # Signal the on-chain listener thread to stop
    print('-'*75)
    
    Global_Model=Init_Global_model # bytes
    Models=[]
    task_info= {}
    ratchet_renge=2
    accuracy_list=[]
   
    # --- Main FL Rounds Loop ---
    for r in range(1,round_count+1):    
        print(f"\n====================== ROUND {r} START ======================") # DEBUG
        Task_id=int(str(project_id)+str(r))
        task_info['Round number'] = r
        task_info['Model hash'] = Hash_model # Hash of the previous round's global model
        task_info['Project id'] = project_id
        task_info['Task id'] = Task_id
        task_info['Deadline Task'] = int(time.time()) + 100000

    # Publish Task
        hash_pubkeys='None'
        if r%ratchet_renge==0:     # Assymmetric ratcheting condition
            ecdh, kyber = generate_keys() 
            hash_pubkeys=hash_data(kyber.pk+ecdh.pk)
            Tx_p = publish_task(r, Hash_model, hash_pubkeys, Task_id, project_id, task_info['Deadline Task'])       
            task_info['Publish Tx'] = Tx_p
            print(f"DEBUG SERVER: Asymmetric ratcheting triggered. New key hash: {hash_pubkeys[:10]}...") # DEBUG
        else: 
            Tx_p = publish_task(r, Hash_model, hash_pubkeys, Task_id, project_id, task_info['Deadline Task'])    
            task_info['Publish Tx'] = Tx_p 
            print("DEBUG SERVER: Symmetric ratcheting only for this round.") # DEBUG

        json_task_info = json.dumps(task_info, indent=4)
        
        # Wrap the global model (bytes) for encryption
        if r!=1 and HE_algorithm!='None':
            wraped_global_model=wrapfiles(('task_info.json',json_task_info.encode()), ('global_HE_model.bin',aggregated_HE_model))  # aggregated_HE_model is bytes
        else:
            wraped_global_model=wrapfiles(('task_info.json',json_task_info.encode()), ('global_model.pth',Global_Model))  # Global_Model is bytes
        
        print(f"DEBUG SERVER: Global Model wrapped. Size: {len(wraped_global_model)} bytes.") # DEBUG


        print(f"Start Round {r}: Waiting for local model updates...\n"+'='*20)        
        event_queue = Queue()
        block_filter =  w3 .eth.filter('latest')
        worker = Thread(target=listen_for_updates, args=(block_filter,event_queue), daemon=True)
        worker.start()
        client_addrs=[]
        update_dict={}
        cnt_models=0
        T= False  # Termination flag


        while True:  # Wait for update model
            if not event_queue.empty():
                print(f'Received {cnt_models+1} Local model update Tx:')
                event = event_queue.get()
                r_update = event['args']['round']
                Task_id_update = event['args']['taskId']
                tx_u = event['transactionHash'].hex()
                project_id_update= event['args']['project_id']
                client_addr = event['args']['clientAddress']
                Hash_local_model = event['args']['HashModel']
                Hash_ct_epk_a = event['args']['hash_ct_epk']
                
                print(f"DEBUG SERVER: Processing Tx from {client_addr}. Hash: {Hash_local_model[:10]}...") # DEBUG

                if r_update==r and Task_id_update==Task_id and project_id_update==project_id:
                    update_dict[client_addr]= {'round': r_update, 'Task id':Task_id_update , 
                                                   'Tx_u': tx_u, 'Project id':project_id_update, 
                                                   'Local model hash':Hash_local_model} 
                else:
                    print('DEBUG SERVER: Model update info not related to current round/project. Skipping.') # DEBUG
                    continue
                    
                print(json.dumps(update_dict[client_addr], indent=4))
                client_addrs.append(client_addr)
                
            # recieved model info and verification
                time.sleep(2)
                if HE_algorithm!='None':
                    time.sleep(9)
                
                # Retrieve model data received via off-chain listener
                Recieved_model=model_info.get(client_addr, {}).get('model_data')
                if Recieved_model is None:
                    print(f"DEBUG SERVER: ERROR: Model data missing for {client_addr} in model_info.") # DEBUG
                    continue # Skip this client

                unwrapped_msg=unwrap_files(Recieved_model)
                signature=unwrapped_msg['signature.bin']
                local_model_ct=unwrapped_msg['Local_model.enc']
                
                print(f"DEBUG SERVER: Verifying signature using Tx: {tx_u}") # DEBUG
                verify_sign(signature, local_model_ct, pubKey_from_tx(tx_u,w3))
                
                Model_key=bytes.fromhex(clients_dict[client_addr]['Model key'])
                dec_wrapfile=AES_decrypt_data(Model_key,local_model_ct)
                unwraped=unwrap_files(dec_wrapfile)
                Local_model_info =unwraped['Local_model_info.json']
                
                print(f"DEBUG SERVER: Decrypted local model info: {Local_model_info.decode('utf-8')[:100]}...") # DEBUG

                if Hash_ct_epk_a!='None':  # Check on-chain and off-chain hash(ct||epk)
                    assert clients_dict[client_addr]['Hash_ct_epk_a'] == Hash_ct_epk_a  , " off- and on-chain keys not match :("   
                    print("DEBUG SERVER: Asymmetric Key Hash Check PASSED.") # DEBUG

                if HE_algorithm=='None':
                    Local_model=unwraped[f'local_model_{client_addr}.pth']
                    assert Hash_local_model==hash_data(Local_model), f" on-chain and off-chain Hash of local model {client_addr} are not match :("    # Hash check on model bytes
                    Res, Feedback_score = analyze_model(Local_model,Task_id_update,project_id_update)
                    if Res:
                        cnt_models+=1  # save local model for using in aggregation
                        open(main_dir + f"/server/files/local models/local_model_{client_addr}.pth",'wb').write(Local_model)  
                        Tx_f=feedback_TX (r,Task_id, project_id,client_addr, Feedback_score, T)    
                else:
                    local_HE_model=unwraped[f'local_HE_model_{client_addr}.bin']
                    assert Hash_local_model==hash_data(local_HE_model), f" on-chain and off-chain Hash of local model {client_addr} are not match :("    # Hash check on model bytes
                    cnt_models+=1  # save local model for using in aggregation
                    open(main_dir + f"/server/files/local models/local_HE_model_{client_addr}.bin",'wb').write(local_HE_model)
                    Feedback_score=0  
                    Tx_f=feedback_TX (r,Task_id, project_id, client_addr, Feedback_score, T) 
                
                print(f"DEBUG SERVER: Successfully processed local model {cnt_models}/{registered_cnt}.") # DEBUG

                if cnt_models==registered_cnt:
                    print("DEBUG SERVER: All required models received. Starting aggregation.") # DEBUG
                    if HE_algorithm=='None':
                        normal_aggregated,accuracy=aggregate.aggregate_models(client_addrs,HE_algorithm,Dataset_type)
                        Global_Model=pickle.dumps(normal_aggregated.state_dict())
                        Hash_model = hash_data(Global_Model)
                        torch.save(normal_aggregated.state_dict(), main_dir+'/server/files/global_model.pth')
                        print('*'*40+f'\nAccuracy global model in round {r}: {accuracy:.5f}\n'+'*'*40)
                    else:
                        aggregated_HE_model= aggregate.aggregate_models(client_addrs,HE_algorithm,Dataset_type) 
                        Hash_model = hash_data(aggregated_HE_model)
                        open(main_dir + f"/server/files/global_HE_model.bin",'wb').write(aggregated_HE_model) 
                    print(f"DEBUG SERVER: Aggregation complete. New Global Hash: {Hash_model[:10]}...") # DEBUG
                    break
            else:
                time.sleep(2)

        # Symmetric ratcheting of model-key for each client at the end of round
        print("\nDEBUG SERVER: Starting symmetric ratcheting for all clients.") # DEBUG
        for addr in clients_dict:   
            chain_key=bytes.fromhex(clients_dict[addr]['Chain key'])  # get previous chain key
            if  r%ratchet_renge==0: # Check if asymmetric ratcheting happened this round
                Root_key=bytes.fromhex(clients_dict[addr]['Root key'])
                chain_key, Model_key = HKDF(Root_key, 32, salt_s, SHA384, 2)
                print(f"DEBUG SERVER: Client {addr[:6]}...: Used Root Key for symmetric derivation.") # DEBUG
            else:
                chain_key, Model_key = HKDF(chain_key, 32, salt_s, SHA384, 2)
                print(f"DEBUG SERVER: Client {addr[:6]}...: Used Chain Key for symmetric derivation.") # DEBUG

            clients_dict[addr]['Model key'] = Model_key.hex()
            clients_dict[addr]['Chain key'] = chain_key.hex()  # update keys of dict of clients
            
        salt_s=(bytes_to_long(salt_s)+1).to_bytes(32, byteorder='big')        #  salt_s  increment(update salt) 
        salt_a=(bytes_to_long(salt_a)+1).to_bytes(32, byteorder='big')        #  salt_a  increment(update salt)           
        print(f"DEBUG SERVER: Ratcheting finished. New salt_s: {salt_s.hex()[:10]}...") # DEBUG
        
finish_tash(Task_id,project_id) # transaction termination for recording on   blockchain
finish_project(project_id)
print("--- SERVER FINISHED ---") # DEBUG: Execution ends here