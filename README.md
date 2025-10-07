
## Step-by-Step Setup

## 1. Clone the Project

```bash
    git clone <REPO_URL>
    cd pqbfl
```
## 2. Install Ganache (Local Ethereum Blockchain)
Download Ganache from the official Truffle Suite website.

Ganache simulates a local Ethereum blockchain for testing and development

## 3. Install Python Dependencies
``` bash
    pip install -r requirements.txt
```

## 4. Compile Post-Quantum Cryptography Library
``` bash
git clone https://github.com/kpdemetriou/pqcrypto.git
cd pqcrypto
sudo python3 compile.py
```
This compiles the C-based Post-Quantum Cryptographic primitives used in PQBFL.


## 5. Setup and Run Infrastructure

🔨 1. Compile & Deploy the Smart Contract
``` bash
cd contract
python compile-deploy.py <ETH_ADDRESS> <PRIVATE_KEY>
```
<ETH_ADDRESS> and <PRIVATE_KEY> must be taken from Ganache.

On success, this will deploy the contract and output a contract address.

👉 Save this address — it is required for both server and participants.

Use the server’s Ethereum credentials for this step.



📦 2. Install & Run IPFS
Download IPFS (Kubo) from https://dist.ipfs.tech/#kubo
.

Extract the folder (e.g., C:\ipfs).

Add the folder path containing ipfs.exe to your Environment Variables.

Run the following commands:
``` bash
ipfs init
ipfs daemon

```
Keep the IPFS daemon running throughout the experiment.



## 🚀 Running PQBFL
Open two terminals — one for the Server and one for the Participant (Client).

▶️ Server
``` bash
python pqbfl.py -m server \
  -c <CONTRACT_ADDRESS> \
  -ek <ETH_PRIVATE_KEY> \
  -idor <PROJECT_ID> \
  -r <NUM_ROUNDS> \
  -p <NUM_PARTICIPANTS> \
  -d <DATASET> \
  -H <ENCRYPTION_TYPE>
```

🧑‍💻 Participant (Client)
```
python pqbfl.py -m participant \
  -c <CONTRACT_ADDRESS> \
  -ek <ETH_PRIVATE_KEY> \
  -e <NUM_EPOCHS> \
  -d <DATASET> \
  -H <ENCRYPTION_TYPE>
```

## 🧩 Parameters Reference
| Parameter            | Description                                      |
| -------------------- | ------------------------------------------------ |
| `<CONTRACT_ADDRESS>` | Smart contract address returned from deploy step |
| `<ETH_PRIVATE_KEY>`  | Ethereum private key (from Ganache)              |
| `<PROJECT_ID>`       | Unique ID for the FL project                     |
| `<NUM_ROUNDS>`       | Number of federated training rounds              |
| `<NUM_PARTICIPANTS>` | Number of clients expected                       |
| `<NUM_EPOCHS>`       | Local training epochs per client                 |
| `<DATASET>`          | Dataset to be used (`MNIST` or `UCI_HAR`)        |
| `<ENCRYPTION_TYPE>`  | Encryption scheme: `CKKS` or `BFV`               |



## ✅ Example Commands
Server
``` bash 
python pqbfl.py -m server -c 0xAbc123... -ek a1b2c3... -idor proj01 -r 5 -p 3 -d MNIST -H CKKS
```

Client
``` bash
python pqbfl.py -m participant -c 0xAbc123... -ek f6e7d8... -e 10 -d MNIST -H CKKS
```



## 📌 Notes
* Ensure Ganache and IPFS daemon are running before starting.

* Keep all private keys secure.

* The encryption scheme must match between the server and participants.

* PQBFL is modular — you can easily add new datasets or cryptographic schemes.
