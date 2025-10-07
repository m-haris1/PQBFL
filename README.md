Step 1 -> Clone the project
Step 2 ->  Download Ganache https://trufflesuite.com/ganache/
Step 3 -> pip install -r requirements.txt
Step 4 -> 
        Then You have to compile c files
        git clone https://github.com/kpdemetriou/pqcrypto.git
        cd pqcrypto
        sudo python3 compile.py
Step 5 -> Run infrastructure
Here you need to first do
* Compile and Deploy :-
  You must compile and deploy the solidity contract on Ethereum blockchain (gnanache) using ETH address and corresponding Private key
  cd contract
  python compile-deploy.py <ETH address> <Private key> -> This will give a key, store this key somewhere
  Note :- The ETH address and Private Key that you are using to deploy shoud be of server's.

* Now download IPFS:-
    https://dist.ipfs.tech/#kubo (Try to store it in any location in C drive and add the path to environment variabble).
    Now extract the folder and in it , right click on 'ipfs.exe' and select run as admininstration.
    Now, run the following command
      * ipfs init
      * ipfs daemon
* Now open 2 diffrent terminal , one for client and one for server.
    * In client -> python pqbfl.py -m participant -c <CONTRACT_ADDRESS> -ek <ETH_PRIVATE_KEY> -e <NUM_EPOCHS> -d <DATASET> -H <ENCRYPTION_TYPE>
    * In Server -> python pqbfl.py -m server -c <CONTRACT_ADDRESS> -ek <ETH_PRIVATE_KEY> -idor <PROJECT_ID> -r <NUM_ROUNDS> -p <NUM_PARTICIPANTS> -d <DATASET> -H <ENCRYPTION_TYPE>
    Where ETH_PRIVATE_KEY is the private key that you got while deploying your contract in earlier step.
  
    
