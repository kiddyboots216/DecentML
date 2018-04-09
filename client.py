import logging
import pickle
import shutil
import time

import numpy as np
import tensorflow as tf

from models.perceptron import Perceptron
from models.cnn import CNN
from models.lstm import LSTM

from ethereum-utils import is_address
from web3.auto import w3
from web3 import Web3, HTTPProvider


web3 = Web3(HTTPProvider('http://localhost:8545'))

logging.basicConfig(level=logging.DEBUG,
                    format='[Client] %(asctime)s %(levelname)s %(message)s')

contractAbi = [{'constant': False,
  'inputs': [],
  'name': 'inverseScale',
  'outputs': [{'name': '', 'type': 'bool'}],
  'payable': False,
  'stateMutability': 'nonpayable',
  'type': 'function'},
 {'constant': True,
  'inputs': [{'name': '', 'type': 'uint256'}],
  'name': 'keyList',
  'outputs': [{'name': '', 'type': 'int256'}],
  'payable': False,
  'stateMutability': 'view',
  'type': 'function'},
 {'constant': True,
  'inputs': [],
  'name': 'vectorLength',
  'outputs': [{'name': '', 'type': 'uint256'}],
  'payable': False,
  'stateMutability': 'view',
  'type': 'function'},
 {'constant': True,
  'inputs': [],
  'name': 'totalNumData',
  'outputs': [{'name': '', 'type': 'int256'}],
  'payable': False,
  'stateMutability': 'view',
  'type': 'function'},
 {'constant': False,
  'inputs': [{'name': 'update', 'type': 'int256[]'},
             {'name': 'key', 'type': 'int256'},
             {'name': 'numData', 'type': 'int256'}],
  'name': 'sendResponse',
  'outputs': [{'name': '', 'type': 'int256[]'}],
  'payable': False,
  'stateMutability': 'nonpayable',
  'type': 'function'},
 {'constant': True,
  'inputs': [{'name': '', 'type': 'int256'}, {'name': '', 'type': 'uint256'}],
  'name': 'weights',
  'outputs': [{'name': '', 'type': 'int256'}],
  'payable': False,
  'stateMutability': 'view',
  'type': 'function'},
 {'constant': True,
  'inputs': [],
  'name': 'numberOfResponses',
  'outputs': [{'name': '', 'type': 'int256'}],
  'payable': False,
  'stateMutability': 'view',
  'type': 'function'},
 {'constant': True,
  'inputs': [],
  'name': 'moreThanOne',
  'outputs': [{'name': '', 'type': 'bool'}],
  'payable': False,
  'stateMutability': 'view',
  'type': 'function'},
 {'inputs': [{'name': '_vectorLength', 'type': 'uint256'},
             {'name': '_keyList', 'type': 'int256[]'}],
  'payable': False,
  'stateMutability': 'nonpayable',
  'type': 'constructor'},
 {'anonymous': False,
  'inputs': [{'indexed': False, 'name': 'client', 'type': 'address'}],
  'name': 'ClientSelected',
  'type': 'event'},
 {'anonymous': False,
  'inputs': [{'indexed': False, 'name': 'n', 'type': 'int256'}],
  'name': 'ResponseReceived',
  'type': 'event'}]

class Client(object):
    def __init__(self, iden, X_train, y_train, masterAddress = None, clientAddress=None):
        self.iden = iden
        self.X_train = X_train
        self.y_train = y_train
        if masterAddress:
            assert(is_address(masterAddress))
            self.masterContract = web3.eth.contract(
                address=masterAddress,
                abi=contractAbi)
        else:
            #TODO: Figure out what to do in event that a master address is not supplied
            self.masterContract = None
        if address:
            assert(is_address(clientAddress))
            self.clientAddress = clientAddress
        else:
            #TODO: Initialize client 'container' address if it wasn't assigned one
            PASSPHRASE = 'panda'
            self.clientAddress = acct = web3.personal.newAccount(PASSPHRASE)
            assert(is_address(self.clientAddress)) 
        self.buyerContract = None          

    def setup_model(self, model_type):
        self.model_type = model_type
        if model_type == "perceptron":
            self.model = Perceptron()
        elif model_type == "cnn":
            #TODO: Support CNN
            self.model = CNN()
        elif model_type == "lstm":
            #TODO: Support LSTM
            self.model = LSTM()
        else:
            raise ValueError("Model {0} not supported.".format(model_type))

    # def setup_training(self, batch_size, epochs, learning_rate):
    #     self.batch_size = self.X_train.shape[0] if batch_size == -1 else batch_size
    #     self.epochs = epochs
    #     self.params = {'learning_rate': learning_rate}

    def train(self, weights, config):
        #TODO: Make train() only need to take in the config argument ONCE
        logging.info('Training just started.')
        assert weights != None, 'weights must not be None.'
        batch_size = self.X_train.shape[0] if config["batch_size"] == -1 \
            else config["batch_size"]
        epochs = config["epochs"]
        learning_rate = config["learning_rate"]
        params = {'new_weights': weights, 'learning_rate': learning_rate}

        classifier = tf.estimator.Estimator(
            model_fn=self.model.get_model,
            model_dir=self.get_checkpoints_folder(),
            params = params
        )
        # tensors_to_log = {"probabilities": "softmax_tensor"}
        # logging_hook = tf.train.LoggingTensorHook(
        #     tensors=tensors_to_log, every_n_iter=50)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.X_train},
            y=self.y_train,
            batch_size=batch_size,
            num_epochs=epochs,
            shuffle=True
        )
        classifier.train(
            input_fn=train_input_fn,
            # steps=1
            # hooks=[logging_hook]
        )
        logging.info('Training complete.')
        new_weights = self.model.get_weights(self.get_latest_checkpoint())
        shutil.rmtree("./checkpoints-{0}/".format(self.iden))
        update, num_data = new_weights, self.X_train[0].size
        update = self.model.scale_weights(update, num_data)
        return update, num_data

    # def send_weights(self, train_arr, train_key):
    #     #this should call the contract.sendResponse() with the first argument train() as the input
    #     tx_hash = contract_obj.functions.sendResponse(train_arr, train_key, len(train_arr)).transact(
    #         {'from': clientAddress})
    #     tx_receipt = web3.eth.getTransactionReceipt(tx_hash)
    #     log = contract_obj.events.ResponseReceived().processReceipt(tx_receipt)
    #     return log[0]

    def handle_event(self, event):
        print(event)

    def handle_clientSelected_event(self, event):
        #TODO: get weights and config from event
        weights = event.get_weights()
        config = event.get_config()
        update, num_data = train(weights, config)
        tx_hash = contract_obj.functions.receiveResponse(update, num_data).transact(
            {'from': clientAddress})
        tx_receipt = web3.eth.getTransactionReceipt(tx_hash)
        log = contract_obj.events.ResponseReceived().processReceipt(tx_receipt)
        return log[0]

    def handle_queryCreated_event(self, event):
        #TODO: get address of the buyer from the master contract
        address = event.get_address()
        assert(is_address(address))
        self.buyerAddress = adress
        start_listening(buyerAddress = self.buyerAddress)

    def start_listening(self, buyerAddress = None, 
        # event_to_listen = None, 
        poll_interval = 1000):
        #this should set this client to start listening to a specific contract
        #make this non-blocking
        #TODO: Make event filtering work!
        if buyerAddress:
            assert(is_address(buyerAddress))
            self.buyerContract = web3.eth.contract(
                    address=buyerAddress,
                    abi=contractAbi)
            event_filter = self.buyerContract.eventFilter('ClientSelected', {'fromBlock': 'latest'})
            while True:
                for event in event_filter.get_new_entries():
                    handle_clientSelected_event(event)
                time.sleep(poll_interval)
        else:
            event_filter = self.masterContract.eventFilter('QueryCreated', {'fromBlock': 'latest'})
            while True:
                for event in event_filter.get_new_entries():
                    handle_queryCreated_event(event)
                time.sleep(poll_interval)

    def get_checkpoints_folder(self):
        return "./checkpoints-{0}/{1}/".format(self.iden, self.model_type)

    def get_latest_checkpoint(self):
        return tf.train.latest_checkpoint(self.get_checkpoints_folder())