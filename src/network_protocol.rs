use crate::block::Block;
use crate::transaction::Transaction;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use std::fmt;

/// Types de messages réseau pour NeuralChain
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MessageType {
    HandShake,
    Ping,
    Pong,
    GetBlocks,
    Blocks,
    GetTransactions,
    Transactions,
    NewBlock,
    NewTransaction,
    Disconnect,
    PeerExchange,
}

/// Structure principale des messages réseau
#[derive(Serialize, Deserialize, Clone)]
pub struct NetworkMessage {
    pub message_type: MessageType,
    pub peer_id: String,
    pub timestamp: u64,
    pub payload: Vec<u8>,
}

impl fmt::Debug for NetworkMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NetworkMessage")
            .field("message_type", &self.message_type)
            .field("peer_id", &self.peer_id)
            .field("timestamp", &self.timestamp)
            .field("payload_size", &self.payload.len())
            .finish()
    }
}

/// Message de demande de blocs
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetBlocksMessage {
    pub start_height: u64,
    pub count: u32,
    pub include_transactions: bool,
}

/// Message de demande de transactions
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetTransactionsMessage {
    pub transaction_hashes: Vec<Vec<u8>>,
}

/// Message de partage de pairs
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PeerExchangeMessage {
    pub peers: Vec<PeerInfo>,
}

/// Informations sur un pair
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PeerInfo {
    pub id: String,
    pub address: String,
    pub port: u16,
    pub last_seen: u64,
    pub reputation_score: f32,
}

impl NetworkMessage {
    /// Crée un nouveau message réseau
    pub fn new(message_type: MessageType, peer_id: String, payload: Vec<u8>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        Self {
            message_type,
            peer_id,
            timestamp,
            payload,
        }
    }
    
    /// Sérialise le message en binaire
    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).context("Échec de la sérialisation du message réseau")
    }
    
    /// Désérialise un message à partir de données binaires
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).context("Échec de la désérialisation du message réseau")
    }
    
    /// Crée un message pour demander des blocs
    pub fn new_get_blocks(peer_id: String, start_height: u64, count: u32) -> Result<Self> {
        let request = GetBlocksMessage {
            start_height,
            count,
            include_transactions: true,
        };
        
        let payload = bincode::serialize(&request)
            .context("Échec de la sérialisation de la demande de blocs")?;
            
        Ok(Self::new(MessageType::GetBlocks, peer_id, payload))
    }
    
    /// Crée un message pour annoncer un nouveau bloc
    pub fn new_block(peer_id: String, block: &Block) -> Result<Self> {
        let payload = bincode::serialize(block)
            .context("Échec de la sérialisation du bloc")?;
            
        Ok(Self::new(MessageType::NewBlock, peer_id, payload))
    }
    
    /// Crée un message pour annoncer une nouvelle transaction
    pub fn new_transaction(peer_id: String, transaction: &Transaction) -> Result<Self> {
        let payload = bincode::serialize(transaction)
            .context("Échec de la sérialisation de la transaction")?;
            
        Ok(Self::new(MessageType::NewTransaction, peer_id, payload))
    }
}
