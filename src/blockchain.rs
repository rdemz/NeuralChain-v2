use crate::block::Block;
use anyhow::{Result, Context, bail};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Structure principale de la blockchain
pub struct Blockchain {
    /// Chaîne de blocs stockée par hash
    blocks: HashMap<Vec<u8>, Block>,
    /// Index de hauteur pour accès rapide
    height_index: HashMap<u64, Vec<u8>>,
    /// Hash du bloc le plus récent
    latest_block_hash: Option<Vec<u8>>,
    /// Hauteur du bloc le plus récent
    latest_height: u64,
}

impl Blockchain {
    /// Crée une nouvelle blockchain vide
    pub fn new_empty() -> Self {
        Self {
            blocks: HashMap::new(),
            height_index: HashMap::new(),
            latest_block_hash: None,
            latest_height: 0,
        }
    }
    
    /// Crée une nouvelle blockchain avec le bloc de genèse
    pub fn new() -> Self {
        let mut blockchain = Self::new_empty();
        let genesis = Block::genesis();
        blockchain.add_block(genesis).expect("Le bloc de genèse devrait être valide");
        blockchain
    }
    
    /// Ajoute un bloc à la blockchain
    pub fn add_block(&mut self, block: Block) -> Result<()> {
        let block_hash = block.hash.clone();
        let block_height = block.height;
        
        // Vérifier si le bloc existe déjà
        if self.blocks.contains_key(&block_hash) {
            bail!("Le bloc existe déjà dans la blockchain");
        }
        
        // Vérifier la cohérence de la hauteur
        if block_height > 0 && !self.blocks.contains_key(&block.prev_hash) {
            bail!("Le bloc parent n'existe pas dans la blockchain");
        }
        
        // Ajouter le bloc aux structures de données
        self.blocks.insert(block_hash.clone(), block);
        self.height_index.insert(block_height, block_hash.clone());
        
        // Mettre à jour le dernier bloc si nécessaire
        if block_height > self.latest_height {
            self.latest_height = block_height;
            self.latest_block_hash = Some(block_hash);
        }
        
        Ok(())
    }
    
    /// Récupère un bloc par son hash
    pub fn get_block(&self, hash: &[u8]) -> Option<Block> {
        self.blocks.get(hash).cloned()
    }
    
    /// Récupère un bloc par sa hauteur
    pub fn get_block_by_height(&self, height: u64) -> Option<Block> {
        self.height_index.get(&height)
            .and_then(|hash| self.get_block(hash))
    }
    
    /// Récupère le dernier bloc de la chaîne
    pub fn get_latest_block(&self) -> Option<Block> {
        self.latest_block_hash
            .as_ref()
            .and_then(|hash| self.get_block(hash))
    }
    
    /// Retourne la hauteur actuelle de la blockchain
    pub fn get_current_height(&self) -> u64 {
        self.latest_height
    }
    
    /// Vérifie si un bloc existe dans la blockchain
    pub fn contains_block(&self, hash: &[u8]) -> bool {
        self.blocks.contains_key(hash)
    }
    
    /// Retourne les N derniers blocs dans l'ordre chronologique
    pub fn get_last_n_blocks(&self, n: usize) -> Vec<Block> {
        let mut blocks = Vec::with_capacity(n);
        let mut current_height = self.latest_height;
        
        while blocks.len() < n && current_height > 0 {
            if let Some(block) = self.get_block_by_height(current_height) {
                blocks.push(block);
            }
            current_height = current_height.saturating_sub(1);
        }
        
        blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_blockchain_creation() {
        let blockchain = Blockchain::new();
        assert_eq!(blockchain.get_current_height(), 0);
        
        // Vérifier que le bloc de genèse existe
        let genesis = blockchain.get_block_by_height(0);
        assert!(genesis.is_some());
    }
    
    #[test]
    fn test_add_block() {
        let mut blockchain = Blockchain::new();
        let genesis = blockchain.get_block_by_height(0).unwrap();
        
        // Créer un nouveau bloc
        let new_block = Block {
            version: 1,
            height: 1,
            timestamp: 12345,
            prev_hash: genesis.hash.clone(),
            merkle_root: vec![0; 32],
            difficulty: 1,
            nonce: 0,
            transactions: vec![],
            hash: vec![1; 32],
        };
        
        // Ajouter le bloc
        assert!(blockchain.add_block(new_block.clone()).is_ok());
        
        // Vérifier que le bloc a été ajouté
        let retrieved_block = blockchain.get_block_by_height(1);
        assert!(retrieved_block.is_some());
        assert_eq!(retrieved_block.unwrap().hash, new_block.hash);
    }
}
