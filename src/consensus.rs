use crate::block::Block;
use crate::blockchain::Blockchain;
use crate::neural_pow::NeuralPoW;
use crate::transaction::Transaction;
use anyhow::{Result, Context, bail};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Configuration du module de consensus
pub struct ConsensusConfig {
    pub target_block_time_ms: u64,
    pub min_difficulty: u64,
    pub difficulty_adjustment_interval: u64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            target_block_time_ms: 10_000,  // 10 secondes
            min_difficulty: 100,
            difficulty_adjustment_interval: 10,  // Ajustement tous les 10 blocs
        }
    }
}

/// Moteur de consensus pour NeuralChain
pub struct ConsensusEngine {
    blockchain: Arc<Mutex<Blockchain>>,
    pow_system: Arc<NeuralPoW>,
    config: ConsensusConfig,
}

impl ConsensusEngine {
    /// Crée un nouveau moteur de consensus
    pub fn new(
        blockchain: Arc<Mutex<Blockchain>>,
        pow_system: Arc<NeuralPoW>,
        config: ConsensusConfig,
    ) -> Self {
        Self {
            blockchain,
            pow_system,
            config,
        }
    }
    
    /// Vérifie la validité d'un bloc
    pub async fn verify_block(&self, block: &Block) -> Result<bool> {
        // 1. Vérifier la structure de base du bloc
        if block.version < 1 {
            bail!("Version de bloc invalide");
        }
        
        // 2. Vérifier l'ordre chronologique
        let blockchain = self.blockchain.lock().await;
        if block.height > 0 {
            let parent = blockchain.get_block(&block.prev_hash)
                .context("Bloc parent introuvable")?;
                
            if block.timestamp <= parent.timestamp {
                bail!("Horodatage de bloc invalide");
            }
        }
        drop(blockchain);
        
        // 3. Vérifier la preuve de travail
        let difficulty = if block.height == 0 {
            // Bloc de genèse
            self.config.min_difficulty
        } else {
            block.difficulty
        };
        
        if !self.pow_system.verify_pow(&block.hash, block.nonce, difficulty).await {
            bail!("Preuve de travail invalide");
        }
        
        // 4. Valider les transactions
        self.validate_transactions(&block.transactions).await?;
        
        Ok(true)
    }
    
    /// Calcule la difficulté pour le prochain bloc
    pub async fn calculate_next_difficulty(&self, last_block: &Block) -> u64 {
        // Si c'est trop tôt pour un ajustement, retourner la difficulté actuelle
        if last_block.height % self.config.difficulty_adjustment_interval != 0 {
            return last_block.difficulty;
        }
        
        // Obtenir les blocs pour calculer le temps moyen
        let blockchain = self.blockchain.lock().await;
        let start_height = last_block.height.saturating_sub(self.config.difficulty_adjustment_interval);
        let start_block = match blockchain.get_block_by_height(start_height) {
            Some(block) => block,
            None => return self.config.min_difficulty,
        };
        drop(blockchain);
        
        // Calculer le temps écoulé
        let time_span = last_block.timestamp.saturating_sub(start_block.timestamp);
        let target_time = self.config.target_block_time_ms * self.config.difficulty_adjustment_interval;
        
        // Ajuster la difficulté en fonction du ratio
        let mut new_difficulty = (last_block.difficulty as f64 * target_time as f64 / time_span as f64).round() as u64;
        
        // Limiter l'ajustement à ±25%
        let max_adjustment = last_block.difficulty / 4;
        if new_difficulty > last_block.difficulty + max_adjustment {
            new_difficulty = last_block.difficulty + max_adjustment;
        } else if new_difficulty < last_block.difficulty.saturating_sub(max_adjustment) {
            new_difficulty = last_block.difficulty.saturating_sub(max_adjustment);
        }
        
        // Appliquer la difficulté minimale
        new_difficulty.max(self.config.min_difficulty)
    }
    
    /// Crée un nouveau bloc prêt pour le minage
    pub async fn create_new_block(
        &self,
        parent_hash: Vec<u8>,
        height: u64,
        timestamp: u64,
        transactions: Vec<Transaction>,
        nonce: u64,
    ) -> Result<Block> {
        // Calculer la racine de Merkle
        let merkle_root = self.calculate_merkle_root(&transactions);
        
        // Déterminer la difficulté
        let blockchain = self.blockchain.lock().await;
        let parent = blockchain.get_block(&parent_hash)
            .context("Bloc parent introuvable")?;
        drop(blockchain);
        
        let difficulty = self.calculate_next_difficulty(&parent).await;
        
        // Créer le bloc
        let mut block = Block {
            version: 1,
            height,
            timestamp,
            prev_hash: parent_hash,
            merkle_root,
            difficulty,
            nonce,
            transactions,
            hash: vec![0; 32],  // Sera calculé par la méthode calculate_hash
        };
        
        // Calculer le hash du bloc
        block.hash = self.pow_system.compute_pow_hash(&block.calculate_header_hash(), nonce).await;
        
        Ok(block)
    }
    
    /// Valide un ensemble de transactions
    pub async fn validate_transactions(&self, transactions: &[Transaction]) -> Result<bool> {
        // Validation simple pour l'instant
        // TODO: Implémentation complète
        Ok(true)
    }
    
    /// Calcule la racine de Merkle pour un ensemble de transactions
    fn calculate_merkle_root(&self, transactions: &[Transaction]) -> Vec<u8> {
        // Version simplifiée pour l'exemple
        if transactions.is_empty() {
            return vec![0; 32];
        }
        
        // Dans une implémentation réelle, nous construirions un arbre de Merkle
        // Pour l'instant, nous concaténons simplement les hashes
        let mut combined = Vec::new();
        for tx in transactions {
            combined.extend_from_slice(&tx.hash);
        }
        
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&combined);
        hasher.finalize().to_vec()
    }
    
    /// Getter pour le système PoW
    pub fn get_pow_system(&self) -> Arc<NeuralPoW> {
        self.pow_system.clone()
    }
}
