use crate::blockchain::{Blockchain, BlockchainState};
use crate::block::Block;
use crate::neural_pow::NeuralPoW;
use crate::transaction::Transaction;
use crate::consensus::ConsensusEngine;
use anyhow::{Result, Context};
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc, broadcast};
use tokio::time::{sleep, Duration};
use tracing::{info, warn, error, debug};

/// Configuration pour le mineur continu
pub struct ContinuousMinerConfig {
    /// Intervalle entre les tentatives de minage en millisecondes
    pub mining_interval_ms: u64,
    /// Nombre maximum d'itérations de minage par tentative
    pub max_iterations_per_attempt: u64,
    /// Taille maximale du pool de transactions
    pub max_tx_pool_size: usize,
    /// Nombre maximum de transactions par bloc
    pub max_tx_per_block: usize,
}

impl Default for ContinuousMinerConfig {
    fn default() -> Self {
        Self {
            mining_interval_ms: 100,
            max_iterations_per_attempt: 10_000,
            max_tx_pool_size: 10_000,
            max_tx_per_block: 1_000,
        }
    }
}

/// Mineur continu pour NeuralChain
pub struct ContinuousMiner {
    blockchain: Arc<Mutex<Blockchain>>,
    pow_system: Arc<NeuralPoW>,
    consensus_engine: Arc<ConsensusEngine>,
    config: ContinuousMinerConfig,
    transaction_pool: Vec<Transaction>,
    new_block_tx: mpsc::Sender<Block>,
    stop_mining_rx: broadcast::Receiver<()>,
    stop_mining_tx: broadcast::Sender<()>,
}

impl ContinuousMiner {
    /// Crée une nouvelle instance du mineur continu
    pub fn new(
        blockchain: Arc<Mutex<Blockchain>>,
        pow_system: Arc<NeuralPoW>,
        consensus_engine: Arc<ConsensusEngine>,
        config: ContinuousMinerConfig,
    ) -> Self {
        let (new_block_tx, _) = mpsc::channel(100);
        let (stop_mining_tx, stop_mining_rx) = broadcast::channel(16);
        
        Self {
            blockchain,
            pow_system,
            consensus_engine,
            config,
            transaction_pool: Vec::new(),
            new_block_tx,
            stop_mining_rx,
            stop_mining_tx,
        }
    }
    
    /// Retourne un clone de l'émetteur de nouveaux blocs
    pub fn get_new_block_sender(&self) -> mpsc::Sender<Block> {
        self.new_block_tx.clone()
    }
    
    /// Retourne un récepteur pour arrêter le minage
    pub fn get_stop_mining_receiver(&self) -> broadcast::Receiver<()> {
        self.stop_mining_tx.subscribe()
    }
    
    /// Ajoute une transaction au pool
    pub fn add_transaction(&mut self, transaction: Transaction) -> Result<bool> {
        // Vérifier si le pool est plein
        if self.transaction_pool.len() >= self.config.max_tx_pool_size {
            return Ok(false);
        }
        
        // Vérifier si la transaction est valide
        if !transaction.verify_signature()? {
            return Ok(false);
        }
        
        // Ajouter au pool
        self.transaction_pool.push(transaction);
        
        Ok(true)
    }
    
    /// Démarre la boucle de minage
    pub async fn start_mining(&mut self) -> Result<()> {
        info!("Démarrage de la boucle de minage continue");
        
        let mut interval = tokio::time::interval(Duration::from_millis(self.config.mining_interval_ms));
        
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Vérifier l'état de la blockchain
                    let blockchain_state = {
                        let blockchain = self.blockchain.lock().await;
                        blockchain.get_state()
                    };
                    
                    if blockchain_state != BlockchainState::Active {
                        debug!("Blockchain non active, attente pour le minage");
                        continue;
                    }
                    
                    // Mine un bloc si possible
                    if let Err(e) = self.attempt_mining().await {
                        error!("Erreur pendant le minage : {}", e);
                    }
                }
                _ = self.stop_mining_rx.recv() => {
                    info!("Signal d'arrêt reçu, fin du minage");
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    /// Effectue une tentative de minage
    async fn attempt_mining(&mut self) -> Result<Option<Block>> {
        // Sélectionner les transactions pour le prochain bloc
        let selected_transactions = self.select_transactions();
        
        // Préparer le prochain bloc
        let next_block = self.prepare_next_block(selected_transactions).await?;
        
        // Tenter de miner le bloc
        let mining_result = self.pow_system
            .mine_block_limited(&next_block, self.config.max_iterations_per_attempt)
            .await?;
            
        // Si le minage a réussi
        if let Some((nonce, pow_hash)) = mining_result {
            // Créer le bloc final avec le nonce et le hash trouvés
            let mut mined_block = next_block.clone();
            mined_block.nonce = nonce;
            mined_block.hash = pow_hash;
            
            // Vérifier le bloc miné
            if self.consensus_engine.verify_block(&mined_block).await? {
                // Ajouter le bloc à la blockchain
                {
                    let mut blockchain = self.blockchain.lock().await;
                    blockchain.add_block(mined_block.clone())?;
                }
                
                info!("Nouveau bloc miné avec succès : hauteur={}, hash={:?}",
                      mined_block.height, &mined_block.hash[0..8]);
                      
                // Supprimer les transactions incluses du pool
                self.remove_mined_transactions(&mined_block.transactions);
                
                // Signaler le nouveau bloc
                if let Err(e) = self.new_block_tx.send(mined_block.clone()).await {
                    warn!("Impossible d'envoyer le nouveau bloc : {}", e);
                }
                
                return Ok(Some(mined_block));
            } else {
                warn!("Le bloc miné n'a pas passé la validation");
            }
        }
        
        Ok(None)
    }
    
    /// Prépare le prochain bloc à miner
    async fn prepare_next_block(&self, transactions: Vec<Transaction>) -> Result<Block> {
        let blockchain = self.blockchain.lock().await;
        let latest_block = blockchain.get_latest_block()
            .context("La blockchain n'a pas de bloc le plus récent")?;
            
        // Relâcher le verrou
        drop(blockchain);
        
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
            
        // Utiliser le moteur de consensus pour créer un nouveau bloc
        let next_block = self.consensus_engine.create_new_block(
            latest_block.hash.clone(),
            latest_block.height + 1,
            timestamp,
            transactions,
            0, // Le nonce sera trouvé pendant le minage
        ).await?;
        
        Ok(next_block)
    }
    
    /// Sélectionne les transactions à inclure dans le prochain bloc
    fn select_transactions(&self) -> Vec<Transaction> {
        // Pour la simplicité, prenons simplement les premières transactions
        self.transaction_pool.iter()
            .take(self.config.max_tx_per_block)
            .cloned()
            .collect()
    }
    
    /// Supprime les transactions minées du pool
    fn remove_mined_transactions(&mut self, mined_transactions: &[Transaction]) {
        if mined_transactions.is_empty() {
            return;
        }
        
        // Créer un ensemble des hashes des transactions minées
        let mined_tx_hashes: std::collections::HashSet<&Vec<u8>> = mined_transactions
            .iter()
            .map(|tx| &tx.hash)
            .collect();
            
        // Filtrer le pool pour ne garder que les transactions non minées
        self.transaction_pool.retain(|tx| !mined_tx_hashes.contains(&tx.hash));
    }
    
    /// Arrête le minage
    pub fn stop_mining(&self) {
        if let Err(e) = self.stop_mining_tx.send(()) {
            warn!("Impossible d'envoyer le signal d'arrêt : {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::TransactionType;
    
    #[tokio::test]
    async fn test_transaction_pool() {
        // Créer les composants nécessaires
        let blockchain = Arc::new(Mutex::new(Blockchain::new()));
        let pow_system = Arc::new(NeuralPoW::new(blockchain.clone()));
        let consensus_engine = Arc::new(ConsensusEngine::new(
            blockchain.clone(),
            pow_system.clone(),
            Default::default(),
        ));
        
        let mut miner = ContinuousMiner::new(
            blockchain.clone(),
            pow_system.clone(),
            consensus_engine.clone(),
            Default::default(),
        );
        
        // Créer une transaction de test
        let tx = Transaction::new(
            TransactionType::Transfer,
            vec![1; 32],
            Some(vec![2; 32]),
            100,
            10,
            1,
            vec![],
        );
        
        // Ajouter au pool
        assert!(miner.add_transaction(tx.clone()).unwrap());
        
        // Vérifier que la transaction est dans le pool
        assert_eq!(miner.transaction_pool.len(), 1);
        
        // Simuler un minage
        miner.remove_mined_transactions(&[tx]);
        
        // Vérifier que la transaction a été supprimée
        assert_eq!(miner.transaction_pool.len(), 0);
    }
}
