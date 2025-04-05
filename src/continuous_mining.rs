use crate::block::Block;
use crate::blockchain::Blockchain;
use crate::transaction::Transaction;
use anyhow::{Result, Context};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing;

/// Structure pour gérer le minage continu
pub struct ContinuousMining {
    /// Référence à la blockchain
    blockchain: Arc<Mutex<Blockchain>>,
    /// File d'attente des transactions en attente
    mempool: Vec<Transaction>,
    /// Difficulté de minage actuelle
    current_difficulty: u32,
    /// Limite de transactions par bloc
    tx_limit_per_block: usize,
    /// Indicateur d'arrêt du minage
    stop_mining: bool,
}

impl ContinuousMining {
    /// Crée une nouvelle instance de minage continu
    pub fn new(blockchain: Arc<Mutex<Blockchain>>, initial_difficulty: u32) -> Self {
        Self {
            blockchain,
            mempool: Vec::new(),
            current_difficulty: initial_difficulty,
            tx_limit_per_block: 100, // Valeur par défaut
            stop_mining: false,
        }
    }
    
    /// Ajoute une transaction à la mempool
    pub fn add_transaction(&mut self, transaction: Transaction) -> Result<()> {
        // Vérifier la signature de la transaction
        if let Ok(false) = transaction.verify_signature() {
            return Err(anyhow::anyhow!("Signature de transaction invalide"));
        }
        
        // Ajouter à la mempool
        self.mempool.push(transaction);
        Ok(())
    }
    
    /// Commence le minage en continu
    pub async fn start_mining(&mut self) -> Result<()> {
        self.stop_mining = false;
        
        while !self.stop_mining {
            // Obtenir le dernier bloc
            let latest_block = {
                let blockchain = self.blockchain.lock().await;
                match blockchain.get_latest_block() {
                    Some(block) => block,
                    None => {
                        // Cas spécial: la blockchain est vide, créer un bloc de genèse
                        Block::genesis()
                    }
                }
            };
            
            // Préparer un nouveau bloc
            let selected_transactions = self.select_transactions();
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
                
            let mut new_block = Block::new(
                &latest_block,
                selected_transactions,
                timestamp,
                self.current_difficulty
            );
            
            // Miner le bloc
            if let Err(e) = new_block.mine() {
                tracing::error!("Erreur lors du minage: {}", e);
                continue;
            }
            
            // Ajouter le bloc à la blockchain
            {
                let mut blockchain = self.blockchain.lock().await;
                if let Err(e) = blockchain.add_block(new_block) {
                    tracing::error!("Erreur lors de l'ajout du bloc: {}", e);
                }
            }
            
            // Attente courte entre les blocs pour éviter de surcharger le système
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            
            // Ajuster la difficulté si nécessaire (logique à implémenter)
            self.adjust_difficulty().await?;
        }
        
        Ok(())
    }
    
    /// Arrête le minage
    pub fn stop_mining(&mut self) {
        self.stop_mining = true;
    }
    
    /// Sélectionne les transactions à inclure dans le prochain bloc
    fn select_transactions(&mut self) -> Vec<Transaction> {
        // Trier les transactions par frais décroissants
        self.mempool.sort_by(|a, b| b.fee.cmp(&a.fee));
        
        // Prendre les N premières transactions (où N est la limite par bloc)
        let selected: Vec<Transaction> = self.mempool
            .drain(0..std::cmp::min(self.tx_limit_per_block, self.mempool.len()))
            .collect();
            
        selected
    }
    
    /// Ajuste la difficulté en fonction du temps de minage des blocs récents
    async fn adjust_difficulty(&mut self) -> Result<()> {
        // Logique d'ajustement de la difficulté à implémenter
        // (cette fonction simple ne fait rien pour le moment)
        Ok(())
    }
    
    /// Définit la limite de transactions par bloc
    pub fn set_tx_limit(&mut self, limit: usize) {
        self.tx_limit_per_block = limit;
    }
    
    /// Obtient le nombre de transactions en attente
    pub fn get_mempool_size(&self) -> usize {
        self.mempool.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::{Transaction, TransactionType};
    
    // Ces tests nécessitent tokio
    #[tokio::test]
    async fn test_add_transaction() {
        let blockchain = Arc::new(Mutex::new(Blockchain::new()));
        let mut miner = ContinuousMining::new(blockchain, 1);
        
        // Créer un wallet pour signer une transaction
        let mut wallet = crate::wallet::Wallet::new().unwrap();
        let tx = wallet.create_transaction(
            None,
            100,
            10,
            TransactionType::Transfer,
            vec![],
        ).unwrap();
        
        // Ajouter la transaction à la mempool
        let result = miner.add_transaction(tx);
        assert!(result.is_ok());
        
        // Vérifier que la transaction a été ajoutée
        assert_eq!(miner.get_mempool_size(), 1);
    }
}
