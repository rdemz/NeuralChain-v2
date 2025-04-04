use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::sync::Arc;
use parking_lot::RwLock;
use std::time::{Duration, Instant};
use tracing::{trace, debug, warn};

use crate::transaction::Transaction;

/// Structure de mempool optimisée
pub struct OptimizedMempool {
    // Map d'ID de transaction à transaction
    txs: RwLock<HashMap<[u8; 32], Transaction>>,
    
    // Structure ordonnée pour récupérer efficacement les transactions avec les frais les plus élevés
    // Utilise un tas pour maintenir l'ordre avec O(log n) pour insertion/suppression
    fee_priority_queue: RwLock<BinaryHeap<TransactionPriority>>,
    
    // Map d'expéditeur aux transactions, pour la gestion des nonces
    sender_txs: RwLock<HashMap<[u8; 32], Vec<[u8; 32]>>>,
    
    // Horodatages d'arrivée pour gérer l'expiration
    arrival_times: RwLock<HashMap<[u8; 32], Instant>>,
    
    // Configuration
    max_size: usize,
    max_age: Duration,
    
    // Statistiques
    stats: RwLock<MempoolStats>,
}

/// Statistiques du mempool
#[derive(Default)]
struct MempoolStats {
    tx_added: u64,
    tx_rejected: u64,
    tx_expired: u64,
    tx_included_in_block: u64,
    tx_replaced: u64,
}

/// Structure pour ordonner les transactions par priorité de frais
#[derive(Clone, Eq, PartialEq)]
struct TransactionPriority {
    tx_id: [u8; 32],
    fee_per_byte: u64,
    fee: u64,
    arrival_time: Instant,
}

// Implémentation pour le tas binaire
impl Ord for TransactionPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // D'abord comparer les frais par octet
        self.fee_per_byte.cmp(&other.fee_per_byte)
            // En cas d'égalité, comparer les frais totaux
            .then(self.fee.cmp(&other.fee))
            // En cas d'égalité, les transactions plus anciennes ont la priorité (FIFO)
            .then(other.arrival_time.cmp(&self.arrival_time))
    }
}

impl PartialOrd for TransactionPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl OptimizedMempool {
    /// Crée un nouveau mempool avec une taille maximale et une durée de vie
    pub fn new(max_size: usize, max_age_secs: u64) -> Self {
        Self {
            txs: RwLock::new(HashMap::with_capacity(max_size)),
            fee_priority_queue: RwLock::new(BinaryHeap::with_capacity(max_size)),
            sender_txs: RwLock::new(HashMap::new()),
            arrival_times: RwLock::new(HashMap::with_capacity(max_size)),
            max_size,
            max_age: Duration::from_secs(max_age_secs),
            stats: RwLock::new(MempoolStats::default()),
        }
    }
    
    /// Ajoute une transaction au mempool
    pub fn add_transaction(&self, tx: Transaction) -> bool {
        let tx_id = tx.id;
        let sender = tx.sender;
        
        // Vérifier si la transaction existe déjà
        {
            let txs = self.txs.read();
            if txs.contains_key(&tx_id) {
                // Transaction déjà dans le mempool
                trace!("Transaction {} déjà dans le mempool", hex::encode(&tx_id[0..4]));
                
                let mut stats = self.stats.write();
                stats.tx_rejected += 1;
                
                return false;
            }
        }
        
        // Vérifier si le mempool est plein
        {
            let txs = self.txs.read();
            if txs.len() >= self.max_size {
                // Supprimer la transaction avec les frais les plus bas si nécessaire
                debug!("Mempool plein, tentative de suppression de la transaction avec les frais les plus bas");
                
                if !self.remove_lowest_fee_transaction() {
                    warn!("Impossible de supprimer la transaction avec les frais les plus bas");
                    
                    let mut stats = self.stats.write();
                    stats.tx_rejected += 1;
                    
                    return false;
                }
            }
        }
        
        // Calculer la priorité (frais par octet)
        let tx_size = tx.size();
        let fee_per_byte = if tx_size > 0 {
            tx.fee / tx_size as u64
        } else {
            tx.fee
        };
        
        let priority = TransactionPriority {
            tx_id,
            fee_per_byte,
            fee: tx.fee,
            arrival_time: Instant::now(),
        };
        
        // Ajouter la transaction dans toutes les structures
        {
            let mut txs = self.txs.write();
            let mut fee_pq = self.fee_priority_queue.write();
            let mut sender_txs = self.sender_txs.write();
            let mut arrival_times = self.arrival_times.write();
            
            // Ajouter aux structures
            txs.insert(tx_id, tx);
            fee_pq.push(priority);
            arrival_times.insert(tx_id, Instant::now());
            
            // Ajouter à la map d'expéditeur
            sender_txs.entry(sender)
                .or_insert_with(Vec::new)
                .push(tx_id);
            
            let mut stats = self.stats.write();
            stats.tx_added += 1;
        }
        
        true
    }
    
    /// Supprime une transaction du mempool
    pub fn remove_transaction(&self, tx_id: &[u8; 32]) -> Option<Transaction> {
        let mut txs = self.txs.write();
        let mut fee_pq = self.fee_priority_queue.write();
        let mut arrival_times = self.arrival_times.write();
        
        // Supprimer de la map principale
        let tx = txs.remove(tx_id)?;
        
        // Supprimer des autres structures
        // Note: La suppression du tas binaire est coûteuse (O(n)),
        // nous le ferons paresseusement lors de l'extraction des transactions
        
        arrival_times.remove(tx_id);
        
        // Supprimer de la liste de l'expéditeur
        let mut sender_txs = self.sender_txs.write();
        if let Some(sender_list) = sender_txs.get_mut(&tx.sender) {
            sender_list.retain(|id| id != tx_id);
            
            // Supprimer l'entrée si c'était la dernière transaction de cet expéditeur
            if sender_list.is_empty() {
                sender_txs.remove(&tx.sender);
            }
        }
        
        Some(tx)
    }
    
    /// Nettoie les transactions expirées et renvoie le nombre supprimé
    pub fn clean_expired(&self) -> usize {
        let now = Instant::now();
        let mut expired = Vec::new();
        
        // Identifier les transactions expirées
        {
            let arrival_times = self.arrival_times.read();
            
            for (tx_id, time) in arrival_times.iter() {
                if now.duration_since(*time) > self.max_age {
                    expired.push(*tx_id);
                }
            }
        }
        
        // Supprimer les transactions expirées
        let count = expired.len();
        
        for tx_id in expired {
            self.remove_transaction(&tx_id);
        }
        
        // Mettre à jour les statistiques
        {
            let mut stats = self.stats.write();
            stats.tx_expired += count as u64;
        }
        
        count
    }
    
    /// Récupère les transactions pour un bloc, ordonnées par priorité de frais
    pub fn get_transactions_for_block(&self, max_count: usize, max_size: usize) -> Vec<Transaction> {
        let mut result = Vec::new();
        let mut total_size = 0;
        
        // Copier la file de priorité pour ne pas modifier l'originale
        let mut pq = self.fee_priority_queue.read().clone();
        
        // Collecter les transactions valides (non expirées)
        let now = Instant::now();
        let txs = self.txs.read();
        let arrival_times = self.arrival_times.read();
        
        while result.len() < max_count && !pq.is_empty() {
            // Récupérer la transaction avec les frais les plus élevés
            if let Some(priority) = pq.pop() {
                // Vérifier si la transaction est toujours dans le mempool
                if let Some(tx) = txs.get(&priority.tx_id) {
                    // Vérifier si la transaction n'est pas expirée
                    if let Some(arrival_time) = arrival_times.get(&priority.tx_id) {
                        if now.duration_since(*arrival_time) <= self.max_age {
                            let tx_size = tx.size();
                            
                            // Vérifier si le bloc a assez d'espace
                            if total_size + tx_size <= max_size {
                                result.push(tx.clone());
                                total_size += tx_size;
                            }
                        }
                    }
                }
            }
        }
        
        result
    }
    
    /// Renvoie toutes les transactions d'un expéditeur
    pub fn get_transactions_by_sender(&self, sender: &[u8; 32]) -> Vec<Transaction> {
        let mut result = Vec::new();
        
        // Récupérer les IDs des transactions de l'expéditeur
        let sender_txs = self.sender_txs.read();
        let tx_ids = match sender_txs.get(sender) {
            Some(ids) => ids,
            None => return result,
        };
        
        // Récupérer les transactions
        let txs = self.txs.read();
        
        for tx_id in tx_ids {
            if let Some(tx) = txs.get(tx_id) {
                result.push(tx.clone());
            }
        }
        
        // Trier par nonce pour un traitement ordonné
        result.sort_by_key(|tx| tx.nonce);
        
        result
    }
    
    /// Vérifie si une transaction est dans le mempool
    pub fn contains_transaction(&self, tx_id: &[u8; 32]) -> bool {
        self.txs.read().contains_key(tx_id)
    }
    
    /// Récupère une transaction par son ID
    pub fn get_transaction(&self, tx_id: &[u8; 32]) -> Option<Transaction> {
        self.txs.read().get(tx_id).cloned()
    }
    
    /// Renvoie la taille actuelle du mempool
    pub fn size(&self) -> usize {
        self.txs.read().len()
    }
    
    /// Supprime la transaction avec les frais les plus bas
    fn remove_lowest_fee_transaction(&self) -> bool {
        // Reconstruire la file de priorité pour s'assurer qu'elle contient des transactions valides
        let mut txs_to_remove = Vec::new();
        
        {
            let txs = self.txs.read();
            let arrival_times = self.arrival_times.read();
            let now = Instant::now();
            
            // Créer une nouvelle file de priorité
            let mut new_pq = BinaryHeap::new();
            
            // Vérifier chaque transaction
            for (tx_id, tx) in txs.iter() {
                // Vérifier si la transaction est expirée
                if let Some(arrival_time) = arrival_times.get(tx_id) {
                    if now.duration_since(*arrival_time) > self.max_age {
                        txs_to_remove.push(*tx_id);
                        continue;
                    }
                }
                
                // Calculer la priorité
                let tx_size = tx.size();
                let fee_per_byte = if tx_size > 0 {
                    tx.fee / tx_size as u64
                } else {
                    tx.fee
                };
                
                // Ajouter à la nouvelle file
                new_pq.push(TransactionPriority {
                    tx_id: *tx_id,
                    fee_per_byte,
                    fee: tx.fee,
                    arrival_time: arrival_times.get(tx_id).copied().unwrap_or_else(Instant::now),
                });
            }
            
            // Mettre à jour la file de priorité
            *self.fee_priority_queue.write() = new_pq;
        }
        
        // Supprimer les transactions expirées trouvées
        for tx_id in txs_to_remove {
            self.remove_transaction(&tx_id);
        }
        
        // S'il y a des transactions expirées, pas besoin d'en supprimer plus
        if !txs_to_remove.is_empty() {
            return true;
        }
        
        // Sinon, supprimer la transaction avec les frais les plus bas
        let mut fee_pq = self.fee_priority_queue.write();
        
        if let Some(lowest) = fee_pq.pop() {
            // Supprimer la transaction
            self.remove_transaction(&lowest.tx_id);
            
            let mut stats = self.stats.write();
            stats.tx_replaced += 1;
            
            true
        } else {
            false
        }
    }
    
    /// Récupère les statistiques du mempool
    pub fn get_stats(&self) -> (u64, u64, u64, u64, u64) {
        let stats = self.stats.read();
        (
            stats.tx_added,
            stats.tx_rejected,
            stats.tx_expired,
            stats.tx_included_in_block,
            stats.tx_replaced,
        )
    }
    
    /// Réinitialise les statistiques du mempool
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = MempoolStats::default();
    }
    
    /// Marque des transactions comme incluses dans un bloc
    pub fn mark_transactions_included(&self, tx_ids: &[[u8; 32]]) {
        let mut stats = self.stats.write();
        stats.tx_included_in_block += tx_ids.len() as u64;
        
        // Supprimer les transactions du mempool
        for tx_id in tx_ids {
            self.remove_transaction(tx_id);
        }
    }
}
