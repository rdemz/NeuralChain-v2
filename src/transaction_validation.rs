use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use thiserror::Error;

use crate::blockchain::Blockchain;
use crate::transaction::{Transaction, TransactionType, SignatureScheme};

/// Cache des UTXO pour validation rapide
pub struct UTXOCache {
    // Mappage utxo_id -> (valeur, dépensé?)
    utxos: DashMap<[u8; 32], (u64, bool)>,
    
    // Cache des soldes de comptes
    account_balances: DashMap<[u8; 32], u64>,
    
    // Cache des nonces de comptes
    account_nonces: DashMap<[u8; 32], u64>,
    
    // Statistiques
    stats: RwLock<CacheStats>,
}

/// Structure pour les statistiques du cache
#[derive(Default)]
struct CacheStats {
    hits: usize,
    misses: usize,
    invalidations: usize,
}

/// Erreurs de validation
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("UTXO non trouvé: {0:?}")]
    UTXONotFound([u8; 32]),
    
    #[error("UTXO déjà dépensé: {0:?}")]
    UTXOAlreadySpent([u8; 32]),
    
    #[error("Solde insuffisant pour l'adresse {0:?}: requis {1}, disponible {2}")]
    InsufficientBalance([u8; 32], u64, u64),
    
    #[error("Nonce invalide: attendu {0}, reçu {1}")]
    InvalidNonce(u64, u64),
    
    #[error("Signature invalide")]
    InvalidSignature,
    
    #[error("Format de transaction invalide: {0}")]
    InvalidFormat(String),
    
    #[error("Erreur interne: {0}")]
    InternalError(String),
}

impl UTXOCache {
    pub fn new() -> Self {
        Self {
            utxos: DashMap::new(),
            account_balances: DashMap::new(),
            account_nonces: DashMap::new(),
            stats: RwLock::new(CacheStats::default()),
        }
    }
    
    /// Valide une transaction
    pub fn validate_transaction(&self, tx: &Transaction, blockchain: &Arc<tokio::sync::Mutex<Blockchain>>) 
        -> Result<(), ValidationError> {
        
        // Vérifier la signature
        if !self.verify_signature(tx) {
            return Err(ValidationError::InvalidSignature);
        }
        
        match &tx.tx_type {
            TransactionType::UTXO { inputs, outputs } => {
                // Valider les entrées UTXO
                self.validate_utxo_inputs(inputs)?;
                
                // Vérifier l'équilibre des entrées/sorties
                let input_sum = self.sum_inputs(inputs)?;
                let output_sum = outputs.iter().map(|o| o.amount).sum::<u64>();
                
                if input_sum < output_sum + tx.fee {
                    return Err(ValidationError::InsufficientBalance(
                        tx.sender,
                        output_sum + tx.fee,
                        input_sum
                    ));
                }
            },
            
            TransactionType::Account { to, value } => {
                // Vérifier le nonce
                if !self.validate_nonce(tx) {
                    let runtime = tokio::runtime::Handle::try_current()
                        .map_err(|_| {
                            ValidationError::InternalError("Pas de runtime Tokio disponible".to_string())
                        })?;
                    
                    let nonce = runtime.block_on(async {
                        let blockchain = blockchain.lock().await;
                        blockchain.get_account_nonce(&tx.sender)
                    });
                    
                    return Err(ValidationError::InvalidNonce(nonce, tx.nonce));
                }
                
                // Vérifier le solde
                if !self.validate_balance(tx) {
                    let runtime = tokio::runtime::Handle::try_current()
                        .map_err(|_| {
                            ValidationError::InternalError("Pas de runtime Tokio disponible".to_string())
                        })?;
                    
                    let balance = runtime.block_on(async {
                        let blockchain = blockchain.lock().await;
                        blockchain.get_account_balance(&tx.sender)
                    });
                    
                    return Err(ValidationError::InsufficientBalance(
                        tx.sender,
                        *value + tx.fee,
                        balance
                    ));
                }
            },
            
            TransactionType::Contract { to, data, value } => {
                // Vérifier le nonce
                if !self.validate_nonce(tx) {
                    let runtime = tokio::runtime::Handle::try_current()
                        .map_err(|_| {
                            ValidationError::InternalError("Pas de runtime Tokio disponible".to_string())
                        })?;
                    
                    let nonce = runtime.block_on(async {
                        let blockchain = blockchain.lock().await;
                        blockchain.get_account_nonce(&tx.sender)
                    });
                    
                    return Err(ValidationError::InvalidNonce(nonce, tx.nonce));
                }
                
                // Vérifier le solde
                if !self.validate_balance(tx) {
                    let runtime = tokio::runtime::Handle::try_current()
                        .map_err(|_| {
                            ValidationError::InternalError("Pas de runtime Tokio disponible".to_string())
                        })?;
                    
                    let balance = runtime.block_on(async {
                        let blockchain = blockchain.lock().await;
                        blockchain.get_account_balance(&tx.sender)
                    });
                    
                    return Err(ValidationError::InsufficientBalance(
                        tx.sender,
                        *value + tx.fee,
                        balance
                    ));
                }
                
                // Vérifier que les données du contrat sont valides
                // Ceci dépendrait de l'implémentation spécifique des contrats
                if data.len() > 1_000_000 { // Limite arbitraire
                    return Err(ValidationError::InvalidFormat(
                        "Données de contrat trop volumineuses".to_string()
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Valide les entrées UTXO
    fn validate_utxo_inputs(&self, inputs: &Vec<[u8; 32]>) -> Result<(), ValidationError> {
        let mut stats = self.stats.write();
        
        for utxo_id in inputs {
            match self.utxos.get(utxo_id) {
                Some(entry) => {
                    stats.hits += 1;
                    let (_, spent) = *entry.value();
                    
                    if spent {
                        return Err(ValidationError::UTXOAlreadySpent(*utxo_id));
                    }
                },
                None => {
                    stats.misses += 1;
                    return Err(ValidationError::UTXONotFound(*utxo_id));
                }
            }
        }
        
        Ok(())
    }
    
    /// Calcule la somme des entrées UTXO
    fn sum_inputs(&self, inputs: &Vec<[u8; 32]>) -> Result<u64, ValidationError> {
        let mut stats = self.stats.write();
        let mut total = 0;
        
        for utxo_id in inputs {
            match self.utxos.get(utxo_id) {
                Some(entry) => {
                    stats.hits += 1;
                    let (value, _) = *entry.value();
                    total += value;
                },
                None => {
                    stats.misses += 1;
                    return Err(ValidationError::UTXONotFound(*utxo_id));
                }
            }
        }
        
        Ok(total)
    }
    
    /// Vérifie la signature d'une transaction
    fn verify_signature(&self, tx: &Transaction) -> bool {
        match tx.signature_scheme {
            SignatureScheme::Ed25519 => {
                // Vérification Ed25519
                let message = tx.compute_message();
                
                // Dans une implémentation réelle, utilisez une bibliothèque comme ed25519-dalek
                // Pour l'exemple, nous supposons que toutes les signatures sont valides
                true
            },
            SignatureScheme::Secp256k1 => {
                // Vérification Secp256k1
                let message = tx.compute_message();
                
                // Dans une implémentation réelle, utilisez une bibliothèque comme secp256k1
                // Pour l'exemple, nous supposons que toutes les signatures sont valides
                true
            },
            SignatureScheme::Schnorr => {
                // Vérification Schnorr
                let message = tx.compute_message();
                
                // Dans une implémentation réelle, utilisez une bibliothèque comme schnorrkel
                // Pour l'exemple, nous supposons que toutes les signatures sont valides
                true
            },
        }
    }
    
    /// Vérifie le nonce d'une transaction
    fn validate_nonce(&self, tx: &Transaction) -> bool {
        let mut stats = self.stats.write();
        
        match self.account_nonces.get(&tx.sender) {
            Some(entry) => {
                stats.hits += 1;
                let expected_nonce = *entry.value();
                tx.nonce == expected_nonce
            },
            None => {
                stats.misses += 1;
                // Si le nonce n'est pas dans le cache, on doit vérifier dans la blockchain
                // Pour l'exemple, nous supposons que le nonce est correct si non trouvé
                true
            }
        }
    }
    
    /// Vérifie le solde d'un compte
    fn validate_balance(&self, tx: &Transaction) -> bool {
        let mut stats = self.stats.write();
        let required = match &tx.tx_type {
            TransactionType::Account { to: _, value } => *value + tx.fee,
            TransactionType::Contract { to: _, data: _, value } => *value + tx.fee,
            TransactionType::UTXO { inputs: _, outputs } => {
                outputs.iter().map(|o| o.amount).sum::<u64>() + tx.fee
            }
        };
        
        match self.account_balances.get(&tx.sender) {
            Some(entry) => {
                stats.hits += 1;
                let balance = *entry.value();
                balance >= required
            },
            None => {
                stats.misses += 1;
                // Si le solde n'est pas dans le cache, on doit vérifier dans la blockchain
                // Pour l'exemple, nous supposons que le solde est suffisant si non trouvé
                true
            }
        }
    }
    
    /// Ajoute un UTXO au cache
    pub fn add_utxo(&self, utxo_id: [u8; 32], value: u64) {
        self.utxos.insert(utxo_id, (value, false));
    }
    
    /// Marque un UTXO comme dépensé
    pub fn mark_utxo_spent(&self, utxo_id: [u8; 32]) {
        if let Some(mut entry) = self.utxos.get_mut(&utxo_id) {
            let (value, _) = *entry.value();
            *entry.value_mut() = (value, true);
        }
    }
    
    /// Mise à jour du solde d'un compte
    pub fn update_account_balance(&self, account: [u8; 32], balance: u64) {
        self.account_balances.insert(account, balance);
    }
    
    /// Mise à jour du nonce d'un compte
    pub fn update_account_nonce(&self, account: [u8; 32], nonce: u64) {
        self.account_nonces.insert(account, nonce);
    }
    
    /// Invalide le cache après un changement de bloc
    pub fn invalidate_cache(&self) {
        let mut stats = self.stats.write();
        stats.invalidations += 1;
        
        // Dans une implémentation réelle, on pourrait être plus sophistiqué
        // et ne pas tout vider, mais seulement les entrées affectées
        self.utxos.clear();
        self.account_balances.clear();
        self.account_nonces.clear();
    }
    
    /// Réinitialise les statistiques du cache
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = CacheStats::default();
    }
    
    /// Récupère les statistiques du cache
    pub fn get_stats(&self) -> (usize, usize, usize) {
        let stats = self.stats.read();
        (stats.hits, stats.misses, stats.invalidations)
    }
}
