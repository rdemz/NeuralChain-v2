use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};

use crate::transaction::Transaction;

/// Type d'identifiant de bloc
pub type BlockId = [u8; 32];

/// En-tête d'un bloc
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockHeader {
    pub version: u8,                // Version du format de bloc
    pub parent_hash: [u8; 32],      // Hash du bloc parent
    pub timestamp: u64,             // Horodatage Unix en secondes
    pub merkle_root: [u8; 32],      // Racine de l'arbre de Merkle des transactions
    pub height: u64,                // Hauteur du bloc dans la blockchain
    pub nonce: u64,                 // Nonce pour le mining PoW
    pub miner: [u8; 32],            // Adresse du mineur ayant créé le bloc
    pub difficulty: u32,            // Difficulté utilisée pour ce bloc
}

/// Structure complète d'un bloc
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub header: BlockHeader,        // En-tête du bloc
    pub hash: [u8; 32],             // Hash de l'en-tête
    pub transactions: Vec<Transaction>, // Transactions incluses dans le bloc
    pub height: u64,                // Hauteur du bloc (redondant avec header.height pour accès direct)
    pub size: usize,                // Taille du bloc en octets
    pub tx_count: usize,            // Nombre de transactions
    pub total_fees: u64,            // Total des frais de transaction
    pub timestamp: u64,             // Horodatage (redondant avec header.timestamp pour accès direct)
    pub miner: [u8; 32],            // Adresse du mineur (redondant avec header.miner pour accès direct)
    pub difficulty: u32,            // Difficulté (redondant avec header.difficulty pour accès direct)
    pub target: Vec<u8>,            // Cible de difficulté
    
    // Métriques supplémentaires pour l'analyse
    pub state_updates: Option<HashMap<String, Vec<u8>>>, // Changements d'état spécifiques
    pub neural_factor: Option<f32>, // Facteur d'adaptation neuronal appliqué
}

impl Block {
    /// Crée un nouveau bloc
    pub fn new(
        header: BlockHeader,
        transactions: Vec<Transaction>,
        target: Vec<u8>,
    ) -> Self {
        // Calculer la racine de l'arbre de Merkle
        let mut header_with_merkle = header.clone();
        header_with_merkle.merkle_root = Self::calculate_merkle_root(&transactions);
        
        // Calculer le hash du bloc
        let hash = Self::calculate_hash(&header_with_merkle);
        
        // Calculer la taille totale approximative
        let size = std::mem::size_of::<BlockHeader>() + 
                   transactions.iter().map(|tx| tx.size()).sum::<usize>();
        
        // Calculer les frais totaux
        let total_fees = transactions.iter().map(|tx| tx.fee).sum();
        
        Self {
            hash,
            height: header.height,
            tx_count: transactions.len(),
            size,
            total_fees,
            timestamp: header.timestamp,
            miner: header.miner,
            difficulty: header.difficulty,
            header: header_with_merkle,
            transactions,
            target,
            state_updates: None,
            neural_factor: None,
        }
    }
    
    /// Calcule le hash d'un en-tête de bloc
    pub fn calculate_hash(header: &BlockHeader) -> [u8; 32] {
        // Sérialiser l'en-tête
        let serialized = bincode::serialize(header)
            .expect("Échec de la sérialisation de l'en-tête");
        
        // Calculer le hash SHA-256
        let mut hasher = Sha256::new();
        hasher.update(&serialized);
        let result = hasher.finalize();
        
        // Convertir le résultat en tableau de 32 octets
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
    
    /// Calcule la racine de l'arbre de Merkle pour un ensemble de transactions
    pub fn calculate_merkle_root(transactions: &[Transaction]) -> [u8; 32] {
        if transactions.is_empty() {
            return [0u8; 32];
        }
        
        // Extraire les hash de transactions
        let mut hashes: Vec<[u8; 32]> = transactions.iter()
            .map(|tx| tx.id)
            .collect();
        
        // Construire l'arbre de Merkle
        while hashes.len() > 1 {
            if hashes.len() % 2 == 1 {
                // Dupliquer le dernier élément si nombre impair
                hashes.push(*hashes.last().unwrap());
            }
            
            let mut new_hashes = Vec::with_capacity(hashes.len() / 2);
            
            // Combiner les paires de hash
            for i in (0..hashes.len()).step_by(2) {
                let mut hasher = Sha256::new();
                hasher.update(&hashes[i]);
                hasher.update(&hashes[i + 1]);
                let result = hasher.finalize();
                
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&result);
                new_hashes.push(hash);
            }
            
            hashes = new_hashes;
        }
        
        hashes[0]
    }
    
    /// Valide un bloc
    pub fn validate(&self) -> bool {
        // Vérifier le hash du bloc
        let calculated_hash = Self::calculate_hash(&self.header);
        if calculated_hash != self.hash {
            return false;
        }
        
        // Vérifier la racine de Merkle
        let calculated_merkle_root = Self::calculate_merkle_root(&self.transactions);
        if calculated_merkle_root != self.header.merkle_root {
            return false;
        }
        
        // Autres validations...
        true
    }
    
    /// Récupère une transaction par son ID
    pub fn get_transaction(&self, tx_id: &[u8; 32]) -> Option<&Transaction> {
        self.transactions.iter()
            .find(|tx| tx.id == *tx_id)
    }
    
    /// Vérifie si le bloc contient une transaction
    pub fn contains_transaction(&self, tx_id: &[u8; 32]) -> bool {
        self.transactions.iter()
            .any(|tx| tx.id == *tx_id)
    }
    
    /// Récupère les transactions d'un type spécifique
    pub fn get_transactions_by_type(&self, tx_type: u8) -> Vec<&Transaction> {
        self.transactions.iter()
            .filter(|tx| tx.tx_type_code() == tx_type)
            .collect()
    }
    
    /// Calcule le temps de confirmer depuis le bloc précédent
    pub fn time_since_previous_block(&self, prev_block_timestamp: u64) -> u64 {
        if self.timestamp <= prev_block_timestamp {
            return 0;
        }
        self.timestamp - prev_block_timestamp
    }
    
    /// Récupère le miner reward (pas dans les transactions)
    pub fn miner_reward(&self) -> u64 {
        // La récompense de base peut varier selon la hauteur du bloc
        let base_reward = if self.height < 1_000_000 {
            5_000_000_000 // 50 NCH
        } else if self.height < 2_000_000 {
            2_500_000_000 // 25 NCH
        } else if self.height < 4_000_000 {
            1_250_000_000 // 12.5 NCH
        } else {
            625_000_000 // 6.25 NCH
        };
        
        // Ajouter les frais de transaction
        base_reward + self.total_fees
    }
    
    /// Sérialise le bloc en binaire
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Échec de la sérialisation du bloc")
    }
    
    /// Désérialise un bloc à partir de données binaires
    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }
}

/// Bloc de genèse
impl Block {
    pub fn genesis() -> Self {
        let header = BlockHeader {
            version: 1,
            parent_hash: [0; 32],
            timestamp: 1714500000, // Avril 2024
            merkle_root: [0; 32],
            height: 0,
            nonce: 0,
            miner: [0; 32],
            difficulty: 1,
        };
        
        Block::new(header, Vec::new(), vec![255; 32])
    }
}
