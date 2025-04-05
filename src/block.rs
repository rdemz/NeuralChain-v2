use crate::transaction::Transaction;
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};

/// Structure d'un bloc de la blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Version du protocole
    pub version: u32,
    /// Hauteur du bloc dans la chaîne
    pub height: u64,
    /// Horodatage Unix
    pub timestamp: u64,
    /// Hash du bloc précédent
    pub prev_hash: Vec<u8>,
    /// Racine de l'arbre de Merkle des transactions
    pub merkle_root: Vec<u8>,
    /// Difficulté cible
    pub difficulty: u32,
    /// Nonce trouvé lors du minage
    pub nonce: u64,
    /// Liste des transactions incluses dans le bloc
    pub transactions: Vec<Transaction>,
    /// Hash du bloc (calculé)
    pub hash: Vec<u8>,
}

impl Block {
    /// Crée un bloc de genèse
    pub fn genesis() -> Self {
        let timestamp = 1649625600; // 2022-04-11 00:00:00 UTC
        let mut genesis = Self {
            version: 1,
            height: 0,
            timestamp,
            prev_hash: vec![0; 32], // Genesis block has no previous hash
            merkle_root: vec![0; 32], // Empty merkle root for genesis block
            difficulty: 1,
            nonce: 0,
            transactions: vec![], // No transactions in genesis block
            hash: vec![],
        };
        
        // Calcul du hash du bloc
        let hash = genesis.calculate_hash();
        genesis.hash = hash;
        genesis
    }
    
    /// Crée un nouveau bloc à partir du bloc précédent et des transactions
    pub fn new(
        prev_block: &Block, 
        transactions: Vec<Transaction>, 
        timestamp: u64, 
        difficulty: u32
    ) -> Self {
        let merkle_root = Self::calculate_merkle_root(&transactions);
        let height = prev_block.height + 1;
        
        let mut block = Self {
            version: 1,
            height,
            timestamp,
            prev_hash: prev_block.hash.clone(),
            merkle_root,
            difficulty,
            nonce: 0,
            transactions,
            hash: vec![],
        };
        
        // Initialiser le hash (sera recalculé pendant le minage)
        let hash = block.calculate_hash();
        block.hash = hash;
        
        block
    }
    
    /// Calcule le hash du bloc
    pub fn calculate_hash(&self) -> Vec<u8> {
        // Structure à hacher: version + height + timestamp + prev_hash + merkle_root + difficulty + nonce
        let mut hasher = Sha256::new();
        
        // Ajouter chaque composant au hash
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.height.to_le_bytes());
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(&self.prev_hash);
        hasher.update(&self.merkle_root);
        hasher.update(&self.difficulty.to_le_bytes());
        hasher.update(&self.nonce.to_le_bytes());
        
        // Calculer le hash final
        hasher.finalize().to_vec()
    }
    
    /// Mine le bloc en cherchant un nonce valide
    pub fn mine(&mut self) -> Result<()> {
        // Valeur cible pour le minage (simplifiée)
        let target = 1u128 << (128 - self.difficulty as u128);
        
        while self.nonce < u64::MAX {
            // Calculer le hash avec le nonce actuel
            self.hash = self.calculate_hash();
            
            // Convertir les 16 premiers octets du hash en u128 pour comparaison
            let mut hash_value = 0u128;
            for (i, &byte) in self.hash.iter().take(16).enumerate() {
                hash_value |= (byte as u128) << ((15 - i) * 8);
            }
            
            // Vérifier si le hash est inférieur à la cible
            if hash_value < target {
                return Ok(());
            }
            
            // Essayer avec un nouveau nonce
            self.nonce += 1;
        }
        
        bail!("Échec du minage: aucun nonce valide trouvé")
    }
    
    /// Vérifie si le hash du bloc est valide par rapport à la difficulté
    pub fn is_hash_valid(&self) -> bool {
        // Recalculer le hash pour vérification
        let hash = self.calculate_hash();
        if hash != self.hash {
            return false;
        }
        
        // Valeur cible pour la difficulté
        let target = 1u128 << (128 - self.difficulty as u128);
        
        // Convertir les 16 premiers octets du hash en u128 pour comparaison
        let mut hash_value = 0u128;
        for (i, &byte) in self.hash.iter().take(16).enumerate() {
            hash_value |= (byte as u128) << ((15 - i) * 8);
        }
        
        // Le hash doit être inférieur à la cible
        hash_value < target
    }
    
    /// Calcule la racine de l'arbre de Merkle des transactions
    pub fn calculate_merkle_root(transactions: &[Transaction]) -> Vec<u8> {
        if transactions.is_empty() {
            // Retourner un hash de zéros pour une liste vide
            return vec![0; 32];
        }
        
        // Obtenir les hashes de chaque transaction
        let mut hashes: Vec<Vec<u8>> = transactions
            .iter()
            .map(|tx| tx.get_id().unwrap_or_else(|_| vec![0; 32]))
            .collect();
        
        // Construire l'arbre jusqu'à obtenir une seule racine
        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            
            // Traiter les paires de hashes
            for chunk in hashes.chunks(2) {
                let mut hasher = Sha256::new();
                hasher.update(&chunk[0]);
                
                // Si le nombre de hashes est impair, dupliquer le dernier
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    hasher.update(&chunk[0]);
                }
                
                next_level.push(hasher.finalize().to_vec());
            }
            
            hashes = next_level;
        }
        
        hashes[0].clone()
    }
    
    /// Vérifie la validité d'un bloc
    pub fn validate(&self, prev_block: Option<&Block>) -> Result<bool> {
        // Vérifier le hash du bloc
        if !self.is_hash_valid() {
            return Ok(false);
        }
        
        // Vérifier la racine de Merkle
        let calculated_merkle_root = Self::calculate_merkle_root(&self.transactions);
        if calculated_merkle_root != self.merkle_root {
            return Ok(false);
        }
        
        // Vérifier la liaison avec le bloc précédent (sauf pour le genesis)
        if self.height > 0 {
            match prev_block {
                Some(prev) => {
                    // Vérifier la continuité de la hauteur
                    if self.height != prev.height + 1 {
                        return Ok(false);
                    }
                    
                    // Vérifier que le hash précédent correspond
                    if self.prev_hash != prev.hash {
                        return Ok(false);
                    }
                },
                None => return Ok(false), // Bloc non-genesis sans prédécesseur
            }
        }
        
        // Vérifier chaque transaction
        for tx in &self.transactions {
            if let Ok(false) = tx.verify_signature() {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
}

/// Fonction utilitaire pour bail (utilisé dans `mine()`)
pub fn bail<T>(msg: &str) -> Result<T> {
    Err(anyhow::anyhow!(msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::{Transaction, TransactionType};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    #[test]
    fn test_genesis_block() {
        let genesis = Block::genesis();
        
        assert_eq!(genesis.height, 0);
        assert_eq!(genesis.prev_hash, vec![0; 32]);
        assert!(genesis.is_hash_valid());
    }
    
    #[test]
    fn test_block_hash() {
        let mut block = Block::genesis();
        let original_hash = block.hash.clone();
        
        // Modifier le nonce devrait changer le hash
        block.nonce = 42;
        block.hash = block.calculate_hash();
        
        assert_ne!(original_hash, block.hash);
        assert!(block.is_hash_valid());
    }
}
