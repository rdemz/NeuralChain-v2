use std::sync::Arc;
use sha2::{Sha256, Digest};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use rand::{Rng, thread_rng};

use crate::block::BlockHeader;
use crate::blockchain::{Blockchain, BlockchainState};
use crate::utils::time::{normalized_time_of_day, normalized_day_of_week};
use crate::monitoring::BLOCK_MINING_TIME;
use anyhow::Result;

/// Système de preuve de travail neuronal adaptatif
pub struct NeuralPoW {
    difficulty_base: u32,
    neural_matrix: Arc<NeuralMatrix>,
    adaptation_period: u64,  // Nombre de blocs entre les ajustements
    last_adaptation_height: u64,
}

/// Matrice neuronale pour l'adaptation de la difficulté
pub struct NeuralMatrix {
    // Matrice de poids qui évolue avec l'historique de la blockchain
    weights: Vec<Vec<f32>>,
    input_dimension: usize,
    output_dimension: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoWResult {
    pub nonce: u64,
    pub hash: Vec<u8>,
    pub target: Vec<u8>,
    pub difficulty: f64,
    pub mining_time_ms: u64,
}

impl NeuralPoW {
    /// Crée une nouvelle instance du système PoW neuronal
    pub fn new(difficulty: u32) -> Self {
        // Initialisation avec une matrice de base qui évoluera
        let neural_matrix = NeuralMatrix::new(256, 32);
        
        Self {
            difficulty_base: difficulty,
            neural_matrix: Arc::new(neural_matrix),
            adaptation_period: 2016, // ~2 semaines comme Bitcoin
            last_adaptation_height: 0,
        }
    }
    
    /// Calcule la cible en fonction de l'en-tête du bloc et de l'état de la blockchain
    pub fn calculate_target(&self, block_header: &BlockHeader, blockchain_state: &BlockchainState) -> Vec<u8> {
        // Extraire des caractéristiques de l'état de la blockchain
        let features = self.extract_blockchain_features(blockchain_state);
        
        // Passer ces caractéristiques à travers la matrice neuronale
        let neural_factor = self.neural_matrix.process(&features);
        
        // Ajuster la difficulté de base avec le facteur neural
        let adaptive_difficulty = (self.difficulty_base as f32 * neural_factor).round() as u32;
        
        // Calculer la cible comme dans Bitcoin mais avec notre difficulté adaptative
        self.calculate_pow_target(adaptive_difficulty)
    }
    
    /// Vérifie si un hash est valide par rapport à une cible
    pub fn validate_block_hash(&self, hash: &[u8], target: &[u8]) -> bool {
        // Vérification que le hash est inférieur à la cible
        for (h, t) in hash.iter().zip(target.iter()) {
            if h > t { return false; }
            if h < t { return true; }
        }
        true
    }
    
    /// Mine un bloc en cherchant un nonce valide
    pub fn mine_block(&self, block_header: &[u8], target: &[u8], max_nonce: u64) -> Option<PoWResult> {
        let start_time = std::time::Instant::now();
        
        // Mining parallélisé avec la bibliothèque rayon
        let result = (0..max_nonce).into_par_iter().find_map(|nonce| {
            let hash = self.hash_with_nonce(block_header, nonce);
            if self.validate_block_hash(&hash, target) {
                Some((nonce, hash))
            } else {
                None
            }
        });
        
        if let Some((nonce, hash)) = result {
            let mining_time_ms = start_time.elapsed().as_millis() as u64;
            
            // Enregistrer la métrique
            BLOCK_MINING_TIME.record(mining_time_ms as f64);
            
            return Some(PoWResult {
                nonce,
                hash,
                target: target.to_vec(),
                difficulty: self.calculate_difficulty(target),
                mining_time_ms,
            });
        }
        
        None
    }
    
    /// Met à jour les poids neuronaux en fonction des données de la blockchain
    pub fn update_neural_weights(&mut self, blockchain: &Blockchain) -> Result<()> {
        let current_height = blockchain.height();
        
        // Vérifier si nous avons atteint la période d'adaptation
        if current_height < self.last_adaptation_height + self.adaptation_period {
            return Ok(());
        }
        
        // Analyser l'historique récent pour détecter les patterns d'activité
        let transaction_patterns = self.analyze_transaction_patterns(blockchain)?;
        let mining_distribution = self.analyze_mining_distribution(blockchain)?;
        
        // Créer une nouvelle matrice avec les poids calculés
        let new_matrix = NeuralMatrix {
            weights: calculate_optimal_weights(
                self.neural_matrix.weights.clone(),
                transaction_patterns,
                mining_distribution
            ),
            input_dimension: self.neural_matrix.input_dimension,
            output_dimension: self.neural_matrix.output_dimension,
        };
        
        // Remplacer l'ancienne matrice
        self.neural_matrix = Arc::new(new_matrix);
        self.last_adaptation_height = current_height;
        
        Ok(())
    }
    
    /// Calcule le hash d'un en-tête avec un nonce spécifique
    fn hash_with_nonce(&self, header: &[u8], nonce: u64) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(header);
        hasher.update(nonce.to_le_bytes());
        
        // Double hash SHA-256 comme Bitcoin
        let first_hash = hasher.finalize();
        let mut second_hasher = Sha256::new();
        second_hasher.update(first_hash);
        
        second_hasher.finalize().to_vec()
    }
    
    /// Calcule la cible PoW à partir de la difficulté
    fn calculate_pow_target(&self, difficulty: u32) -> Vec<u8> {
        // Format similaire à Bitcoin : 256 bits avec des zéros en fonction de la difficulté
        let mut target = vec![0xff; 32]; // 32 octets = 256 bits
        
        // Plus la difficulté est élevée, plus il y a de zéros au début
        let leading_zeros = std::cmp::min(difficulty, 256) as usize;
        let full_bytes = leading_zeros / 8;
        let remaining_bits = leading_zeros % 8;
        
        // Mettre à zéro les octets complets
        for i in 0..full_bytes {
            target[i] = 0;
        }
        
        // Ajuster l'octet partiel si nécessaire
        if full_bytes < 32 && remaining_bits > 0 {
            target[full_bytes] = 0xff >> remaining_bits;
        }
        
        target
    }
    
    /// Calcule la difficulté à partir de la cible
    fn calculate_difficulty(&self, target: &[u8]) -> f64 {
        // Compter les zéros de poids fort (MSB)
        let mut leading_zeros = 0;
        for &byte in target {
            if byte == 0 {
                leading_zeros += 8;
            } else {
                // Compter les bits à zéro dans l'octet non-nul
                let mut mask = 0x80; // 10000000 en binaire
                while mask > 0 && (byte & mask) == 0 {
                    leading_zeros += 1;
                    mask >>= 1;
                }
                break;
            }
        }
        
        // Convertir en difficulté (approximation)
        2f64.powi(leading_zeros as i32)
    }
    
    /// Extraction de caractéristiques de la blockchain pour l'adaptation
    fn extract_blockchain_features(&self, state: &BlockchainState) -> Vec<f32> {
        let mut features = Vec::with_capacity(256);
        
        // Caractéristiques temporelles (heure de la journée, jour de la semaine)
        features.push(normalized_time_of_day());
        features.push(normalized_day_of_week());
        
        // Caractéristiques de la* ▋
