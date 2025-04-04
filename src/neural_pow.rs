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
        
        // Caractéristiques de la blockchain
        features.push(state.mempool_size as f32 / 10000.0); // Normalisé
        features.push(state.average_fee_last_100_blocks / state.max_fee_ever);
        features.push(state.hashrate_estimate / 1_000_000_000.0); // En GH/s
        
        // Ajouter d'autres caractéristiques spécifiques à NeuralChain
        features.push(state.network_activity_score);
        features.push(state.node_distribution_entropy);
        
        // Caractéristiques de mining récentes
        features.push(state.avg_block_time_last_100 / 60.0); // En minutes, normalisé
        features.push(state.avg_transactions_per_block_last_100 / 1000.0);
        features.push(state.difficulty_adjustment_factor);
        
        // Caractéristiques économiques
        features.push(state.average_transaction_value / state.max_transaction_value_ever);
        features.push(state.fee_to_reward_ratio);
        features.push(state.utxo_set_size as f32 / 1_000_000.0); // En millions
        
        // Compléter à 256 dimensions avec des zéros
        features.resize(256, 0.0);
        features
    }
    
    /// Analyse les patterns de transaction dans la blockchain
    fn analyze_transaction_patterns(&self, blockchain: &Blockchain) -> Result<TransactionPatterns> {
        // Récupérer les blocs récents
        let recent_blocks = blockchain.get_last_n_blocks(100)?;
        
        // Calculer diverses statistiques sur les transactions
        let total_tx_count = recent_blocks.iter().map(|b| b.transactions.len()).sum::<usize>();
        let avg_tx_per_block = if recent_blocks.is_empty() {
            0.0
        } else {
            total_tx_count as f32 / recent_blocks.len() as f32
        };
        
        // Distribution des tailles de transaction
        let mut tx_sizes = Vec::new();
        for block in &recent_blocks {
            for tx in &block.transactions {
                tx_sizes.push(tx.payload.len());
            }
        }
        
        // Trier pour calculer la médiane et les percentiles
        tx_sizes.sort_unstable();
        
        let patterns = TransactionPatterns {
            avg_tx_per_block,
            median_tx_size: if !tx_sizes.is_empty() {
                tx_sizes[tx_sizes.len() / 2] as f32
            } else {
                0.0
            },
            tx_size_variance: calculate_variance(&tx_sizes),
            tx_temporal_density: calculate_tx_temporal_density(&recent_blocks),
        };
        
        Ok(patterns)
    }
    
    /// Analyse la distribution du mining dans la blockchain
    fn analyze_mining_distribution(&self, blockchain: &Blockchain) -> Result<MiningDistribution> {
        // Récupérer les blocs récents
        let recent_blocks = blockchain.get_last_n_blocks(1000)?;
        
        // Compter les blocs par mineur
        let mut miner_counts = std::collections::HashMap::new();
        for block in &recent_blocks {
            *miner_counts.entry(block.miner.clone()).or_insert(0) += 1;
        }
        
        // Calculer l'indice de Gini pour mesurer l'inégalité de distribution
        let gini_index = calculate_gini_index(&miner_counts);
        
        // Calculer l'entropie de la distribution
        let entropy = calculate_entropy(&miner_counts, recent_blocks.len());
        
        let distribution = MiningDistribution {
            unique_miners: miner_counts.len(),
            gini_index,
            entropy,
            largest_miner_percentage: calculate_largest_percentage(&miner_counts, recent_blocks.len()),
        };
        
        Ok(distribution)
    }
}

impl NeuralMatrix {
    /// Créer une nouvelle matrice neuronale avec des dimensions spécifiées
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Initialisation avec des poids aléatoires
        let mut rng = thread_rng();
        let weights = (0..output_dim)
            .map(|_| (0..input_dim).map(|_| rng.gen::<f32>() * 0.1 - 0.05).collect())
            .collect();
        
        Self {
            weights,
            input_dimension: input_dim,
            output_dimension: output_dim,
        }
    }
    
    /// Traite un vecteur d'entrée à travers la matrice neuronale
    pub fn process(&self, input: &[f32]) -> f32 {
        // Vérifier que l'entrée a la bonne dimension
        assert_eq!(input.len(), self.input_dimension, "Input dimension mismatch");
        
        // Calculer la sortie en utilisant la matrice de poids
        let mut output = vec![0.0; self.output_dimension];
        
        for (i, row) in self.weights.iter().enumerate() {
            let mut sum = 0.0;
            for (j, &weight) in row.iter().enumerate() {
                sum += weight * input[j];
            }
            output[i] = activation_function(sum);
        }
        
        // Réduire à un seul facteur multiplicatif entre 0.5 et 2.0
        // Cela permet d'ajuster la difficulté de manière limitée mais significative
        let avg_output = output.iter().sum::<f32>() / output.len() as f32;
        0.5 + avg_output * 1.5 // Facteur entre 0.5x et 2.0x la difficulté de base
    }
}

// Fonction d'activation sigmoïde
fn activation_function(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Structures auxiliaires pour l'analyse
struct TransactionPatterns {
    avg_tx_per_block: f32,
    median_tx_size: f32,
    tx_size_variance: f32,
    tx_temporal_density: f32,
}

struct MiningDistribution {
    unique_miners: usize,
    gini_index: f32,
    entropy: f32,
    largest_miner_percentage: f32,
}

// Fonctions utilitaires d'analyse statistique
fn calculate_variance(values: &[usize]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<usize>() as f32 / values.len() as f32;
    let variance_sum = values.iter()
        .map(|&x| {
            let diff = x as f32 - mean;
            diff * diff
        })
        .sum::<f32>();
    
    variance_sum / values.len() as f32
}

fn calculate_tx_temporal_density(blocks: &[crate::block::Block]) -> f32 {
    if blocks.len() <= 1 {
        return 1.0;
    }
    
    // Calculer l'écart-type des intervalles de temps entre blocs
    let mut intervals = Vec::with_capacity(blocks.len() - 1);
    for i in 1..blocks.len() {
        let time_diff = blocks[i].timestamp - blocks[i-1].timestamp;
        intervals.push(time_diff as f32);
    }
    
    let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
    let variance = intervals.iter()
        .map(|&x| {
            let diff = x - mean_interval;
            diff * diff
        })
        .sum::<f32>() / intervals.len() as f32;
    
    let std_dev = variance.sqrt();
    
    // Normaliser: plus la déviation est petite, plus la densité est élevée (régulière)
    1.0 / (1.0 + std_dev / mean_interval)
}

fn calculate_gini_index(miner_counts: &std::collections::HashMap<String, i32>) -> f32 {
    if miner_counts.is_empty() {
        return 0.0;
    }
    
    let values: Vec<i32> = miner_counts.values().cloned().collect();
    let n = values.len() as f32;
    
    if n <= 1.0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    for &x in &values {
        for &y in &values {
            sum += (x - y).abs() as f32;
        }
    }
    
    let mean = values.iter().sum::<i32>() as f32 / n;
    sum / (2.0 * n * n * mean)
}

fn calculate_entropy(miner_counts: &std::collections::HashMap<String, i32>, total_blocks: usize) -> f32 {
    if miner_counts.is_empty() || total_blocks == 0 {
        return 0.0;
    }
    
    let total_blocks = total_blocks as f32;
    let mut entropy = 0.0;
    
    for &count in miner_counts.values() {
        let p = count as f32 / total_blocks;
        entropy -= p * p.ln();
    }
    
    entropy
}

fn calculate_largest_percentage(miner_counts: &std::collections::HashMap<String, i32>, total_blocks: usize) -> f32 {
    if miner_counts.is_empty() || total_blocks == 0 {
        return 0.0;
    }
    
    let max_count = miner_counts.values().cloned().max().unwrap_or(0);
    max_count as f32 / total_blocks as f32 * 100.0
}

// Calcule les poids optimaux de la matrice neuronale en fonction des patterns
fn calculate_optimal_weights(
    current_weights: Vec<Vec<f32>>,
    tx_patterns: TransactionPatterns,
    mining_distribution: MiningDistribution
) -> Vec<Vec<f32>> {
    // Copier les poids actuels
    let mut new_weights = current_weights.clone();
    
    // Facteurs d'ajustement basés sur les analyses
    let tx_adjustment = calculate_tx_adjustment(&tx_patterns);
    let mining_adjustment = calculate_mining_adjustment(&mining_distribution);
    
    // Appliquer les ajustements aux poids
    for row in &mut new_weights {
        for weight in row {
            // Ajuster avec une petite perturbation basée sur nos facteurs
            let perturbation = (tx_adjustment + mining_adjustment) / 2.0;
            *weight += perturbation * 0.01; // Petit ajustement progressif
            
            // Limiter les valeurs des poids pour éviter l'explosion
            *weight = weight.clamp(-1.0, 1.0);
        }
    }
    
    new_weights
}

fn calculate_tx_adjustment(patterns: &TransactionPatterns) -> f32 {
    // Ajuster en fonction de la variabilité des transactions
    // Plus de variabilité -> ajustement positif (plus difficile)
    // Moins de variabilité -> ajustement négatif (plus facile)
    let var_factor = (patterns.tx_size_variance / 10000.0).min(1.0);
    let density_factor = (1.0 - patterns.tx_temporal_density) * 2.0;
    
    var_factor + density_factor - 0.5
}

fn calculate_mining_adjustment(distribution: &MiningDistribution) -> f32 {
    // Ajuster en fonction de la centralisation du mining
    // Plus centralisé -> ajustement positif (plus difficile)
    // Plus distribué -> ajustement négatif (plus facile)
    let centralization = distribution.gini_index;
    let diversity = distribution.entropy / (distribution.unique_miners as f32).ln().max(1.0);
    
    centralization - diversity
}
