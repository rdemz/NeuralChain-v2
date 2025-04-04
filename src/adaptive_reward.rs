use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::Result;

/// Système de récompense adaptative pour NeuralChain
pub struct AdaptiveReward {
    // Paramètres du système de récompense
    base_reward: u64,
    difficulty_multiplier: f64,
    network_participation_factor: f64,
    last_adjustments: Vec<f64>,
}

impl AdaptiveReward {
    /// Crée une nouvelle instance du système de récompense adaptative
    pub fn new(base_reward: u64) -> Self {
        Self {
            base_reward,
            difficulty_multiplier: 1.0,
            network_participation_factor: 1.0,
            last_adjustments: Vec::new(),
        }
    }

    /// Calcule la récompense de bloc en fonction des paramètres du réseau
    pub async fn calculate_block_reward(&self, block_difficulty: u64, network_participation: f64) -> u64 {
        let reward = self.base_reward as f64
            * self.difficulty_multiplier
            * (block_difficulty as f64 / 1_000_000.0).min(10.0)
            * self.network_participation_factor
            * network_participation.max(0.5).min(1.5);
        
        reward.round() as u64
    }

    /// Ajuste les paramètres de récompense en fonction des conditions du réseau
    pub async fn adjust_parameters(&mut self, block_time_ms: u64, target_block_time_ms: u64) {
        // Ajustement basé sur le temps de bloc
        let time_ratio = target_block_time_ms as f64 / block_time_ms.max(1) as f64;
        let adjustment_factor = time_ratio.max(0.9).min(1.1);
        
        // Mettre à jour le multiplicateur de difficulté
        self.difficulty_multiplier *= adjustment_factor;
        
        // Limiter le multiplicateur
        self.difficulty_multiplier = self.difficulty_multiplier.max(0.5).min(2.0);
        
        // Enregistrer l'ajustement
        self.last_adjustments.push(adjustment_factor);
        if self.last_adjustments.len() > 100 {
            self.last_adjustments.remove(0);
        }
    }
    
    /// Obtient les statistiques d'ajustement
    pub fn get_adjustment_statistics(&self) -> (f64, f64) {
        if self.last_adjustments.is_empty() {
            return (1.0, 0.0);
        }
        
        let average: f64 = self.last_adjustments.iter().sum::<f64>() / self.last_adjustments.len() as f64;
        
        // Calculer l'écart-type
        let variance = self.last_adjustments.iter()
            .map(|&x| (x - average).powi(2))
            .sum::<f64>() / self.last_adjustments.len() as f64;
        
        let std_dev = variance.sqrt();
        
        (average, std_dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_calculate_block_reward() {
        let reward_system = AdaptiveReward::new(100);
        
        // Vérifier la récompense pour différents niveaux de difficulté
        let reward = reward_system.calculate_block_reward(1_000_000, 1.0).await;
        assert_eq!(reward, 100);
        
        // Vérifier que la difficulté élevée augmente la récompense
        let reward_high_diff = reward_system.calculate_block_reward(2_000_000, 1.0).await;
        assert!(reward_high_diff > reward);
    }
    
    #[tokio::test]
    async fn test_adjust_parameters() {
        let mut reward_system = AdaptiveReward::new(100);
        
        // Tester l'ajustement quand le temps de bloc est plus long que prévu
        reward_system.adjust_parameters(12000, 10000).await;
        assert!(reward_system.difficulty_multiplier < 1.0);
        
        // Tester l'ajustement quand le temps de bloc est plus court que prévu
        reward_system.adjust_parameters(8000, 10000).await;
        assert!(reward_system.difficulty_multiplier > 0.9);
    }
}
