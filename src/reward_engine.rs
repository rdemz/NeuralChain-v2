//! Reward engine for NeuralChain-v2
//! Calculates rewards based on reputation and AI analysis

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::transaction::{Transaction, TransactionType};
use crate::account::Account;

/// Activity type for reward calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivityType {
    /// Block validation/mining
    BlockValidation,
    /// Transaction validation
    TransactionValidation,
    /// Content creation/contribution
    ContentContribution,
    /// Network support
    Support,
    /// Smart contract execution
    ContractExecution,
    /// Governance participation
    Governance,
}

/// Reward calculation engine
pub struct RewardEngine {
    /// Base reward amounts per activity
    base_rewards: HashMap<ActivityType, u64>,
    /// Reputation coefficient for reward calculation
    reputation_coefficient: f64,
    /// Intelligence factor (for AI-based rewards)
    intelligence_factor: f64,
    /// Reward adjustment history
    adjustments: Vec<(u64, f64)>, // (timestamp, adjustment factor)
    /// Maximum rewards per day per account
    daily_reward_caps: HashMap<ActivityType, u64>,
    /// Account reward tracking
    account_rewards: HashMap<Vec<u8>, HashMap<ActivityType, (u64, u64)>>, // public_key -> (activity -> (amount, timestamp))
}

impl RewardEngine {
    /// Create a new reward engine with default settings
    pub fn new() -> Self {
        let mut base_rewards = HashMap::new();
        base_rewards.insert(ActivityType::BlockValidation, 50);
        base_rewards.insert(ActivityType::TransactionValidation, 5);
        base_rewards.insert(ActivityType::ContentContribution, 20);
        base_rewards.insert(ActivityType::Support, 10);
        base_rewards.insert(ActivityType::ContractExecution, 15);
        base_rewards.insert(ActivityType::Governance, 30);
        
        let mut daily_reward_caps = HashMap::new();
        daily_reward_caps.insert(ActivityType::BlockValidation, 1000);
        daily_reward_caps.insert(ActivityType::TransactionValidation, 500);
        daily_reward_caps.insert(ActivityType::ContentContribution, 200);
        daily_reward_caps.insert(ActivityType::Support, 100);
        daily_reward_caps.insert(ActivityType::ContractExecution, 300);
        daily_reward_caps.insert(ActivityType::Governance, 100);
        
        RewardEngine {
            base_rewards,
            reputation_coefficient: 0.002, // 0.2% per reputation point
            intelligence_factor: 1.0,
            adjustments: Vec::new(),
            daily_reward_caps,
            account_rewards: HashMap::new(),
        }
    }
    
    /// Calculate reward for an activity
    pub fn calculate_reward(
        &mut self,
        account: &Account,
        activity: ActivityType,
        quality_score: u32,
        energy_contribution: u32,
    ) -> u64 {
        let base_reward = *self.base_rewards.get(&activity).unwrap_or(&0);
        if base_reward == 0 {
            return 0;
        }
        
        // Check daily reward cap
        if self.would_exceed_daily_cap(account, activity, base_reward) {
            return 0;
        }
        
        // Calculate reputation bonus (higher reputation gives higher rewards)
        let reputation_multiplier = 1.0 + (account.reputation as f64 * self.reputation_coefficient);
        
        // Quality multiplier (0.5-2.0x)
        let quality_multiplier = 0.5 + (quality_score as f64 / 100.0) * 1.5;
        
        // Energy contribution bonus
        let energy_multiplier = 1.0 + (energy_contribution as f64 / 200.0);
        
        // Network adjustment factor
        let network_factor = self.get_network_adjustment_factor();
        
        // Calculate total reward
        let total_reward = (base_reward as f64 * 
                           reputation_multiplier * 
                           quality_multiplier * 
                           energy_multiplier * 
                           network_factor *
                           self.intelligence_factor) as u64;
                           
        // Record this reward
        self.record_account_reward(account, activity, total_reward);
                           
        total_reward
    }
    
    /// Create a reward transaction
    pub fn create_reward_transaction(
        &self,
        recipient_public_key: &[u8],
        amount: u64,
        activity: ActivityType,
    ) -> Transaction {
        let mut transaction = Transaction::new(&[], recipient_public_key, amount);
        transaction.transaction_type = TransactionType::Reward;
        
        // Add activity data
        let activity_code = match activity {
            ActivityType::BlockValidation => 1,
            ActivityType::TransactionValidation => 2,
            ActivityType::ContentContribution => 3,
            ActivityType::Support => 4,
            ActivityType::ContractExecution => 5,
            ActivityType::Governance => 6,
        };
        
        transaction.data = vec![activity_code];
        
        transaction
    }
    
    /// Check if a new reward would exceed the daily cap for an account
    fn would_exceed_daily_cap(&self, account: &Account, activity: ActivityType, new_amount: u64) -> bool {
        let cap = self.daily_reward_caps.get(&activity).cloned().unwrap_or(u64::MAX);
        
        if let Some(activity_map) = self.account_rewards.get(&account.public_key) {
            if let Some((total, timestamp)) = activity_map.get(&activity) {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                    
                // If this reward is from today (last 24 hours)
                if now - timestamp <= 86400 {
                    return total.saturating_add(new_amount) > cap;
                }
            }
        }
        
        false
    }
    
    /// Record an account reward
    fn record_account_reward(&mut self, account: &Account, activity: ActivityType, amount: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        let activity_map = self.account_rewards
            .entry(account.public_key.clone())
            .or_insert_with(HashMap::new);
            
        let entry = activity_map
            .entry(activity)
            .or_insert((0, now));
            
        if now - entry.1 > 86400 {
            // Reset if more than 24 hours old
            *entry = (amount, now);
        } else {
            // Add to existing amount if within 24 hours
            entry.0 = entry.0.saturating_add(amount);
            entry.1 = now; // Update timestamp
        }
    }
    
    /// Add a network adjustment factor
    pub fn add_network_adjustment(&mut self, adjustment_factor: f64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        self.adjustments.push((now, adjustment_factor));
        
        // Keep only recent adjustments (last 7 days)
        self.adjustments.retain(|(ts, _)| now - ts <= 7 * 86400);
    }
    
    /// Get the current network adjustment factor
    fn get_network_adjustment_factor(&self) -> f64 {
        if self.adjustments.is_empty() {
            return 1.0;
        }
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        // Calculate weighted average of recent adjustments
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        
        for (timestamp, factor) in &self.adjustments {
            let age = now - timestamp;
            if age <= 86400 {
                // Adjustments from last 24 hours have full weight
                let weight = 1.0;
                total_weight += weight;
                weighted_sum += factor * weight;
            } else {
                // Older adjustments have diminishing weight
                let weight = 1.0 - (age - 86400) as f64 / (6 * 86400) as f64;
                if weight > 0.0 {
                    total_weight += weight;
                    weighted_sum += factor * weight;
                }
            }
        }
        
        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            1.0
        }
    }
    
    /// Set intelligence factor for AI-based rewards
    pub fn set_intelligence_factor(&mut self, factor: f64) {
        if factor >= 0.5 && factor <= 2.0 {
            self.intelligence_factor = factor;
        }
    }
    
    /// Update base rewards for an activity
    pub fn update_base_reward(&mut self, activity: ActivityType, amount: u64) {
        self.base_rewards.insert(activity, amount);
    }
    
    /// Update daily reward cap for an activity
    pub fn update_daily_cap(&mut self, activity: ActivityType, cap: u64) {
        self.daily_reward_caps.insert(activity, cap);
    }
    
    /// Calculate AI-based quality score using transaction history and network data
    pub fn calculate_ai_quality_score(
        &self,
        account: &Account,
        recent_activities: &[ActivityType],
        network_contribution: u32,
    ) -> u32 {
        // This would normally involve a complex AI model
        // For this example, we'll use a simplified formula
        
        // Weight recent activities
        let activity_score = recent_activities.iter().map(|activity| {
            match activity {
                ActivityType::BlockValidation => 5,
                ActivityType::TransactionValidation => 2,
                ActivityType::ContentContribution => 4,
                ActivityType::Support => 3,
                ActivityType::ContractExecution => 3,
                ActivityType::Governance => 4,
            }
        }).sum::<u32>();
        
        // Account reputation factor
        let reputation_factor = std::cmp::min(account.reputation, 1000) as u32;
        
        // Network contribution factor
        let contribution_factor = std::cmp::min(network_contribution, 500);
        
        // Calculate final score (0-100)
        let raw_score = activity_score * 2 + reputation_factor / 20 + contribution_factor / 10;
        std::cmp::min(raw_score, 100)
    }
    
    /// Apply neural analysis to optimize rewards (simulated AI component)
    pub fn neural_optimization(&mut self, network_data: &[u8], current_epoch: u64) -> f64 {
        // This would be a complex AI/ML model in production
        // Here we'll use a simplified simulation based on network data

        // Extract indicators from network data (in a real system, this would be actual metrics)
        let network_health: f64 = if !network_data.is_empty() {
            (network_data[0] as f64) / 255.0
        } else {
            0.5 // Default health
        };
        
        let participation_rate: f64 = if network_data.len() > 1 {
            (network_data[1] as f64) / 255.0
        } else {
            0.4 // Default participation
        };
        
        let energy_efficiency: f64 = if network_data.len() > 2 {
            (network_data[2] as f64) / 255.0
        } else {
            0.6 // Default efficiency
        };
        
        // Weight factors
        let health_weight = 0.4;
        let participation_weight = 0.3;
        let efficiency_weight = 0.3;
        
        // Cyclical factor based on epoch (simulates network rhythm)
        let epoch_factor = 0.8 + 0.2 * ((current_epoch % 100) as f64 / 100.0);
        
        // Calculate optimal intelligence factor
        let optimal_factor = (
            network_health * health_weight +
            participation_rate * participation_weight +
            energy_efficiency * efficiency_weight
        ) * epoch_factor;
        
        // Clamp to allowed range and apply
        let clamped_factor = optimal_factor.max(0.5).min(2.0);
        self.set_intelligence_factor(clamped_factor);
        
        clamped_factor
    }
    
    /// Get total rewards distributed to an account in the last 24 hours
    pub fn get_account_daily_rewards(&self, account_public_key: &[u8]) -> u64 {
        if let Some(activities) = self.account_rewards.get(account_public_key) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
                
            activities.values()
                .filter(|(_, timestamp)| now - timestamp <= 86400)
                .map(|(amount, _)| amount)
                .sum()
        } else {
            0
        }
    }
    
    /// Get account activity statistics for analysis
    pub fn get_account_activity_stats(&self, account_public_key: &[u8]) -> HashMap<ActivityType, u64> {
        let mut stats = HashMap::new();
        
        if let Some(activities) = self.account_rewards.get(account_public_key) {
            for (activity, (amount, _)) in activities {
                *stats.entry(*activity).or_insert(0) += amount;
            }
        }
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reward_calculation() {
        let mut reward_engine = RewardEngine::new();
        let account = Account::new("test_user");
        
        let reward = reward_engine.calculate_reward(
            &account,
            ActivityType::BlockValidation,
            80,  // quality score
            150, // energy contribution
        );
        
        // Should be non-zero
        assert!(reward > 0);
        
        // Second reward should be tracked
        let reward2 = reward_engine.calculate_reward(
            &account,
            ActivityType::BlockValidation,
            80,
            150,
        );
        
        assert!(reward2 > 0);
    }
    
    #[test]
    fn test_neural_optimization() {
        let mut reward_engine = RewardEngine::new();
        
        // Test with different network states
        let healthy_network = [200, 180, 160]; // High health, participation, efficiency
        let factor1 = reward_engine.neural_optimization(&healthy_network, 1);
        
        let struggling_network = [100, 90, 80];
        let factor2 = reward_engine.neural_optimization(&struggling_network, 2);
        
        // Healthy network should have higher intelligence factor
        assert!(factor1 > factor2);
    }
    
    #[test]
    fn test_reward_caps() {
        let mut reward_engine = RewardEngine::new();
        let account = Account::new("test_user");
        
        // Set a very low cap
        reward_engine.update_daily_cap(ActivityType::BlockValidation, 10);
        
        // First reward should work
        let reward1 = reward_engine.calculate_reward(
            &account,
            ActivityType::BlockValidation,
            80,
            100,
        );
        
        assert!(reward1 > 0);
        
        // With a low cap, this should hit the limit and return 0
        let reward2 = reward_engine.calculate_reward(
            &account,
            ActivityType::BlockValidation,
            80,
            100,
        );
        
        assert_eq!(reward2, 0);
    }
}
