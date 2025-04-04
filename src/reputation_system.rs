use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::Mutex;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration};

use crate::blockchain::{BlockId, Blockchain};
use crate::transaction::Transaction;

/// Type d'identifiant de nœud (pour l'instant une chaîne, pourrait être un type plus complexe)
pub type NodeId = String;

/// Système de réputation à plusieurs niveaux
pub struct ReputationSystem {
    node_scores: DashMap<NodeId, ReputationScore>,
    transaction_records: DashMap<Uuid, TransactionRecord>,
    reputation_consensus: Arc<Mutex<ReputationConsensus>>,
    verification_pool: Arc<Mutex<VerificationPool>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReputationScore {
    // Score de base entre 0 et 100
    pub base_score: u8,
    // Contributions spécifiques dans différents domaines
    pub mining_contribution: u32,
    pub transaction_validation: u32,
    pub network_stability: u32,
    pub data_provision: u32,
    // Historique des actions
    pub history: VecDeque<ReputationEvent>,
    // Dernier mise à jour
    pub last_update: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReputationEvent {
    SuccessfulMining { block_id: BlockId, difficulty: u32, timestamp: DateTime<Utc> },
    ValidationContribution { tx_count: u32, accuracy: f32, timestamp: DateTime<Utc> },
    NetworkUptime { hours: u32, reliability: f32, timestamp: DateTime<Utc> },
    MaliciousActivity { description: String, severity: u8, timestamp: DateTime<Utc> },
    DataProvision { size_kb: u32, quality: f32, timestamp: DateTime<Utc> },
}

/// Privilèges accordés à un nœud en fonction de sa réputation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodePrivileges {
    pub transaction_priority: TransactionPriority,
    pub fee_discount: f32,        // Réduction sur les frais (0.0-1.0)
    pub vote_weight: u8,          // Poids de vote dans les décisions de gouvernance
    pub data_access_level: DataAccessLevel,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TransactionPriority {
    Highest,  // Traité avant tout le monde
    High,     // Traité avant les transactions normales
    Normal,   // Priorité standard
    Low,      // Traité après les autres
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataAccessLevel {
    Complete,  // Accès à toutes les données analytiques
    Extended,  // Accès étendu aux données
    Standard,  // Accès standard
    Basic,     // Accès limité
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PenaltyAction {
    None,      // Aucune pénalité
    Warn,      // Avertissement
    Throttle { factor: f32 }, // Limiter les fonctionnalités
    Blacklist, // Bloquer temporairement
}

/// Enregistrement d'une transaction pour le système de réputation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransactionRecord {
    pub tx_id: Uuid,
    pub submitter: NodeId,
    pub verifiers: Vec<NodeId>,
    pub submission_time: DateTime<Utc>,
    pub verification_time: Option<DateTime<Utc>>,
    pub inclusion_time: Option<DateTime<Utc>>,
    pub final_status: TransactionStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TransactionStatus {
    Pending,
    Verified,
    Rejected,
    Included,
    Expired,
}

/// Système de consensus de réputation
pub struct ReputationConsensus {
    recent_reports: HashMap<NodeId, Vec<ReputationReport>>,
    vote_weights: HashMap<NodeId, f32>,
    last_update: DateTime<Utc>,
}

#[derive(Clone, Debug)]
struct ReputationReport {
    reporter: NodeId,
    subject: NodeId,
    event_type: ReputationEventType,
    evidence: Vec<u8>,
    timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ReputationEventType {
    Positive,
    Negative,
    Neutral,
}

impl ReputationSystem {
    pub fn new() -> Self {
        Self {
            node_scores: DashMap::new(),
            transaction_records: DashMap::new(),
            reputation_consensus: Arc::new(Mutex::new(ReputationConsensus::new())),
            verification_pool: Arc::new(Mutex::new(VerificationPool::new())),
        }
    }
    
    // Calcul du score de réputation global d'un nœud
    pub fn calculate_node_score(&self, node_id: &NodeId) -> Option<f32> {
        if let Some(score) = self.node_scores.get(node_id) {
            let base = score.base_score as f32;
            let mining = normalize_contribution(score.mining_contribution);
            let validation = normalize_contribution(score.transaction_validation);
            let stability = normalize_contribution(score.network_stability);
            let data = normalize_contribution(score.data_provision);
            
            // Formule pondérée pour le score global
            let global_score = 
                (base * 0.4) + 
                (mining * 0.2) + 
                (validation * 0.2) + 
                (stability * 0.1) + 
                (data * 0.1);
                
            Some(global_score)
        } else {
            None
        }
    }
    
    // Détermine les privilèges du nœud en fonction de sa réputation
    pub fn determine_node_privileges(&self, node_id: &NodeId) -> NodePrivileges {
        let score = self.calculate_node_score(node_id).unwrap_or(0.0);
        
        if score >= 90.0 {
            NodePrivileges {
                transaction_priority: TransactionPriority::Highest,
                fee_discount: 0.50, // 50% de réduction sur les frais
                vote_weight: 5,     // 5x le poids de vote standard
                data_access_level: DataAccessLevel::Complete,
            }
        } else if score >= 75.0 {
            NodePrivileges {
                transaction_priority: TransactionPriority::High,
                fee_discount: 0.25, // 25% de réduction
                vote_weight: 3,
                data_access_level: DataAccessLevel::Extended,
            }
        } else if score >= 50.0 {
            NodePrivileges {
                transaction_priority: TransactionPriority::Normal,
                fee_discount: 0.10,
                vote_weight: 1,
                data_access_level: DataAccessLevel::Standard,
            }
        } else {
            NodePrivileges {
                transaction_priority: TransactionPriority::Low,
                fee_discount: 0.0,
                vote_weight: 1,
                data_access_level: DataAccessLevel::Basic,
            }
        }
    }
    
    // Vérifie si un nœud doit être pénalisé pour comportement suspect
    pub fn check_and_penalize(&self, node_id: &NodeId) -> PenaltyAction {
        if let Some(score) = self.node_scores.get(node_id) {
            // Détecter les comportements malveillants dans l'historique
            let malicious_events = score.history.iter()
                .filter_map(|event| 
                    if let ReputationEvent::MaliciousActivity { severity, .. } = event {
                        Some(*severity)
                 ▋
