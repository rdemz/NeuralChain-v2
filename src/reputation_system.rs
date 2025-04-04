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
                    } else {
                        None
                    }
                )
                .collect::<Vec<_>>();
            
            if !malicious_events.is_empty() {
                let avg_severity = malicious_events.iter().sum::<u8>() as f32 / malicious_events.len() as f32;
                
                if avg_severity > 8.0 {
                    return PenaltyAction::Blacklist;
                } else if avg_severity > 5.0 {
                    return PenaltyAction::Throttle { factor: 0.5 };
                } else if avg_severity > 3.0 {
                    return PenaltyAction::Warn;
                }
            }
        }
        
        PenaltyAction::None
    }
    
    // Enregistre un événement de mining réussi
    pub fn record_successful_mining(
        &self, 
        node_id: NodeId, 
        block_id: BlockId, 
        difficulty: u32
    ) {
        let event = ReputationEvent::SuccessfulMining {
            block_id,
            difficulty,
            timestamp: Utc::now(),
        };
        
        self.update_node_score(node_id, |score| {
            score.mining_contribution += difficulty as u32;
            score.history.push_back(event.clone());
            
            // Limiter la taille de l'historique
            while score.history.len() > 100 {
                score.history.pop_front();
            }
            
            score.last_update = Utc::now();
        });
    }
    
    // Enregistre une contribution de validation
    pub fn record_validation_contribution(
        &self,
        node_id: NodeId,
        tx_count: u32,
        accuracy: f32
    ) {
        let event = ReputationEvent::ValidationContribution {
            tx_count,
            accuracy,
            timestamp: Utc::now(),
        };
        
        self.update_node_score(node_id, |score| {
            // La contribution augmente en fonction du nombre et de la précision
            let contribution = (tx_count as f32 * accuracy).round() as u32;
            score.transaction_validation += contribution;
            score.history.push_back(event.clone());
            
            while score.history.len() > 100 {
                score.history.pop_front();
            }
            
            score.last_update = Utc::now();
        });
    }
    
    // Enregistre la stabilité réseau d'un nœud
    pub fn record_network_uptime(
        &self,
        node_id: NodeId,
        hours: u32,
        reliability: f32
    ) {
        let event = ReputationEvent::NetworkUptime {
            hours,
            reliability,
            timestamp: Utc::now(),
        };
        
        self.update_node_score(node_id, |score| {
            let contribution = (hours as f32 * reliability).round() as u32;
            score.network_stability += contribution;
            score.history.push_back(event.clone());
            
            while score.history.len() > 100 {
                score.history.pop_front();
            }
            
            score.last_update = Utc::now();
        });
    }
    
    // Enregistre une activité malveillante
    pub fn record_malicious_activity(
        &self,
        node_id: NodeId,
        description: String,
        severity: u8
    ) {
        let event = ReputationEvent::MaliciousActivity {
            description,
            severity,
            timestamp: Utc::now(),
        };
        
        self.update_node_score(node_id, |score| {
            // Réduire le score de base en fonction de la gravité
            let penalty = std::cmp::min(severity, score.base_score);
            score.base_score -= penalty;
            score.history.push_back(event.clone());
            
            while score.history.len() > 100 {
                score.history.pop_front();
            }
            
            score.last_update = Utc::now();
        });
    }
    
    // Enregistre la fourniture de données
    pub fn record_data_provision(
        &self,
        node_id: NodeId,
        size_kb: u32,
        quality: f32
    ) {
        let event = ReputationEvent::DataProvision {
            size_kb,
            quality,
            timestamp: Utc::now(),
        };
        
        self.update_node_score(node_id, |score| {
            let contribution = (size_kb as f32 * quality).round() as u32;
            score.data_provision += contribution;
            score.history.push_back(event.clone());
            
            while score.history.len() > 100 {
                score.history.pop_front();
            }
            
            score.last_update = Utc::now();
        });
    }
    
    // Décroître les scores avec le temps pour encourager la participation continue
    pub fn decay_scores(&self, decay_factor: f32) {
        for mut score in self.node_scores.iter_mut() {
            // Appliquer une dépréciation exponentielle aux contributions
            score.mining_contribution = (score.mining_contribution as f32 * decay_factor) as u32;
            score.transaction_validation = (score.transaction_validation as f32 * decay_factor) as u32;
            score.network_stability = (score.network_stability as f32 * decay_factor) as u32;
            score.data_provision = (score.data_provision as f32 * decay_factor) as u32;
            
            score.last_update = Utc::now();
        }
    }
    
    // Mise à jour du score de réputation d'un nœud
    fn update_node_score<F>(&self, node_id: NodeId, update_fn: F) 
    where 
        F: FnOnce(&mut ReputationScore)
    {
        let mut score = self.node_scores.entry(node_id).or_insert_with(|| {
            // Score initial pour un nouveau nœud
            ReputationScore {
                base_score: 50, // Score initial moyen
                mining_contribution: 0,
                transaction_validation: 0,
                network_stability: 0,
                data_provision: 0,
                history: VecDeque::new(),
                last_update: Utc::now(),
            }
        });
        
        update_fn(&mut score);
    }
    
    // Soumettre une transaction au pool de vérification
    pub async fn submit_transaction_for_verification(
        &self,
        tx: Transaction,
        submitter: NodeId
    ) -> Result<(), String> {
        let mut verification_pool = self.verification_pool.lock().await;
        verification_pool.submit_transaction(tx, submitter);
        Ok(())
    }
    
    // Voter sur une transaction (vérification)
    pub async fn vote_on_transaction(
        &self,
        tx_id: Uuid,
        voter: NodeId,
        is_valid: bool,
        confidence: f32
    ) -> Result<(), String> {
        let mut verification_pool = self.verification_pool.lock().await;
        
        let vote = if is_valid {
            VerificationVote::Valid(confidence)
        } else {
            VerificationVote::Invalid(confidence)
        };
        
        verification_pool.vote_on_transaction(tx_id, voter, vote);
        Ok(())
    }
    
    // Vérifier le statut d'une transaction
    pub async fn check_transaction_status(&self, tx_id: &Uuid) -> TransactionStatus {
        let mut verification_pool = self.verification_pool.lock().await;
        verification_pool.check_transaction_status(tx_id)
    }
}

// Consensus de réputation
impl ReputationConsensus {
    pub fn new() -> Self {
        Self {
            recent_reports: HashMap::new(),
            vote_weights: HashMap::new(),
            last_update: Utc::now(),
        }
    }
    
    // Traiter un nouveau rapport de réputation
    pub fn process_report(&mut self, report: ReputationReport) -> bool {
        let reports = self.recent_reports
            .entry(report.subject.clone())
            .or_insert_with(Vec::new);
        
        // Ajouter le rapport à la liste
        reports.push(report.clone());
        
        // Nettoyer les anciens rapports (plus de 24h)
        let cutoff = Utc::now() - Duration::hours(24);
        reports.retain(|r| r.timestamp > cutoff);
        
        // Vérifier s'il y a consensus
        let event_counts = self.count_events(reports);
        let total_reports = reports.len();
        
        // Si plus de 66% des rapports sont du même type, il y a consensus
        let threshold = 0.66;
        
        for (event_type, count) in event_counts {
            if count as f32 / total_reports as f32 >= threshold {
                return true;
            }
        }
        
        false
    }
    
    // Compter les types d'événements dans les rapports
    fn count_events(&self, reports: &[ReputationReport]) -> HashMap<ReputationEventType, usize> {
        let mut counts = HashMap::new();
        
        for report in reports {
            *counts.entry(report.event_type.clone()).or_insert(0) += 1;
        }
        
        counts
    }
}

// Système de vérification collective des transactions
pub struct VerificationPool {
    pending_transactions: HashMap<Uuid, PendingTransaction>,
    verification_votes: HashMap<Uuid, HashMap<NodeId, VerificationVote>>,
}

#[derive(Clone, Debug)]
pub struct PendingTransaction {
    pub transaction: Transaction,
    pub submitter: NodeId,
    pub submission_time: DateTime<Utc>,
    pub state: TransactionStatus,
}

#[derive(Clone, Debug)]
pub enum VerificationVote {
    Valid(f32),   // Avec niveau de confiance (0.0-1.0)
    Invalid(f32), // Avec niveau de confiance (0.0-1.0)
}

impl VerificationPool {
    pub fn new() -> Self {
        Self {
            pending_transactions: HashMap::new(),
            verification_votes: HashMap::new(),
        }
    }
    
    // Soumettre une transaction à vérifier
    pub fn submit_transaction(&mut self, tx: Transaction, submitter: NodeId) {
        let tx_id = tx.id;
        self.pending_transactions.insert(tx_id, PendingTransaction {
            transaction: tx,
            submitter,
            submission_time: Utc::now(),
            state: TransactionStatus::Pending,
        });
        
        self.verification_votes.insert(tx_id, HashMap::new());
    }
    
    // Voter sur une transaction (vérification)
    pub fn vote_on_transaction(&mut self, tx_id: Uuid, voter: NodeId, vote: VerificationVote) {
        if let Some(votes) = self.verification_votes.get_mut(&tx_id) {
            votes.insert(voter, vote);
        }
    }
    
    // Vérifier si une transaction est validée
    pub fn check_transaction_status(&mut self, tx_id: &Uuid) -> TransactionStatus {
        if let Some(pending_tx) = self.pending_transactions.get(tx_id) {
            return pending_tx.state.clone();
        }
        
        if let Some(votes) = self.verification_votes.get(tx_id) {
            let total_votes = votes.len();
            if total_votes < 3 {
                return TransactionStatus::Pending;
            }
            
            let positive_votes = votes.values()
                .filter(|v| matches!(v, VerificationVote::Valid(_)))
                .count();
            
            let consensus_threshold = 0.66; // 66%
            let consensus_ratio = positive_votes as f32 / total_votes as f32;
            
            if consensus_ratio >= consensus_threshold {
                if let Some(pending_tx) = self.pending_transactions.get_mut(tx_id) {
                    pending_tx.state = TransactionStatus::Verified;
                }
                return TransactionStatus::Verified;
            } else if total_votes >= 5 {
                if let Some(pending_tx) = self.pending_transactions.get_mut(tx_id) {
                    pending_tx.state = TransactionStatus::Rejected;
                }
                return TransactionStatus::Rejected;
            } else {
                return TransactionStatus::Pending;
            }
        }
        
        TransactionStatus::Pending
    }
    
    // Nettoyer les transactions expirées
    pub fn clean_expired_transactions(&mut self) -> Vec<Uuid> {
        let now = Utc::now();
        let expiry = Duration::hours(6); // 6 heures d'expiration
        let mut expired = Vec::new();
        
        for (tx_id, pending) in &mut self.pending_transactions {
            if now - pending.submission_time > expiry {
                pending.state = TransactionStatus::Expired;
                expired.push(*tx_id);
            }
        }
        
        expired
    }
}

// Fonction utilitaire pour normaliser les contributions
fn normalize_contribution(value: u32) -> f32 {
    // Fonction sigmoïde pour normaliser les valeurs entre 0 et 100
    let x = value as f32 / 1000.0; // Échelle arbitraire
    100.0 / (1.0 + (-x).exp())
}
