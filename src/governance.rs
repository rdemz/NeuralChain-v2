use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use uuid::Uuid;
use thiserror::Error;
use std::sync::Arc;
use tokio::sync::Mutex;
use dashmap::DashMap;
use std::cmp::Ordering;

// Types d'identifiants
pub type ProposalId = Uuid;
pub type AccountId = String; // Représente une adresse de compte
pub type VoteId = Uuid;

// Types de propositions possibles
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProposalType {
    ParameterChange(ParameterChangeProposal),
    CodeUpgrade(CodeUpgradeProposal),
    Treasury(TreasuryProposal),
    NetworkUpgrade(NetworkUpgradeProposal),
    ReputationAction(ReputationActionProposal),
    GenericProposal(GenericProposal),
}

// Proposition de changement de paramètres
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterChangeProposal {
    parameter_name: String,
    current_value: GovernanceParameterValue,
    proposed_value: GovernanceParameterValue,
    justification: String,
}

// Proposition de mise à jour de code
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeUpgradeProposal {
    module_name: String,
    commit_hash: String,
    upgrade_description: String,
    test_results: Vec<TestResult>,
    security_audit: Option<SecurityAudit>,
}

// Proposition de trésorerie
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreasuryProposal {
    recipient: AccountId,
    amount: Decimal,
    purpose: String,
    milestones: Vec<Milestone>,
}

// Proposition de mise à niveau du réseau
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkUpgradeProposal {
    upgrade_type: NetworkUpgradeType,
    activation_height: u64,
    activation_time: Option<DateTime<Utc>>,
    features: Vec<String>,
    rollback_plan: String,
}

// Proposition d'action de réputation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReputationActionProposal {
    target_account: AccountId,
    proposed_action: ReputationAction,
    evidence: Vec<String>,
}

// Proposition générique
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenericProposal {
    title: String,
    description: String,
    external_url: Option<String>,
}

// Types de mises à niveau du réseau
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum NetworkUpgradeType {
    SoftFork,
    HardFork,
    ParameterAdjustment,
    ProtocolExtension,
}

// Actions de réputation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReputationAction {
    Increase { points: u32, reason: String },
    Decrease { points: u32, reason: String },
    Blacklist { duration_days: u32, reason: String },
    Unblacklist { reason: String },
}

// Résultats de test
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestResult {
    test_name: String,
    result: bool,
    details: String,
}

// Audit de sécurité
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SecurityAudit {
    auditor: String,
    date: DateTime<Utc>,
    result: SecurityAuditResult,
    report_url: String,
}

// Résultat d'audit de sécurité
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityAuditResult {
    Passed,
    PassedWithRecommendations,
    FailedMinorIssues,
    FailedMajorIssues,
}

// Milestone pour les propositions de trésorerie
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Milestone {
    description: String,
    deadline: DateTime<Utc>,
    amount: Decimal,
    completed: bool,
}

// Proposition de base avec les champs communs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    id: ProposalId,
    title: String,
    description: String,
    proposer: AccountId,
    proposal_type: ProposalType,
    status: ProposalStatus,
    created_at: DateTime<Utc>,
    voting_starts_at: DateTime<Utc>,
    voting_ends_at: DateTime<Utc>,
    executed_at: Option<DateTime<Utc>>,
    votes: HashMap<VoteType, Decimal>,
    voters: HashSet<AccountId>,
    voting_power: Decimal,
    tags: Vec<String>,
    discussion_url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProposalStatus {
    Draft,
    Active,
    Canceled,
    Defeated,
    Successful,
    Executed,
    Expired,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum VoteType {
    For,
    Against,
    Abstain,
}

// Valeur d'un paramètre de gouvernance
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum GovernanceParameterValue {
    Integer(i64),
    Decimal(Decimal),
    Boolean(bool),
    String(String),
    Duration(i64), // en secondes
}

// Paramètre de gouvernance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GovernanceParameter {
    name: String,
    value: GovernanceParameterValue,
    description: String,
    last_updated: DateTime<Utc>,
}

// Une époque de gouvernance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GovernanceEpoch {
    epoch_number: u32,
    start_time: DateTime<Utc>,
    end_time: Option<DateTime<Utc>>,
    proposals_count: u32,
    participation_rate: f64,
    significant_changes: Vec<String>,
}

// Trésorerie
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Treasury {
    balance: Decimal,
    allocations: HashMap<String, Decimal>,
    transaction_history: VecDeque<TreasuryTransaction>,
}

// Transaction de trésorerie
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreasuryTransaction {
    id: Uuid,
    amount: Decimal,
    description: String,
    recipient: Option<AccountId>,
    proposal_id: Option<ProposalId>,
    timestamp: DateTime<Utc>,
}

// Enregistrement de vote
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoteRecord {
    id: VoteId,
    proposal_id: ProposalId,
    voter: AccountId,
    vote_type: VoteType,
    voting_power_used: Decimal,
    effective_power: Decimal,
    timestamp: DateTime<Utc>,
    delegation: Option<AccountId>,
}

// Erreurs du système de gouvernance
#[derive(Error, Debug)]
pub enum GovernanceError {
    #[error("Proposition non trouvée")]
    ProposalNotFound,
    
    #[error("Proposition non active")]
    ProposalNotActive,
    
    #[error("Proposition non réussie")]
    ProposalNotSuccessful,
    
    #[error("Paramètre non trouvé")]
    ParameterNotFound,
    
    #[error("Période de vote fermée")]
    VotingPeriodClosed,
    
    #[error("Pouvoir de proposition insuffisant")]
    InsufficientPower,
    
    #[error("Pouvoir de vote dépassé")]
    ExceededVotingPower,
    
    #[error("A déjà voté")]
    AlreadyVoted,
    
    #[error("Erreur mathématique")]
    MathError,
    
    #[error("Erreur d'exécution: {0}")]
    ExecutionError(String),
}

// Interface pour les stratégies de vote
pub trait VotingStrategy: Send + Sync {
    fn calculate_effective_power(
        &self, 
        voter: AccountId, 
        raw_power: Decimal, 
        proposal: &Proposal
    ) -> Result<Decimal, GovernanceError>;
}

// Interface pour le moteur d'exécution des propositions
pub trait ProposalExecutionEngine: Send + Sync {
    fn execute(&self, proposal: &Proposal) -> Result<(), GovernanceError>;
}

// Système de gouvernance principal
pub struct GovernanceSystem {
    proposals: HashMap<ProposalId, Proposal>,
    parameters: HashMap<String, GovernanceParameter>,
    epochs: Vec<GovernanceEpoch>,
    current_epoch: usize,
    treasury: Treasury,
    voting_strategy: Box<dyn VotingStrategy>,
    execution_engine: Box<dyn ProposalExecutionEngine>,
    votes: DashMap<VoteId, VoteRecord>,
    delegations: DashMap<AccountId, AccountId>,
}

impl GovernanceSystem {
    pub fn new(
        voting_strategy: Box<dyn VotingStrategy>,
        execution_engine: Box<dyn ProposalExecutionEngine>,
    ) -> Self {
        Self {
            proposals: HashMap::new(),
            parameters: initialize_default_parameters(),
            epochs: vec![GovernanceEpoch::initial()],
            current_epoch: 0,
            treasury: Treasury::new(),
            voting_strategy,
            execution_engine,
            votes: DashMap::new(),
            delegations: DashMap::new(),
        }
    }
    
    // Création d'une proposition
    pub fn create_proposal(&mut self, 
                           proposer: AccountId, 
                           title: String,
                           description: String,
                           proposal_type: ProposalType) -> Result<ProposalId, GovernanceError> {
        // Vérifier si le proposant a suffisamment de pouvoir de proposition
        let min_proposal_power = self.get_parameter("min_proposal_power")
            .and_then(|p| p.as_decimal())
            .ok_or(GovernanceError::ParameterNotFound)?;
            
        let proposer_power = self.calculate_proposal_power(proposer.clone())
            .ok_or(GovernanceError::InsufficientPower)?;
            
        if proposer_power < min_proposal_power {
            return Err(GovernanceError::InsufficientPower);
        }
        
        // Créer la proposition
        let now = Utc::now();
        let voting_duration = Duration::hours(
            self.get_parameter("voting_duration_hours")
                .and_then(|p| p.as_i64())
                .unwrap_or(72) // 3 jours par défaut
        );
        
        let voting_delay = Duration::hours(
            self.get_parameter("voting_delay_hours")
                .and_then(|p| p.as_i64())
                .unwrap_or(24) // 1 jour par défaut
        );
        
        let voting_starts_at = now + voting_delay;
        let voting_ends_at = voting_starts_at + voting_duration;
        
        let id = ProposalId(Uuid::new_v4());
        
        let proposal = Proposal {
            id,
            title,
            description,
            proposer,
            proposal_type,
            status: ProposalStatus::Draft,
            created_at: now,
            voting_starts_at,
            voting_ends_at,
            executed_at: None,
            votes: HashMap::new(),
            voters: HashSet::new(),
            voting_power: Decimal::ZERO,
            tags: Vec::new(),
            discussion_url: None,
        };
        
        self.proposals.insert(id, proposal);
        
        // Mettre à jour l'époque actuelle
        let current_epoch = &mut self.epochs[self.current_epoch];
        current_epoch.proposals_count += 1;
        
        Ok(id)
    }
    
    // Vote sur une proposition
    pub fn vote(&mut self, 
                voter: AccountId, 
                proposal_id: ProposalId, 
                vote_type: VoteType,
                vote_power: Decimal) -> Result<VoteId, GovernanceError> {
        // Récupérer la proposition
        let proposal = self.proposals.get_mut(&proposal_id)
            .ok_or(GovernanceError::ProposalNotFound)?;
            
        // Vérifier que la proposition est active
        if proposal.status != ProposalStatus::Active {
            return Err(GovernanceError::ProposalNotActive);
        }
        
        // Vérifier que nous sommes dans la période de vote
        let now = Utc::now();
        if now < proposal.voting_starts_at || now > proposal.voting_ends_at {
            return Err(GovernanceError::VotingPeriodClosed);
        }
        
        // Vérifier que l'électeur a suffisamment de pouvoir de vote
        let voter_max_power = self.calculate_voting_power(voter.clone())
            .ok_or(GovernanceError::InsufficientPower)?;
            
        if vote_power > voter_max_power {
            return Err(GovernanceError::ExceededVotingPower);
        }
        
        // Vérifier si l'électeur a déjà voté
        if proposal.voters.contains(&voter) {
            return Err(GovernanceError::AlreadyVoted);
        }
        
        // Récupérer la délégation si elle existe
        let effective_voter = if let Some(delegated_to) = self.delegations.get(&voter) {
            // Vérifier que le délégué n'a pas déjà voté
            if proposal.voters.contains(delegated_to.value()) {
                return Err(GovernanceError::AlreadyVoted);
            }
            delegated_to.clone()
        } else {
            voter.clone()
        };
        
        // Appliquer la stratégie de vote (ex: quadratique)
        let effective_power = self.voting_strategy.calculate_effective_power(
            effective_voter.clone(), vote_power, proposal
        )?;
        
        // Enregistrer le vote
        *proposal.votes.entry(vote_type.clone()).or_insert(Decimal::ZERO) += effective_power;
        proposal.voters.insert(effective_voter.clone());
        proposal.voting_power += effective_power;
        
        // Créer l'enregistrement du vote
        let vote_id = VoteId(Uuid::new_v4());
        let vote_record = VoteRecord {
            id: vote_id,
            proposal_id,
            voter: voter.clone(),
            vote_type,
            voting_power_used: vote_power,
            effective_power,
            timestamp: now,
            delegation: if voter != effective_voter {
                Some(effective_voter)
            } else {
                None
            },
        };
        
        self.votes.insert(vote_id, vote_record);
        
        // Vérifier si le quorum est atteint et si la proposition peut être finalisée
        self.check_proposal_finalization(proposal_id)?;
        
        Ok(vote_id)
    }
    
    // Déléguer son pouvoir de vote
    pub fn delegate_vote(&self, delegator: AccountId, delegate: AccountId) -> Result<(), GovernanceError> {
        // Empêcher les délégations circulaires
        let mut current = delegate.clone();
        let mut visited = HashSet::new();
        visited.insert(delegator.clone());
        
        while let Some(next_delegate) = self.delegations.get(&current) {
            let next = next_delegate.value().clone();
            if visited.contains(&next) {
                return Err(GovernanceError::ExecutionError("Délégation circulaire détectée".to_string()));
            }
            
            visited.insert(next.clone());
            current = next;
        }
        
        // Enregistrer la délégation
        self.delegations.insert(delegator, delegate);
        
        Ok(())
    }
    
    // Retirer une délégation
    pub fn undelegate(&self, delegator: &AccountId) -> bool {
        self.delegations.remove(delegator).is_some()
    }
    
    // Exécuter une proposition approuvée
    pub fn execute_proposal(&mut self, proposal_id: ProposalId) -> Result<(), GovernanceError> {
        // Récupérer la proposition
        let proposal = self.proposals.get(&proposal_id)
            .ok_or(GovernanceError::ProposalNotFound)?;
            
        // Vérifier que la proposition est approuvée
        if proposal.status != ProposalStatus::Successful {
            return Err(GovernanceError::ProposalNotSuccessful);
        }
        
        // Exécuter la proposition via le moteur d'exécution
        self.execution_engine.execute(proposal)?;
        
        // Mettre à jour le statut
        if let Some(proposal) = self.proposals.get_mut(&proposal_id) {
            proposal.status = ProposalStatus::Executed;
            proposal.executed_at = Some(Utc::now());
        }
        
        // Si c'est une proposition de trésorerie, enregistrer la transaction
        if let ProposalType::Treasury(treasury_proposal) = &proposal.proposal_type {
            self.treasury.record_transaction(
                treasury_proposal.amount,
                format!("Exécution de la proposition #{}: {}", proposal_id, proposal.title),
                Some(treasury_proposal.recipient.clone()),
                Some(proposal_id),
            );
        }
        
        Ok(())
    }
    
    // Vérifier si une proposition peut être finalisée
    fn check_proposal_finalization(&mut self, proposal_id: ProposalId) -> Result<(), GovernanceError> {
        let proposal = self.proposals.get_mut(&proposal_id)
            .ok_or(GovernanceError::ProposalNotFound)?;
            
        // Vérifier que la proposition est active
        if proposal.status != ProposalStatus::Active {
            return Ok(());
        }
        
        // Calculer le quorum requis
        let quorum_percentage = self.get_parameter("quorum_percentage")
            .and_then(|p| p.as_decimal())
            .unwrap_or(Decimal::new(10, 0)); // 10% par défaut
            
        let total_voting_power = self.calculate_total_voting_power();
        let required_quorum = total_voting_power * quorum_percentage / Decimal::new(100, 0);
        
        // Vérifier si le quorum est atteint
        if proposal.voting_power < required_quorum {
            return Ok(());
        }
        
        // Vérifier si la proposition est acceptée
        let for_votes = proposal.votes.get(&VoteType::For).cloned().unwrap_or(Decimal::ZERO);
        let against_votes = proposal.votes.get(&VoteType::Against).cloned().unwrap_or(Decimal::ZERO);
        
        let approval_threshold = self.get_parameter("approval_threshold")
            .and_then(|p| p.as_decimal())
            .unwrap_or(Decimal::new(50, 0)); // 50% par défaut
            
        let total_cast = for_votes + against_votes;
        if total_cast == Decimal::ZERO {
            return Ok(());
        }
        
        let for_percentage = for_votes * Decimal::new(100, 0) / total_cast;
        
        if for_percentage >= approval_threshold {
            proposal.status = ProposalStatus::Successful;
        } else {
            proposal.status = ProposalStatus::Defeated;
        }
        
        Ok(())
    }
    
    // Nettoyer les propositions expirées
    pub fn clean_expired_proposals(&mut self) -> Vec<ProposalId> {
        let now = Utc::now();
        let mut expired_ids = Vec::new();
        
        for (id, proposal) in &mut self.proposals {
            if proposal.status == ProposalStatus::Active && now > proposal.voting_ends_at {
                proposal.status = ProposalStatus::Expired;
                expired_ids.push(*id);
            }
        }
        
        expired_ids
    }
    
    // Récupérer une proposition par son ID
    pub fn get_proposal(&self, proposal_id: ProposalId) -> Option<&Proposal> {
        self.proposals.get(&proposal_id)
    }
    
    // Récupérer les propositions actives
    pub fn get_active_proposals(&self) -> Vec<&Proposal> {
        self.proposals.values()
            .filter(|p| p.status == ProposalStatus::Active)
            .collect()
    }
    
    // Récupérer l'historique des propositions
    pub fn get_proposal_history(&self, limit: usize) -> Vec<&Proposal> {
        let mut proposals: Vec<&Proposal> = self.proposals.values().collect();
        
        // Trier par date de création (plus récente d'abord)
        proposals.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        
        proposals.into_iter().take(limit).collect()
    }
    
    // Rechercher des propositions
    pub fn search_proposals(&self, query: &str, status: Option<ProposalStatus>, limit: usize) -> Vec<&Proposal> {
        let query = query.to_lowercase();
        let mut results: Vec<&Proposal> = self.proposals.values()
            .filter(|p| {
                // Filtrer par statut si spécifié
                if let Some(ref status) = status {
                    if &p.status != status {
                        return false;
                    }
                }
                
                // Rechercher dans le titre et la description
                p.title.to_lowercase().contains(&query) || 
                p.description.to_lowercase().contains(&query) ||
                p.tags.iter().any(|tag| tag.to_lowercase().contains(&query))
            })
            .collect();
            
        // Trier par pertinence (correspondance dans le titre d'abord)
        results.sort_by(|a, b| {
            let a_title_match = a.title.to_lowercase().contains(&query);
            let b_title_match = b.title.to_lowercase().contains(&query);
            
            match (a_title_match, b_title_match) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => b.created_at.cmp(&a.created_at),
            }
        });
        
        results.into_iter().take(limit).collect()
    }
    
    // Récupérer un paramètre de gouvernance
    pub fn get_parameter(&self, name: &str) -> Option<&GovernanceParameterValue> {
        self.parameters.get(name).map(|p| &p.value)
    }
    
    // Mettre à jour un paramètre de gouvernance
    pub fn update_parameter(&mut self, name: &str, value: GovernanceParameterValue) -> Result<(), GovernanceError> {
        if let Some(param) = self.parameters.get_mut(name) {
            param.value = value;
            param.last_updated = Utc::now();
            Ok(())
        } else {
            Err(GovernanceError::ParameterNotFound)
        }
    }
    
    // Calculer le pouvoir de proposition d'un compte
    fn calculate_proposal_power(&self, account: AccountId) -> Option<Decimal> {
        // Cette fonction dépendrait de l'implémentation spécifique
        // (ex: basée sur les tokens possédés, la réputation, etc.)
        
        // Pour l'exemple, renvoi une valeur factice
        Some(Decimal::new(100, 0))
    }
    
    // Calculer le pouvoir de vote d'un compte
    fn calculate_voting_power(&self, account: AccountId) -> Option<Decimal> {
        // Cette fonction dépendrait de l'implémentation spécifique
        // (ex: basée sur les tokens possédés, la réputation, etc.)
        
        // Pour l'exemple, renvoi une valeur factice
        Some(Decimal::new(100, 0))
    }
    
    // Calculer le pouvoir de vote total dans le système
    fn calculate_total_voting_power(&self) -> Decimal {
        // Cette fonction dépendrait de l'implémentation spécifique
        
        // Pour l'exemple, renvoi une valeur factice
        Decimal::new(10000, 0)
    }
}

// Vote quadratique - implémentation d'une stratégie de vote avancée
pub struct QuadraticVotingStrategy;

impl VotingStrategy for QuadraticVotingStrategy {
    fn calculate_effective_power(&self, 
                                voter: AccountId, 
                                raw_power: Decimal, 
                                proposal: &Proposal) -> Result<Decimal, GovernanceError> {
        // La puissance effective est proportionnelle à la racine carrée de la puissance brute
        // Cela permet aux petits votants d'avoir plus d'impact proportionnellement
        let raw_power_float = raw_power.to_f64()
            .ok_or(GovernanceError::MathError)?;
            
        let effective_power_float = raw_power_float.sqrt();
        
        Decimal::try_from(effective_power_float)
            .map_err(|_| GovernanceError::MathError)
    }
}

// Méthodes pour Treasury
impl Treasury {
    pub fn new() -> Self {
        Self {
            balance: Decimal::ZERO,
            allocations: HashMap::new(),
            transaction_history: VecDeque::new(),
        }
    }
    
    pub fn deposit(&mut self, amount: Decimal) {
        self.balance += amount;
    }
    
    pub fn withdraw(&mut self, amount: Decimal, recipient: AccountId, description: String) -> Result<(), GovernanceError> {
        if amount > self.balance {
            return Err(GovernanceError::ExecutionError("Solde insuffisant".to_string()));
        }
        
        self.balance -= amount;
        self.record_transaction(amount, description, Some(recipient), None);
        
        Ok(())
    }
    
    pub fn record_transaction(&mut self, amount: Decimal, description: String, recipient: Option<AccountId>, proposal_id: Option<ProposalId>) {
        let transaction = TreasuryTransaction {
            id: Uuid::new_v4(),
            amount,
            description,
            recipient,
            proposal_id,
            timestamp: Utc::now(),
        };
        
        self.transaction_history.push_back(transaction);
        
        // Limiter la taille de l'historique
        while self.transaction_history.len() > 1000 {
            self.transaction_history.pop_front();
        }
    }
    
    pub fn allocate_funds(&mut self, category: String, amount: Decimal) -> Result<(), GovernanceError> {
        if amount > self.balance {
            return Err(GovernanceError::ExecutionError("Solde insuffisant".to_string()));
        }
        
        *self.allocations.entry(category.clone()).or_insert(Decimal::ZERO) += amount;
        self.balance -= amount;
        
        self.record_transaction(
            amount,
            format!("Allocation de fonds à la catégorie {}", category),
            None,
            None
        );
        
        Ok(())
    }
    
    pub fn get_balance(&self) -> Decimal {
        self.balance
    }
    
    pub fn get_allocations(&self) -> &HashMap<String, Decimal> {
        &self.allocations
    }
    
    pub fn get_transaction_history(&self, limit: usize) -> Vec<&TreasuryTransaction> {
        self.transaction_history.iter().rev().take(limit).collect()
    }
}

// Méthodes pour GovernanceEpoch
impl GovernanceEpoch {
    pub fn initial() -> Self {
        Self {
            epoch_number: 1,
            start_time: Utc::now(),
            end_time: None,
            proposals_count: 0,
            participation_rate: 0.0,
            significant_changes: Vec::new(),
        }
    }
    
    pub fn new(epoch_number: u32) -> Self {
        Self {
            epoch_number,
            start_time: Utc::now(),
            end_time: None,
            proposals_count: 0,
            participation_rate: 0.0,
            significant_changes: Vec::new(),
        }
    }
    
    pub fn close(&mut self, participation_rate: f64, significant_changes: Vec<String>) {
        self.end_time = Some(Utc::now());
        self.participation_rate = participation_rate;
        self.significant_changes = significant_changes;
    }
}

// Méthodes pour GovernanceParameterValue
impl GovernanceParameterValue {
    pub fn as_i64(&self) -> Option<i64> {
        if let GovernanceParameterValue::Integer(value) = self {
            Some(*value)
        } else if let GovernanceParameterValue::Duration(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    
    pub fn as_decimal(&self) -> Option<Decimal> {
        if let GovernanceParameterValue::Decimal(value) = self {
            Some(*value)
        } else if let GovernanceParameterValue::Integer(value) = self {
            Some(Decimal::from(*value))
        } else {
            None
        }
    }
    
    pub fn as_bool(&self) -> Option<bool> {
        if let GovernanceParameterValue::Boolean(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    
    pub fn as_string(&self) -> Option<&String> {
        if let GovernanceParameterValue::String(value) = self {
            Some(value)
        } else {
            None
        }
    }
}

// Initialiser les paramètres par défaut
fn initialize_default_parameters() -> HashMap<String, GovernanceParameter> {
    let mut params = HashMap::new();
    
    params.insert("quorum_percentage".to_string(), GovernanceParameter {
        name: "quorum_percentage".to_string(),
        value: GovernanceParameterValue::Decimal(Decimal::new(10, 0)), // 10%
        description: "Pourcentage minimum de participation requis pour qu'une proposition soit valide".to_string(),
        last_updated: Utc::now(),
    });
    
    params.insert("approval_threshold".to_string(), GovernanceParameter {
        name: "approval_threshold".to_string(),
        value: GovernanceParameterValue::Decimal(Decimal::new(50, 0)), // 50%
        description: "Pourcentage minimum de votes 'pour' requis pour qu'une proposition soit acceptée".to_string(),
        last_updated: Utc::now(),
    });
    
    params.insert("min_proposal_power".to_string(), GovernanceParameter {
        name: "min_proposal_power".to_string(),
        value: GovernanceParameterValue::Decimal(Decimal::new(10, 0)), // 10
        description: "Pouvoir minimum requis pour soumettre une proposition".to_string(),
        last_updated: Utc::now(),
    });
    
    params.insert("voting_duration_hours".to_string(), GovernanceParameter {
        name: "voting_duration_hours".to_string(),
        value: GovernanceParameterValue::Integer(72), // 3 jours
        description: "Durée de la période de vote en heures".to_string(),
        last_updated: Utc::now(),
    });
    
    params.insert("voting_delay_hours".to_string(), GovernanceParameter {
        name: "voting_delay_hours".to_string(),
        value: GovernanceParameterValue::Integer(24), // 1 jour
        description: "Délai avant le début de la période de vote en heures".to_string(),
        last_updated: Utc::now(),
    });
    
    params
}
