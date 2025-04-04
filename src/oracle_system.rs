use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::Mutex;
use serde::{Serialize, Deserialize};
use rust_decimal::Decimal;
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration};
use async_trait::async_trait;
use thiserror::Error;

// Types d'identifiants
pub type OracleFeedId = Uuid;
pub type ProviderId = Uuid;
pub type OracleDataTypeId = String;

// Types de données supportés par l'oracle
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OracleDataType {
    PriceData(PriceOracleData),
    WeatherData(WeatherOracleData),
    SportsData(SportsOracleData),
    GenericData(GenericOracleData),
}

// Données de prix pour les paires d'actifs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriceOracleData {
    base_asset: String,
    quote_asset: String,
    price: Decimal,
    volume_24h: Option<Decimal>,
    timestamp: DateTime<Utc>,
}

// Données météorologiques
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeatherOracleData {
    location: GeoCoordinates,
    temperature: f32, // en Celsius
    humidity: u8,     // pourcentage
    wind_speed: f32,  // m/s
    precipitation: f32, // mm
    timestamp: DateTime<Utc>,
}

// Coordonnées géographiques
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoCoordinates {
    latitude: f64,
    longitude: f64,
}

// Données sportives
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SportsOracleData {
    sport: String,
    event_id: String,
    home_team: String,
    away_team: String,
    home_score: i32,
    away_score: i32,
    status: GameStatus,
    timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GameStatus {
    NotStarted,
    InProgress,
    Completed,
    Postponed,
    Cancelled,
}

// Données génériques
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenericOracleData {
    data_type: String,
    string_value: Option<String>,
    numeric_value: Option<Decimal>,
    boolean_value: Option<bool>,
    binary_data: Option<Vec<u8>>,
    timestamp: DateTime<Utc>,
}

// Erreurs du système d'oracle
#[derive(Error, Debug)]
pub enum OracleError {
    #[error("Configuration invalide: {0}")]
    InvalidConfiguration(&'static str),
    
    #[error("Fournisseur inconnu: {0}")]
    UnknownProvider(ProviderId),
    
    #[error("Flux inconnu: {0}")]
    UnknownFeed(OracleFeedId),
    
    #[error("Fournisseur non autorisé")]
    UnauthorizedProvider,
    
    #[error("Aucune donnée disponible")]
    NoDataAvailable,
    
    #[error("Type de données non supporté")]
    UnsupportedDataType,
    
    #[error("Erreur réseau: {0}")]
    NetworkError(String),
    
    #[error("Erreur de consensus: {0}")]
    ConsensusError(String),
}

// Fraîcheur des données d'un flux
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DataFreshness {
    Fresh,    // Données récentes
    Warning,  // Données un peu anciennes
    Stale,    // Données périmées
    NoData,   // Aucune donnée
}

// Système principal de gestion des oracles
pub struct AdaptiveOracleSystem {
    data_feeds: DashMap<OracleFeedId, OracleFeed>,
    data_providers: HashMap<ProviderId, OracleProvider>,
    aggregation_strategies: HashMap<OracleDataTypeId, Box<dyn AggregationStrategy>>,
    consensus_mechanism: Box<dyn OracleConsensus>,
    dispute_resolution: Arc<DisputeResolutionSystem>,
    reputation_tracker: Arc<OracleReputationTracker>,
}

// Fournisseur d'oracle
#[derive(Clone)]
pub struct OracleProvider {
    id: ProviderId,
    name: String,
    endpoint: Option<String>,
    supported_data_types: Vec<OracleDataTypeId>,
    public_key: Vec<u8>,
}

// Flux de données oracle
pub struct OracleFeed {
    id: OracleFeedId,
    config: OracleFeedConfig,
    latest_data: Option<OracleDataType>,
    data_history: VecDeque<DataHistoryEntry>,
    last_update: Option<DateTime<Utc>>,
}

// Configuration d'un flux de données
#[derive(Clone)]
pub struct OracleFeedConfig {
    name: String,
    description: String,
    data_type: OracleDataTypeId,
    providers: Vec<ProviderId>,
    min_providers: usize,
    update_interval: Duration,
    aggregation_strategy: AggregationStrategyType,
    threshold: f64, // Pourcentage de votes requis pour le consensus (0.0-1.0)
}

// Entrée d'historique des données
#[derive(Clone)]
pub struct DataHistoryEntry {
    data: OracleDataType,
    timestamp: DateTime<Utc>,
    providers: Vec<ProviderId>,
}

// Soumission d'un fournisseur d'oracle
#[derive(Clone)]
pub struct OracleSubmission {
    feed_id: OracleFeedId,
    provider_id: ProviderId,
    data: OracleDataType,
    timestamp: DateTime<Utc>,
}

// Résultat du processus de consensus
pub struct ConsensusResult {
    consensus_reached: bool,
    aggregated_data: OracleDataType,
    contributing_providers: Vec<ProviderId>,
    outlier_providers: Vec<ProviderId>,
}

// Types de stratégies d'agrégation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AggregationStrategyType {
    Mean,
    Median,
    Mode,
    WeightedAverage,
    TrustScoreWeighted,
    Custom(String),
}

// Trait pour les stratégies d'agrégation
#[async_trait]
pub trait AggregationStrategy: Send + Sync {
    async fn aggregate(&self, submissions: &[OracleSubmission]) -> Result<OracleDataType, OracleError>;
    fn is_outlier(&self, submission: &OracleSubmission, others: &[OracleSubmission]) -> bool;
}

// Trait pour les mécanismes de consensus
#[async_trait]
pub trait OracleConsensus: Send + Sync {
    async fn process_submission(&self, submission: OracleSubmission) -> Result<ConsensusResult, OracleError>;
}

// Système de résolution de disputes
pub struct DisputeResolutionSystem {
    active_disputes: DashMap<DisputeId, Dispute>,
    dispute_voters: DashMap<DisputeId, HashMap<ProviderId, bool>>,
    dispute_resolutions: VecDeque<DisputeResolution>,
}

type DisputeId = Uuid;

#[derive(Clone)]
pub struct Dispute {
    id: DisputeId,
    feed_id: OracleFeedId,
    disputed_data: OracleDataType,
    challenger: ProviderId,
    challenged: ProviderId,
    reason: String,
    evidence: Vec<u8>,
    status: DisputeStatus,
    created_at: DateTime<Utc>,
    resolved_at: Option<DateTime<Utc>>,
}

#[derive(Clone, PartialEq, Eq)]
pub enum DisputeStatus {
    Open,
    Voting,
    Resolved(DisputeOutcome),
    Cancelled,
}

#[derive(Clone, PartialEq, Eq)]
pub enum DisputeOutcome {
    ChallengeFailed,
    ChallengeSucceeded,
    Inconclusive,
}

#[derive(Clone)]
pub struct DisputeResolution {
    dispute_id: DisputeId,
    outcome: DisputeOutcome,
    resolution_time: DateTime<Utc>,
    voter_count: usize,
}

// Tracker de réputation pour les fournisseurs d'oracle
pub struct OracleReputationTracker {
    provider_scores: DashMap<ProviderId, ReputationScore>,
    history: DashMap<ProviderId, VecDeque<ReputationEvent>>,
}

#[derive(Clone)]
pub struct ReputationScore {
    accuracy: f64,
    responsiveness: f64,
    consistency: f64,
    dispute_ratio: f64,
    uptime: f64,
}

#[derive(Clone)]
pub enum ReputationEvent {
    AccurateReport { feed_id: OracleFeedId, timestamp: DateTime<Utc> },
    InaccurateReport { feed_id: OracleFeedId, timestamp: DateTime<Utc> },
    DisputeWon { dispute_id: DisputeId, timestamp: DateTime<Utc> },
    DisputeLost { dispute_id: DisputeId, timestamp: DateTime<Utc> },
    Offline { duration_minutes: u32, timestamp: DateTime<Utc> },
}

impl AdaptiveOracleSystem {
    pub fn new(consensus_mechanism: Box<dyn OracleConsensus>) -> Self {
        Self {
            data_feeds: DashMap::new(),
            data_providers: HashMap::new(),
            aggregation_strategies: HashMap::new(),
            consensus_mechanism,
            dispute_resolution: Arc::new(DisputeResolutionSystem::new()),
            reputation_tracker: Arc::new(OracleReputationTracker::new()),
        }
    }
    
    // Enregistrement d'un nouveau fournisseur d'oracle
    pub fn register_provider(&mut self, provider: OracleProvider) -> ProviderId {
        let provider_id = provider.id;
        self.data_providers.insert(provider_id, provider);
        provider_id
    }
    
    // Création d'un nouveau flux de données
    pub fn create_data_feed(&self, config: OracleFeedConfig) -> Result<OracleFeedId, OracleError> {
        // Valider la configuration
        if config.update_interval < Duration::seconds(1) {
            return Err(OracleError::InvalidConfiguration("L'intervalle de mise à jour est trop petit"));
        }
        
        if config.providers.is_empty() {
            return Err(OracleError::InvalidConfiguration("Aucun fournisseur spécifié"));
        }
        
        // Vérifier que tous les fournisseurs existent
        for provider_id in &config.providers {
            if !self.data_providers.contains_key(provider_id) {
                return Err(OracleError::UnknownProvider(*provider_id));
            }
        }
        
        // Créer le flux
        let feed_id = OracleFeedId(Uuid::new_v4());
        let feed = OracleFeed {
            id: feed_id,
            config,
            latest_data: None,
            data_history: VecDeque::with_capacity(100),
            last_update: None,
        };
        
        self.data_feeds.insert(feed_id, feed);
        
        Ok(feed_id)
    }
    
    // Soumission de données par un fournisseur
    pub async fn submit_data(&self, provider_id: ProviderId, feed_id: OracleFeedId, 
                           data: OracleDataType) -> Result<(), OracleError> {
        // Vérifier que le fournisseur et le flux existent
        if !self.data_providers.contains_key(&provider_id) {
            return Err(OracleError::UnknownProvider(provider_id));
        }
        
        let feed = match self.data_feeds.get(&feed_id) {
            Some(feed) => feed,
            None => return Err(OracleError::UnknownFeed(feed_id)),
        };
        
        // Vérifier que le fournisseur est autorisé pour ce flux
        if !feed.config.providers.contains(&provider_id) {
            return Err(OracleError::UnauthorizedProvider);
        }
        
        // Préparer la soumission
        let submission = OracleSubmission {
            feed_id,
            provider_id,
            data: data.clone(),
            timestamp: Utc::now(),
        };
        
        // Traiter via notre mécanisme de consensus
        let consensus_result = self.consensus_mechanism.process_submission(submission).await?;
        
        // Si le consensus est atteint, mettre à jour les données
        if consensus_result.consensus_reached {
            let mut feed = self.data_feeds.get_mut(&feed_id).unwrap();
            feed.latest_data = Some(consensus_result.aggregated_data.clone());
            feed.last_update = Some(Utc::now());
            
            // Enregistrer dans l'historique
            feed.data_history.push_back(DataHistoryEntry {
                data: consensus_result.aggregated_data,
                timestamp: Utc::now(),
                providers: consensus_result.contributing_providers.clone(),
            });
            
            // Limiter la taille de l'historique
            while feed.data_history.len() > 100 {
                feed.data_history.pop_front();
            }
            
            // Mettre à jour la réputation des fournisseurs
            for provider in &consensus_result.outlier_providers {
                self.reputation_tracker.report_outlier(*provider);
            }
            
            for provider in &consensus_result.contributing_providers {
                self.reputation_tracker.report_contribution(*provider);
            }
        }
        
        Ok(())
    }
    
    // Récupération des dernières données d'un flux
    pub fn get_latest_data(&self, feed_id: OracleFeedId) -> Result<OracleDataType, OracleError> {
        let feed = self.data_feeds.get(&feed_id)
            .ok_or(OracleError::UnknownFeed(feed_id))?;
            
        if let Some(data) = &feed.latest_data {
            Ok(data.clone())
        } else {
            Err(OracleError::NoDataAvailable)
        }
    }
    
    // Récupération de l'historique des données
    pub fn get_data_history(&self, feed_id: OracleFeedId, limit: usize) 
                          -> Result<Vec<DataHistoryEntry>, OracleError> {
        let feed = self.data_feeds.get(&feed_id)
            .ok_or(OracleError::UnknownFeed(feed_id))?;
            
        let history: Vec<DataHistoryEntry> = feed.data_history
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect();
            
        Ok(history)
    }
    
    // Vérification de la fraîcheur des données
    pub fn check_data_freshness(&self, feed_id: OracleFeedId) -> Result<DataFreshness, OracleError> {
        let feed = self.data_feeds.get(&feed_id)
            .ok_or(OracleError::UnknownFeed(feed_id))?;
            
        if let Some(last_update) = feed.last_update {
            let elapsed = Utc::now().signed_duration_since(last_update);
            
            if elapsed > feed.config.update_interval * 3 {
                Ok(DataFreshness::Stale)
            } else if elapsed > feed.config.update_interval {
                Ok(DataFreshness::Warning)
            } else {
                Ok(DataFreshness::Fresh)
            }
        } else {
            Ok(DataFreshness::NoData)
        }
    }
    
    // Créer une dispute sur des données
    pub async fn create_dispute(&self, feed_id: OracleFeedId, disputed_data: OracleDataType,
                              challenger: ProviderId, challenged: ProviderId,
                              reason: String, evidence: Vec<u8>) -> Result<DisputeId, OracleError> {
        // Vérifier que tout existe
        if !self.data_providers.contains_key(&challenger) {
            return Err(OracleError::UnknownProvider(challenger));
        }
        
        if !self.data_providers.contains_key(&challenged) {
            return Err(OracleError::UnknownProvider(challenged));
        }
        
        if !self.data_feeds.contains_key(&feed_id) {
            return Err(OracleError::UnknownFeed(feed_id));
        }
        
        // Créer la dispute
        let dispute_id = self.dispute_resolution.create_dispute(
            feed_id, disputed_data, challenger, challenged, reason, evidence
        ).await;
        
        Ok(dispute_id)
    }
    
    // Voter sur une dispute
    pub async fn vote_on_dispute(&self, dispute_id: DisputeId, voter: ProviderId,
                               vote: bool) -> Result<(), OracleError> {
        if !self.data_providers.contains_key(&voter) {
            return Err(OracleError::UnknownProvider(voter));
        }
        
        self.dispute_resolution.register_vote(dispute_id, voter, vote).await
    }
    
    // Ajouter une stratégie d'agrégation
    pub fn add_aggregation_strategy(&mut self, data_type: &str, strategy: Box<dyn AggregationStrategy>) {
        self.aggregation_strategies.insert(data_type.to_string(), strategy);
    }
}

// Implémentation du tracker de réputation
impl OracleReputationTracker {
    pub fn new() -> Self {
        Self {
            provider_scores: DashMap::new(),
            history: DashMap::new(),
        }
    }
    
    // Signaler une contribution correcte
    pub fn report_contribution(&self, provider_id: ProviderId) {
        let event = ReputationEvent::AccurateReport {
            feed_id: OracleFeedId(Uuid::new_v4()), // Placeholder
            timestamp: Utc::now(),
        };
        
        self.record_event(provider_id, event);
        self.update_score(provider_id, |score| {
            score.accuracy = (score.accuracy * 0.9) + 0.1;
            score.consistency = (score.consistency * 0.95) + 0.05;
        });
    }
    
    // Signaler une valeur aberrante
    pub fn report_outlier(&self, provider_id: ProviderId) {
        let event = ReputationEvent::InaccurateReport {
            feed_id: OracleFeedId(Uuid::new_v4()), // Placeholder
            timestamp: Utc::now(),
        };
        
        self.record_event(provider_id, event);
        self.update_score(provider_id, |score| {
            score.accuracy = (score.accuracy * 0.9) - 0.1;
            score.consistency = (score.consistency * 0.95) - 0.05;
        });
    }
    
    // Enregistrer un événement de réputation
    fn record_event(&self, provider_id: ProviderId, event: ReputationEvent) {
        let history = self.history.entry(provider_id).or_insert_with(|| VecDeque::with_capacity(100));
        history.push_back(event);
        
        // Limiter la taille de l'historique
        while history.len() > 100 {
            history.pop_front();
        }
    }
    
    // Mettre à jour le score d'un fournisseur
    fn update_score<F>(&self, provider_id: ProviderId, updater: F)
    where
        F: FnOnce(&mut ReputationScore),
    {
        let mut score = self.provider_scores.entry(provider_id).or_insert_with(|| ReputationScore {
            accuracy: 0.5,
            responsiveness: 0.5,
            consistency: 0.5,
            dispute_ratio: 0.0,
            uptime: 1.0,
        });
        
        updater(&mut score);
        
        // S'assurer que les scores restent dans les limites
        score.accuracy = score.accuracy.clamp(0.0, 1.0);
        score.consistency = score.consistency.clamp(0.0, 1.0);
    }
    
    // Obtenir la réputation d'un fournisseur
    pub fn get_reputation(&self, provider_id: ProviderId) -> f64 {
        if let Some(score) = self.provider_scores.get(&provider_id) {
            // Calcul pondéré du score total
            (score.accuracy * 0.4) +
            (score.consistency * 0.2) +
            (score.responsiveness * 0.2) +
            (score.uptime * 0.1) -
            (score.dispute_ratio * 0.1)
        } else {
            // Score par défaut pour un nouveau fournisseur
            0.5
        }
    }
}

// Implémentation du système de résolution des disputes
impl DisputeResolutionSystem {
    pub fn new() -> Self {
        Self {
            active_disputes: DashMap::new(),
            dispute_voters: DashMap::new(),
            dispute_resolutions: VecDeque::new(),
        }
    }
    
    // Créer une nouvelle dispute
    pub async fn create_dispute(
        &self,
        feed_id: OracleFeedId,
        disputed_data: OracleDataType,
        challenger: ProviderId,
        challenged: ProviderId,
        reason: String,
        evidence: Vec<u8>
    ) -> DisputeId {
        let dispute_id = DisputeId(Uuid::new_v4());
        
        let dispute = Dispute {
            id: dispute_id,
            feed_id,
            disputed_data,
            challenger,
            challenged,
            reason,
            
