//! Module immune_guard.rs - Système immunitaire distribué auto-adaptatif
//! Inspiré du système immunitaire adaptatif des vertébrés avec mémoire immunologique
//! et capacité de reconnaissance du soi/non-soi.

use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use log::{debug, info, warn, error};
use rand::{thread_rng, Rng};
use rayon::prelude::*;

// Constantes d'optimisation immunitaire
const MEMORY_CELL_CAPACITY: usize = 10000;   // Capacité maximale de cellules mémoires immunitaires
const THREAT_DETECTION_THRESHOLD: f64 = 0.7; // Seuil de détection des menaces
const IMMUNE_RESPONSE_DELAY_MS: u64 = 50;    // Délai de réponse immunitaire (ms)
const ANTIBODY_GENERATION_TIME_MS: u64 = 200; // Temps de génération d'anticorps (ms)
const AUTO_HEALING_FACTOR: f64 = 0.025;      // Facteur de guérison automatique

/// Types de menaces détectables par le système immunitaire
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ThreatType {
    /// Attaque DoS/DDoS (submersion de messages)
    DenialOfService,
    /// Tentative de double dépense
    DoubleSpend,
    /// Bloc invalide ou malformé
    InvalidBlock,
    /// Transaction invalide ou malformée
    InvalidTransaction,
    /// Nœud malveillant ou dysfonctionnel
    MaliciousNode,
    /// Tentative de fork indésirable
    MaliciousFork,
    /// Exploitation d'une vulnérabilité
    Exploitation,
    /// Anomalie comportementale
    BehavioralAnomaly,
    /// Corruption de données
    DataCorruption,
    /// Menace non catégorisée
    Unknown,
}

/// Menace détectée avec contexte et métadonnées
#[derive(Clone, Debug)]
pub struct Threat {
    /// Type de menace
    pub threat_type: ThreatType,
    /// Source de la menace (hash, ID, adresse IP, etc.)
    pub source: Vec<u8>,
    /// Sévérité de la menace (0.0-1.0)
    pub severity: f64,
    /// Confiance dans la détection (0.0-1.0)
    pub confidence: f64,
    /// Caractéristiques de la menace pour identification
    pub signature: Vec<u8>,
    /// Contexte supplémentaire
    pub context: HashMap<String, Vec<u8>>,
    /// Moment de détection
    pub detection_time: Instant,
    /// Menaces similaires récemment observées
    pub correlation_count: u32,
}

/// Réponse immunitaire générée pour contrer une menace
#[derive(Clone, Debug)]
pub struct ImmuneResponse {
    /// Type de menace ciblée
    pub target_threat: ThreatType,
    /// Action de défense primaire
    pub primary_action: DefenseAction,
    /// Actions secondaires
    pub secondary_actions: Vec<DefenseAction>,
    /// Force de la réponse (0.0-1.0)
    pub strength: f64,
    /// Durée d'application
    pub duration: Duration,
    /// Signature unique de la réponse
    pub response_signature: Vec<u8>,
    /// Moment de création
    pub creation_time: Instant,
    /// Efficacité observée (mise à jour après application)
    pub observed_efficacy: Option<f64>,
}

/// Action défensive spécifique
#[derive(Clone, Debug)]
pub enum DefenseAction {
    /// Bloquer une source spécifique
    Block(Vec<u8>),
    /// Imposer un délai à une source
    RateLimit(Vec<u8>, Duration),
    /// Mettre une source en quarantaine pour observation
    Quarantine(Vec<u8>, Duration),
    /// Rejeter un bloc spécifique
    RejectBlock(Vec<u8>),
    /// Rejeter une transaction spécifique
    RejectTransaction(Vec<u8>),
    /// Valider une ressource avec un niveau de scrutin plus élevé
    EnhancedValidation(Vec<u8>),
    /// Demander un consensus particulier
    RequestConsensusCheck(Vec<u8>),
    /// Déclencher une auto-réparation
    TriggerSelfHealing(String, Vec<u8>),
    /// Alerter les autres nœuds d'une menace
    AlertNetwork(ThreatType, Vec<u8>),
    /// Reconfigurer dynamiquement un paramètre
    ReconfigureParameter(String, Vec<u8>),
}

/// Message pour communication intracellulaire du système immunitaire
#[derive(Clone, Debug)]
pub enum ImmuneMessage {
    /// Alerte d'une menace détectée
    ThreatDetected(Threat),
    /// Réponse immunitaire déclenchée
    ResponseActivated(ImmuneResponse),
    /// Rapport d'efficacité d'une réponse
    EfficacyReport(Vec<u8>, f64),
    /// Demande d'aide immunitaire
    RequestAssistance(ThreatType, Vec<u8>),
    /// Offre d'assistance immunitaire
    OfferAssistance(ThreatType, Vec<ImmuneResponse>),
}

/// Cellule mémoire immunitaire pour reconnaissance rapide des menaces connues
#[derive(Clone, Debug)]
struct MemoryCell {
    /// Signature de la menace reconnue
    threat_signature: Vec<u8>,
    /// Type de menace
    threat_type: ThreatType,
    /// Réponses prouvées efficaces
    effective_responses: Vec<ImmuneResponse>,
    /// Nombre de détections réussies
    detection_count: u32,
    /// Dernière activation
    last_activated: Instant,
    /// Efficacité moyenne
    average_efficacy: f64,
}

/// Système immunitaire vivant de la blockchain
pub struct ImmuneGuard {
    // État du système immunitaire
    active: Arc<RwLock<bool>>,
    alert_level: Arc<RwLock<f64>>,
    system_health: Arc<RwLock<f64>>,
    
    // Mémoire immunitaire adaptative
    memory_cells: Arc<Mutex<HashMap<Vec<u8>, MemoryCell>>>,
    threat_signatures: Arc<RwLock<HashMap<ThreatType, Vec<Vec<u8>>>>>,
    
    // Coordination des réponses
    active_responses: Arc<Mutex<Vec<ImmuneResponse>>>,
    pending_responses: Arc<Mutex<VecDeque<ImmuneResponse>>>,
    
    // File d'alertes de menaces
    threat_alerts: Arc<Mutex<VecDeque<Threat>>>,
    
    // Statistiques et état
    detected_threats_count: Arc<RwLock<HashMap<ThreatType, u64>>>,
    false_positives: Arc<RwLock<u64>>,
    system_birth_time: Instant,
    last_assessment_time: Arc<RwLock<Instant>>,
    
    // Seuils adaptatifs
    detection_thresholds: Arc<RwLock<HashMap<ThreatType, f64>>>,
    response_thresholds: Arc<RwLock<HashMap<ThreatType, f64>>>,
    
    // Communications intercellulaires
    message_channel: Arc<Mutex<VecDeque<ImmuneMessage>>>,
    network_channel: Option<Arc<Mutex<VecDeque<ImmuneMessage>>>>,
    
    // Liens avec d'autres systèmes
    neural_link: Option<Arc<RwLock<f64>>>,
    metabolic_link: Option<Arc<RwLock<f64>>>,
}

impl ImmuneGuard {
    /// Crée un nouveau système immunitaire pour la blockchain
    pub fn new() -> Self {
        let mut detection_thresholds = HashMap::new();
        detection_thresholds.insert(ThreatType::DenialOfService, 0.6);
        detection_thresholds.insert(ThreatType::DoubleSpend, 0.8);
        detection_thresholds.insert(ThreatType::InvalidBlock, 0.7);
        detection_thresholds.insert(ThreatType::InvalidTransaction, 0.75);
        detection_thresholds.insert(ThreatType::MaliciousNode, 0.85);
        detection_thresholds.insert(ThreatType::MaliciousFork, 0.9);
        detection_thresholds.insert(ThreatType::Exploitation, 0.8);
        detection_thresholds.insert(ThreatType::BehavioralAnomaly, 0.65);
        detection_thresholds.insert(ThreatType::DataCorruption, 0.7);
        detection_thresholds.insert(ThreatType::Unknown, 0.95);
        
        let response_thresholds = detection_thresholds.clone();
        
        let immune_system = Self {
            active: Arc::new(RwLock::new(true)),
            alert_level: Arc::new(RwLock::new(0.0)),
            system_health: Arc::new(RwLock::new(1.0)),
            
            memory_cells: Arc::new(Mutex::new(HashMap::new())),
            threat_signatures: Arc::new(RwLock::new(HashMap::new())),
            
            active_responses: Arc::new(Mutex::new(Vec::new())),
            pending_responses: Arc::new(Mutex::new(VecDeque::new())),
            
            threat_alerts: Arc::new(Mutex::new(VecDeque::new())),
            
            detected_threats_count: Arc::new(RwLock::new(HashMap::new())),
            false_positives: Arc::new(RwLock::new(0)),
            system_birth_time: Instant::now(),
            last_assessment_time: Arc::new(RwLock::new(Instant::now())),
            
            detection_thresholds: Arc::new(RwLock::new(detection_thresholds)),
            response_thresholds: Arc::new(RwLock::new(response_thresholds)),
            
            message_channel: Arc::new(Mutex::new(VecDeque::new())),
            network_channel: None,
            
            neural_link: None,
            metabolic_link: None,
        };
        
        // Démarrer le système immunitaire actif et autonome
        immune_system.activate_immune_system();
        
        info!("Système immunitaire blockchain créé et activé à {:?}", immune_system.system_birth_time);
        immune_system
    }
    
    /// Active le système immunitaire dans un thread autonome
    fn activate_immune_system(&self) {
        // Cloner les références nécessaires
        let active = Arc::clone(&self.active);
        let alert_level = Arc::clone(&self.alert_level);
        let system_health = Arc::clone(&self.system_health);
        let memory_cells = Arc::clone(&self.memory_cells);
        let threat_alerts = Arc::clone(&self.threat_alerts);
        let active_responses = Arc::clone(&self.active_responses);
        let pending_responses = Arc::clone(&self.pending_responses);
        let message_channel = Arc::clone(&self.message_channel);
        let detected_threats_count = Arc::clone(&self.detected_threats_count);
        let last_assessment_time = Arc::clone(&self.last_assessment_time);
        
        // Lancer le thread autonome du système immunitaire
        std::thread::spawn(move || {
            info!("Système immunitaire activé - surveillance continue démarrée");
            let patrol_cycle = Duration::from_millis(10); // 100 Hz
            
            loop {
                // Vérifier si le système immunitaire est actif
                let is_active = match active.read() {
                    Ok(active_state) => *active_state,
                    Err(_) => true, // Défaut actif en cas d'erreur
                };
                
                if !is_active {
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
                
                // 1. Traiter les alertes de menaces
                if let Ok(mut alerts) = threat_alerts.lock() {
                    if !alerts.is_empty() {
                        // Traiter jusqu'à 10 alertes par cycle pour limiter la charge
                        let process_count = alerts.len().min(10);
                        
                        for _ in 0..process_count {
                            if let Some(threat) = alerts.pop_front() {
                                // Ignorer les menaces de faible confiance
                                if threat.confidence > THREAT_DETECTION_THRESHOLD {
                                    Self::process_threat(
                                        &threat, 
                                        &memory_cells, 
                                        &pending_responses,
                                        &detected_threats_count, 
                                        &alert_level
                                    );
                                }
                            }
                        }
                    }
                }
                
                // 2. Appliquer les réponses immunitaires
                if let Ok(mut pending) = pending_responses.lock() {
                    if !pending.is_empty() {
                        if let Ok(mut active) = active_responses.lock() {
                            // Activer jusqu'à 5 réponses par cycle
                            let process_count = pending.len().min(5);
                            
                            for _ in 0..process_count {
                                if let Some(response) = pending.pop_front() {
                                    debug!("Activating immune response: {:?} against {:?}", 
                                          response.primary_action, response.target_threat);
                                    active.push(response.clone());
                                    
                                    // Publier un message sur l'activation
                                    if let Ok(mut channel) = message_channel.lock() {
                                        channel.push_back(ImmuneMessage::ResponseActivated(response));
                                    }
                                    
                                    // Augmenter le niveau d'alerte
                                    if let Ok(mut alert) = alert_level.write() {
                                        *alert = (*alert * 0.8) + (response.strength * 0.2);
                                        *alert = alert.min(1.0);
                                    }
                                }
                            }
                        }
                    }
                }
                
                // 3. Gérer les réponses actives (expiration, efficacité)
                if let Ok(mut actives) = active_responses.lock() {
                    let now = Instant::now();
                    
                    // Filtrer les réponses expirées
                    actives.retain(|response| {
                        let expired = now.duration_since(response.creation_time) > response.duration;
                        if expired {
                            debug!("Immune response expired: {:?}", response.primary_action);
                        }
                        !expired
                    });
                    
                    // Décliner progressivement le niveau d'alerte si peu de réponses actives
                    if actives.len() < 3 {
                        if let Ok(mut alert) = alert_level.write() {
                            *alert *= 0.99; // Déclin progressif
                        }
                    }
                }
                
                // 4. Auto-guérison progressive
                if let Ok(mut health) = system_health.write() {
                    if *health < 1.0 {
                        *health += AUTO_HEALING_FACTOR * patrol_cycle.as_secs_f64();
                        *health = health.min(1.0);
                    }
                }
                
                // 5. Évaluation périodique du système immunitaire (toutes les 10 secondes)
                if let Ok(mut last_time) = last_assessment_time.write() {
                    if last_time.elapsed() > Duration::from_secs(10) {
                        *last_time = Instant::now();
                        
                        // Évaluation et adaptation des seuils de détection
                        Self::evaluate_immune_system(
                            &memory_cells, 
                            &detected_threats_count,
                            &system_health,
                            &alert_level
                        );
                    }
                }
                
                // Pause courte pour éviter la surcharge CPU
                std::thread::sleep(patrol_cycle);
            }
        });
    }
    
    /// Traiter une menace détectée
    fn process_threat(
        threat: &Threat,
        memory_cells: &Arc<Mutex<HashMap<Vec<u8>, MemoryCell>>>,
        pending_responses: &Arc<Mutex<VecDeque<ImmuneResponse>>>,
        detected_threats_count: &Arc<RwLock<HashMap<ThreatType, u64>>>,
        alert_level: &Arc<RwLock<f64>>
    ) {
        // Mettre à jour les statistiques
        if let Ok(mut counts) = detected_threats_count.write() {
            let count = counts.entry(threat.threat_type.clone()).or_insert(0);
            *count += 1;
        }
        
        // Ajuster le niveau d'alerte
        if let Ok(mut level) = alert_level.write() {
            let alert_increase = threat.severity * threat.confidence * 0.3;
            *level = (*level * 0.7) + alert_increase;
            *level = level.min(1.0);
        }
        
        // 1. Vérifier si c'est une menace connue
        let mut memory_match = None;
        let mut memory_response = None;
        
        if let Ok(memory) = memory_cells.lock() {
            for (sig, cell) in memory.iter() {
                if Self::signature_similarity(&threat.signature, sig) > 0.8 {
                    memory_match = Some(cell.clone());
                    
                    // Trouver la meilleure réponse passée
                    if !cell.effective_responses.is_empty() {
                        let best_response = cell.effective_responses
                            .iter()
                            .max_by(|a, b| {
                                a.observed_efficacy.unwrap_or(0.0)
                                 .partial_cmp(&b.observed_efficacy.unwrap_or(0.0))
                                 .unwrap_or(std::cmp::Ordering::Equal)
                            });
                        
                        if let Some(response) = best_response {
                            memory_response = Some(response.clone());
                        }
                    }
                    
                    break;
                }
            }
        }
        
        // 2. Générer une réponse immunitaire
        let immune_response = if let Some(response) = memory_response {
            // Utiliser une réponse mémoire (plus rapide)
            debug!("Menace reconnue! Utilisation d'une réponse mémoire");
            
            // Adapter légèrement la réponse au contexte actuel
            let mut updated_response = response;
            updated_response.strength = threat.severity.max(0.5);
            updated_response.creation_time = Instant::now();
            updated_response.observed_efficacy = None;
            
            updated_response
        } else {
            // Générer une nouvelle réponse (plus lente)
            debug!("Nouvelle menace détectée! Génération d'anticorps");
            
            // Simuler le temps de génération d'anticorps
            std::thread::sleep(Duration::from_millis(ANTIBODY_GENERATION_TIME_MS));
            
            // Créer une réponse appropriée selon le type de menace
            Self::generate_immune_response(threat)
        };
        
        // 3. Stocker la réponse pour exécution
        if let Ok(mut pending) = pending_responses.lock() {
            pending.push_back(immune_response.clone());
        }
        
        // 4. Mettre à jour la mémoire immunitaire avec cette menace et réponse
        if let Ok(mut memory) = memory_cells.lock() {
            if let Some(mut cell) = memory_match {
                // Mettre à jour une cellule existante
                cell.detection_count += 1;
                cell.last_activated = Instant::now();
                
                // Ajouter cette réponse si nouvelle
                let response_exists = cell.effective_responses.iter()
                    .any(|r| Self::response_similarity(r, &immune_response) > 0.9);
                
                if !response_exists {
                    cell.effective_responses.push(immune_response);
                }
                
                memory.insert(threat.signature.clone(), cell);
            } else {
                // Créer une nouvelle cellule mémoire
                let cell = MemoryCell {
                    threat_signature: threat.signature.clone(),
                    threat_type: threat.threat_type.clone(),
                    effective_responses: vec![immune_response],
                    detection_count: 1,
                    last_activated: Instant::now(),
                    average_efficacy: 0.0,
                };
                
                memory.insert(threat.signature.clone(), cell);
                
                // Limiter la taille de la mémoire immunologique
                if memory.len() > MEMORY_CELL_CAPACITY {
                    // Supprimer la cellule la moins utilisée
                    if let Some((oldest_key, _)) = memory.iter()
                        .min_by_key(|(_, cell)| cell.detection_count) {
                        memory.remove(&oldest_key.clone());
                    }
                }
            }
        }
    }
    
    /// Génère une réponse immunitaire adaptée à une menace spécifique
    fn generate_immune_response(threat: &Threat) -> ImmuneResponse {
        let mut rng = thread_rng();
        
        // Déterminer l'action principale selon le type de menace
        let primary_action = match threat.threat_type {
            ThreatType::DenialOfService => {
                DefenseAction::RateLimit(threat.source.clone(), Duration::from_secs(300))
            },
            ThreatType::DoubleSpend => {
                DefenseAction::RejectTransaction(threat.source.clone())
            },
            ThreatType::InvalidBlock => {
                DefenseAction::RejectBlock(threat.source.clone())
            },
            ThreatType::InvalidTransaction => {
                DefenseAction::RejectTransaction(threat.source.clone())
            },
            ThreatType::MaliciousNode => {
                DefenseAction::Block(threat.source.clone())
            },
            ThreatType::MaliciousFork => {
                DefenseAction::RequestConsensusCheck(threat.source.clone())
            },
            ThreatType::Exploitation => {
                DefenseAction::Quarantine(threat.source.clone(), Duration::from_secs(600))
            },
            ThreatType::BehavioralAnomaly => {
                DefenseAction::EnhancedValidation(threat.source.clone())
            },
            ThreatType::DataCorruption => {
                DefenseAction::TriggerSelfHealing("data_integrity".to_string(), threat.source.clone())
            },
            ThreatType::Unknown => {
                if rng.gen_bool(0.7) {
                    DefenseAction::Quarantine(threat.source.clone(), Duration::from_secs(300))
                } else {
                    DefenseAction::AlertNetwork(ThreatType::Unknown, threat.source.clone())
                }
            },
        };
        
        // Générer des actions secondaires complémentaires
        let mut secondary_actions = Vec::new();
        
        // Toujours alerter le réseau pour les menaces graves
        if threat.severity > 0.8 {
            secondary_actions.push(DefenseAction::AlertNetwork(
                threat.threat_type.clone(), 
                threat.source.clone()
            ));
        }
        
        // Ajouter une surveillance accrue pour les menaces incertaines
        if threat.confidence < 0.9 && threat.severity > 0.5 {
            secondary_actions.push(DefenseAction::EnhancedValidation(
                threat.source.clone()
            ));
        }
        
        // Déclencher l'auto-guérison pour les menaces de corruption
        if matches!(threat.threat_type, 
                   ThreatType::DataCorruption | 
                   ThreatType::MaliciousFork) {
            secondary_actions.push(DefenseAction::TriggerSelfHealing(
                "system_integrity".to_string(), 
                threat.signature.clone()
            ));
        }
        
        // Générer une signature unique pour cette réponse
        let mut response_signature = Vec::with_capacity(32);
        for _ in 0..32 {
            response_signature.push(rng.gen());
        }
        
        // Déterminer la durée de la réponse selon la gravité
        let duration_minutes = if threat.severity < 0.3 {
            5
        } else if threat.severity < 0.6 {
            30
        } else if threat.severity < 0.9 {
            120
        } else {
            1440 // 24 heures
        };
        
        ImmuneResponse {
            target_threat: threat.threat_type.clone(),
            primary_action,
            secondary_actions,
            strength: threat.severity * (0.8 + rng.gen::<f64>() * 0.4), // Légère variation
            duration: Duration::from_secs(duration_minutes * 60),
            response_signature,
            creation_time: Instant::now(),
            observed_efficacy: None,
        }
    }
    
    /// Évalue et adapte le système immunitaire périodiquement
    fn evaluate_immune_system(
        memory_cells: &Arc<Mutex<HashMap<Vec<u8>, MemoryCell>>>,
        detected_threats_count: &Arc<RwLock<HashMap<ThreatType, u64>>>,
        system_health: &Arc<RwLock<f64>>,
        alert_level: &Arc<RwLock<f64>>
    ) {
        // Calculer un score de santé basé sur les menaces récentes
        let threat_count = match detected_threats_count.read() {
            Ok(counts) => {
                let mut total = 0;
                for (_type, count) in counts.iter() {
                    total += *count;
                }
                total
            },
            Err(_) => 0,
        };
        
        // Ajuster la santé du système
        if let Ok(mut health) = system_health.write() {
            if threat_count > 100 {
                *health -= 0.05; // Beaucoup de menaces = santé réduite
            } else if threat_count < 10 {
                *health += 0.01; // Peu de menaces = récupération
            }
            
            *health = health.max(0.0).min(1.0);
        }
        
        // Analyser l'efficacité des réponses immunitaires
        if let Ok(memory) = memory_cells.lock() {
            if !memory.is_empty() {
                let memory_size = memory.len();
                let avg_detections: f64 = memory.values()
                    .map(|cell| cell.detection_count as f64)
                    .sum::<f64>() / memory_size as f64;
                
                if avg_detections > 5.0 {
                    // Le système est expérimenté, réduire progressivement le niveau d'alerte
                    if let Ok(mut alert) = alert_level.write() {
                        *alert *= 0.95;
                    }
                }
            }
        }
    }
    
    /// Détecte une menace potentielle dans le système
    pub fn detect_threat(&self, 
                        threat_type: ThreatType,
                        source: Vec<u8>,
                        signature: Vec<u8>,
                        severity: f64,
                        confidence: f64,
                        context: HashMap<String, Vec<u8>>) -> bool {
        // Vérifier si le système est actif
        let is_active = match self.active.read() {
            Ok(active) => *active,
            Err(_) => return false,
        };
        
        if !is_active {
            return false;
        }
        
        // Vérifier le seuil de détection pour ce type de menace
        let threshold = match self.detection_thresholds.read() {
            Ok(thresholds) => *thresholds.get(&threat_type).unwrap_or(&THREAT_DETECTION_THRESHOLD),
            Err(_) => THREAT_DETECTION_THRESHOLD,
        };
        
        // Si la confiance est trop basse, ignorer (réduit les faux positifs)
        if confidence < threshold {
            // Incrémenter le compteur de faux positifs
            if let Ok(mut fp) = self.false_positives.write() {
                *fp += 1;
            }
            
            debug!("Alerte ignorée: confiance trop basse ({:.2}) < seuil ({:.2}) pour {:?}", 
                  confidence, threshold, threat_type);
            return false;
        }
        
        // Créer la menace
        let threat = Threat {
            threat_type,
            source,
            severity: severity.max(0.0).min(1.0),
            confidence: confidence.max(0.0).min(1.0),
            signature,
            context,
            detection_time: Instant::now(),
            correlation_count: 0, // Sera mis à jour ultérieurement
        };
        
        // Ajouter à la file des menaces
        if let Ok(mut alerts) = self.threat_alerts.lock() {
            alerts.push_back(threat);
            
            // Limiter la taille de la file
            if alerts.len() > 1000 {
                alerts.pop_front();
            }
            
            debug!("Menace détectée et mise en file: {:?} (sévérité: {:.2}, confiance: {:.2})",
                  threat.threat_type, severity, confidence);
            true
        } else {
            error!("Impossible d'accéder à la file de menaces");
            false
        }
    }
    
    /// Évalue l'efficacité d'une réponse immunitaire
    pub fn report_response_efficacy(&self, response_signature: Vec<u8>, efficacy: f64) {
        // Trouver la réponse active correspondante
        if let Ok(mut active_responses) = self.active_responses.lock() {
            for response in active_responses.iter_mut() {
                if response.response_signature == response_signature {
                    // Mettre à jour l'efficacité observée
                    response.observed_efficacy = Some(efficacy);
                    
                    // Mettre à jour la cellule mémoire correspondante
                    if let Ok(mut memory) = self.memory_cells.lock() {
                        for cell in memory.values_mut() {
                            for stored_response in cell.effective_responses.iter_mut() {
                                if stored_response.response_signature == response_signature {
                                    stored_response.observed_efficacy = Some(efficacy);
                                    
                                    // Mettre à jour l'efficacité moyenne
                                    let old_avg = cell.average_efficacy;
                                    let old_weight = cell.detection_count.saturating_sub(1) as f64;
                                    let total = (old_avg * old_weight) + efficacy;
                                    cell.average_efficacy = total / cell.detection_count as f64;
                                    
                                    debug!("Efficacité de réponse mise à jour: {:.2} pour la menace {:?}",
                                          efficacy, cell.threat_type);
                                    break;
                                }
                            }
                        }
                    }
                    
                    break;
                }
            }
        }
    }
    
    /// Communique avec d'autres systèmes immunitaires
    pub fn communicate(&self, message: ImmuneMessage) {
        // Ajouter au canal de messages local
        if let Ok(mut channel) = self.message_channel.lock() {
            channel.push_back(message.clone());
        }
        
        // Si un canal réseau est disponible, y envoyer également
        if let Some(ref network) = self.network_channel {
            if let Ok(mut channel) = network.lock() {
                channel.push_back(message);
            }
        }
    }
    
    /// Connecte ce système immunitaire au système neural
    pub fn connect_to_neural(&mut self, neural_link: Arc<RwLock<f64>>) {
        self.neural_link = Some(neural_link);
        info!("Système immunitaire connecté au système neural");
    }
    
    /// Vérifie la similarité entre deux signatures
    fn signature_similarity(sig1: &[u8], sig2: &[u8]) -> f64 {
        if sig1.len() != sig2.len() || sig1.is_empty() {
            return 0.0;
        }
        
        let mut matches = 0;
        for (a, b) in sig1.iter().zip(sig2.iter()) {
            if a == b {
                matches += 1;
            }
        }
        
        matches as f64 / sig1.len() as f64
    }
    
    /// Vérifie la similarité entre deux réponses
    fn response_similarity(resp1: &ImmuneResponse, resp2: &ImmuneResponse) -> f64 {
        if resp1.target_threat != resp2.target_threat {
            return 0.0;
        }
        
        // Similarité basée sur l'action principale
        let primary_similar = match (&resp1.primary_action, &resp2.primary_action) {
            (DefenseAction::Block(a), DefenseAction::Block(b)) => 
                Self::signature_similarity(a, b),
            (DefenseAction::RateLimit(a, _), DefenseAction::RateLimit(b, _)) => 
                Self::signature_similarity(a, b),
            (DefenseAction::Quarantine(a, _), DefenseAction::Quarantine(b, _)) => 
                Self::signature_similarity(a, b),
            (DefenseAction::RejectBlock(a), DefenseAction::RejectBlock(b)) => 
                Self::signature_similarity(a, b),
            (DefenseAction::RejectTransaction(a), DefenseAction::RejectTransaction(b)) => 
                Self::signature_similarity(a, b),
            (DefenseAction::EnhancedValidation(a), DefenseAction::EnhancedValidation(b)) => 
                Self::signature_similarity(a, b),
            (DefenseAction::RequestConsensusCheck(a), DefenseAction::RequestConsensusCheck(b)) => 
                Self::signature_similarity(a, b),
            (DefenseAction::TriggerSelfHealing(a, _), DefenseAction::TriggerSelfHealing(b, _)) => 
                if a == b { 1.0 } else { 0.5 },
            (DefenseAction::AlertNetwork(a, _), DefenseAction::AlertNetwork(b, _)) => 
                if a == b { 1.0 } else { 0.5 },
            (DefenseAction::ReconfigureParameter(a, _), DefenseAction::ReconfigureParameter(b, _)) => 
                if a == b { 1.0 } else { 0.5 },
            _ => 0.0,
        };
        
        // Similarité de force
        let strength_diff = (resp1.strength - resp2.strength).abs();
        let strength_similar = 1.0 - (strength_diff / 1.0).min(1.0);
        
        // Combiner les similarités
        (primary_similar * 0.7) + (strength_similar * 0.3)
    }
    
    /// Récupère les statistiques vitales du système immunitaire
    pub fn get_vital_stats(&self) -> ImmuneSystemStats {
        let alert_level = match self.alert_level.read() {
            Ok(level) => *level,
            Err(_) => 0.0,
        };
        
        let system_health = match self.system_health.read() {
            Ok(health) => *health,
            Err(_) => 1.0,
        };
        
        let memory_size = match self.memory_cells.lock() {
            Ok(memory) => memory.len(),
            Err(_) => 0,
        };
        
        let active_responses = match self.active_responses.lock() {
            Ok(responses) => responses.len(),
            Err(_) => 0,
        };
        
        let false_positives = match self.false_positives.read() {
            Ok(fp) => *fp,
            Err(_) => 0,
        };
        
        let detected_threats = match self.detected_threats_count.read() {
            Ok(counts) => {
                let mut total = 0;
                for (_type, count) in counts.iter() {
                    total += *count;
                }
                total
            },
            Err(_) => 0,
        };
        
        ImmuneSystemStats {
            alert_level,
            system_health,
            memory_cells_count: memory_size,
            active_responses_count: active_responses,
            detected_threats_count: detected_threats,
            false_positives_count: false_positives,
            age_seconds: self.system_birth_time.elapsed().as_secs(),
        }
    }
}

/// Statistiques vitales du système immunitaire
#[derive(Debug)]
pub struct ImmuneSystemStats {
    pub alert_level: f64,
    pub system_health: f64,
    pub memory_cells_count: usize,
    pub active_responses_count: usize,
    pub detected_threats_count: u64,
    pub false_positives_count: u64,
    pub age_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_immune_system_creation() {
        let immune = ImmuneGuard::new();
        let stats = immune.get_vital_stats();
        
        assert_eq!(stats.memory_cells_count, 0);    // Commence sans mémoire immunitaire
        assert_eq!(stats.active_responses_count, 0); // Pas de réponses actives
        assert_eq!(stats.false_positives_count, 0);  // Pas de faux positifs
        assert!(stats.system_health > 0.99);         // Commence en bonne santé
    }
    
    #[test]
    fn test_threat_detection() {
        let immune = ImmuneGuard::new();
        
        // Simuler des menaces
        let mut context = HashMap::new();
        context.insert("test".to_string(), vec![1, 2, 3]);
        
        let is_detected = immune.detect_threat(
            ThreatType::DenialOfService,
            vec![127, 0, 0, 1],
            vec![1, 2, 3, 4, 5],
            0.8,  // Haute sévérité
            0.9,  // Haute confiance
            context
        );
        
        // La menace devrait être détectée
        assert!(is_detected);
        
        // Attendre que la menace soit traitée
        std::thread::sleep(Duration::from_millis(100));
        
        // Vérifier les statistiques
        let stats = immune.get_vital_stats();
        assert!(stats.detected_threats_count >= 1);
        assert!(stats.alert_level > 0.0);
    }
}
