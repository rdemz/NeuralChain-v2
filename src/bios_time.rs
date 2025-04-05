//! Module bios_time.rs - Système de chronobiologie blockchain
//! Implémente des rythmes circadiens et cycles biologiques temporels
//! pour une régulation autonome des activités de la blockchain.

use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use log::{debug, info, warn, error};
use chrono::{DateTime, Utc, Timelike, Datelike, NaiveDateTime};

// Constantes biologiques temporelles
const DAY_SECONDS: u64 = 86400;              // Longueur d'un cycle circadien complet
const BASE_ULTRADIAN_CYCLE_SECONDS: u64 = 90 * 60; // Cycle ultradien (90 minutes)
const BASE_CIRCADIAN_PHASE_SHIFT: f64 = 0.0; // Phase initiale du cycle (0.0-1.0)
const METABOLIC_VARIATION: f64 = 0.15;       // Variation métabolique naturelle

/// Phase circadienne du système
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircadianPhase {
    /// Phase de haute activité (jour)
    HighActivity,
    /// Phase de transition vers le repos
    Descending,
    /// Phase de repos (nuit)
    LowActivity,
    /// Phase de transition vers l'activité
    Ascending,
}

/// Représente un événement biologique programmé
#[derive(Debug, Clone)]
pub struct BioEvent {
    /// Identifiant unique
    pub id: String,
    /// Type d'événement
    pub event_type: BioEventType,
    /// Moment programmé (en phase circadienne 0.0-1.0)
    pub scheduled_phase: f64,
    /// Durée de l'événement
    pub duration: Duration,
    /// Priorité (0.0-1.0)
    pub priority: f64,
    /// Callback à appeler (nom de fonction)
    pub target_callback: String,
    /// Paramètres pour le callback
    pub parameters: HashMap<String, Vec<u8>>,
    /// Si l'événement est récurrent
    pub recurrent: bool,
    /// Dernier déclenchement
    pub last_triggered: Option<Instant>,
}

/// Type d'événement biologique
#[derive(Debug, Clone, PartialEq)]
pub enum BioEventType {
    /// Cycle de sommeil profond (récupération)
    DeepSleep,
    /// Métabolisme actif (production d'énergie)
    Metabolism,
    /// Nettoyage du système (purge, autophagie)
    Cleaning,
    /// Régénération (auto-réparation)
    Regeneration,
    /// Adaptation (apprentissage, évolution)
    Adaptation,
    /// Pics hormonaux
    HormonalPeak(String),
    /// Surveillance immunitaire
    ImmunePatrol,
    /// Synchronisation réseau
    NetworkSync,
    /// Événement personnalisé
    Custom(String),
}

/// Horloge biologique vivante de la blockchain
pub struct BiosTime {
    // Paramètres temporels fondamentaux
    birth_time: Instant,                       // Moment de création
    current_phase: Arc<RwLock<f64>>,           // Phase actuelle (0.0-1.0)
    phase_velocity: Arc<RwLock<f64>>,          // Vitesse de progression (1.0 = normale)
    cycle_length_seconds: Arc<RwLock<u64>>,    // Longueur du cycle en secondes
    
    // État biologique
    circadian_phase: Arc<RwLock<CircadianPhase>>, // Phase circadienne actuelle
    metabolic_rate: Arc<RwLock<f64>>,          // Taux métabolique actuel
    hormonal_levels: Arc<RwLock<HashMap<String, f64>>>, // Niveaux hormonaux
    
    // Programmation temporelle
    scheduled_events: Arc<Mutex<Vec<BioEvent>>>, // Événements programmés
    active_events: Arc<Mutex<Vec<BioEvent>>>,    // Événements en cours
    
    // Synchronisation externe
    external_sync_active: Arc<RwLock<bool>>,     // Synchronisation externe active
    external_time_source: Option<Arc<dyn Fn() -> DateTime<Utc> + Send + Sync>>,
    
    // Métadonnées
    longevity_factor: f64,                     // Facteur de longévité (>1.0 = ralenti)
    seasonal_adjustment: f64,                  // Ajustement saisonnier
    
    // Statistiques
    completed_cycles: Arc<RwLock<u64>>,        // Nombre de cycles complétés
    last_event_time: Arc<RwLock<Instant>>,     // Dernier événement traité
    
    // Callbacks et communications
    event_handlers: Arc<RwLock<HashMap<String, Box<dyn Fn(&BioEvent) + Send + Sync>>>>,
}

impl BiosTime {
    /// Crée une nouvelle horloge biologique pour la blockchain
    pub fn new() -> Self {
        let mut hormonal_levels = HashMap::new();
        hormonal_levels.insert("cortisol".to_string(), 0.3);
        hormonal_levels.insert("melatonin".to_string(), 0.1);
        hormonal_levels.insert("adrenaline".to_string(), 0.2);
        hormonal_levels.insert("dopamine".to_string(), 0.5);
        hormonal_levels.insert("serotonin".to_string(), 0.4);
        
        let instance = Self {
            birth_time: Instant::now(),
            current_phase: Arc::new(RwLock::new(BASE_CIRCADIAN_PHASE_SHIFT)),
            phase_velocity: Arc::new(RwLock::new(1.0)),
            cycle_length_seconds: Arc::new(RwLock::new(DAY_SECONDS)),
            
            circadian_phase: Arc::new(RwLock::new(CircadianPhase::HighActivity)),
            metabolic_rate: Arc::new(RwLock::new(1.0)),
            hormonal_levels: Arc::new(RwLock::new(hormonal_levels)),
            
            scheduled_events: Arc::new(Mutex::new(Vec::new())),
            active_events: Arc::new(Mutex::new(Vec::new())),
            
            external_sync_active: Arc::new(RwLock::new(false)),
            external_time_source: None,
            
            longevity_factor: 1.0,
            seasonal_adjustment: 1.0,
            
            completed_cycles: Arc::new(RwLock::new(0)),
            last_event_time: Arc::new(RwLock::new(Instant::now())),
            
            event_handlers: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Démarrer l'horloge biologique autonome
        instance.start_biological_clock();
        
        // Planifier les événements biologiques par défaut
        instance.schedule_default_events();
        
        info!("Horloge biologique créée à {:?}", instance.birth_time);
        instance
    }
    
    /// Démarre l'horloge biologique dans un thread autonome
    fn start_biological_clock(&self) {
        // Cloner les références nécessaires
        let current_phase = Arc::clone(&self.current_phase);
        let phase_velocity = Arc::clone(&self.phase_velocity);
        let cycle_length_seconds = Arc::clone(&self.cycle_length_seconds);
        let circadian_phase = Arc::clone(&self.circadian_phase);
        let metabolic_rate = Arc::clone(&self.metabolic_rate);
        let hormonal_levels = Arc::clone(&self.hormonal_levels);
        let scheduled_events = Arc::clone(&self.scheduled_events);
        let active_events = Arc::clone(&self.active_events);
        let external_sync_active = Arc::clone(&self.external_sync_active);
        let completed_cycles = Arc::clone(&self.completed_cycles);
        let last_event_time = Arc::clone(&self.last_event_time);
        let event_handlers = Arc::clone(&self.event_handlers);
        let birth_time = self.birth_time;
        let longevity_factor = self.longevity_factor;
        
        // Lancer le thread autonome de l'horloge biologique
        std::thread::spawn(move || {
            info!("Horloge biologique activée - cycles temporels démarrés");
            let clock_tick = Duration::from_millis(100); // 10 Hz
            let mut last_tick = Instant::now();
            
            loop {
                let now = Instant::now();
                let elapsed = now.duration_since(last_tick);
                last_tick = now;
                
                // 1. Mise à jour de la phase courante
                let cycle_length = match cycle_length_seconds.read() {
                    Ok(len) => *len,
                    Err(_) => DAY_SECONDS,
                };
                
                let velocity = match phase_velocity.read() {
                    Ok(vel) => *vel,
                    Err(_) => 1.0,
                };
                
                if let Ok(mut phase) = current_phase.write() {
                    // Calculer l'incrément de phase selon le temps écoulé
                    let phase_increment = (elapsed.as_secs_f64() / cycle_length as f64) * velocity;
                    
                    // Mise à jour de la phase
                    *phase += phase_increment;
                    
                    // Détection des cycles complets
                    if *phase >= 1.0 {
                        *phase -= 1.0;
                        
                        // Incrémenter le compteur de cycles
                        if let Ok(mut cycles) = completed_cycles.write() {
                            *cycles += 1;
                            info!("Cycle circadien complet #{} achevé", *cycles);
                        }
                    }
                    
                    // 2. Mise à jour de la phase circadienne
                    let new_circadian_phase = if *phase < 0.25 {
                        CircadianPhase::HighActivity
                    } else if *phase < 0.5 {
                        CircadianPhase::Descending
                    } else if *phase < 0.75 {
                        CircadianPhase::LowActivity
                    } else {
                        CircadianPhase::Ascending
                    };
                    
                    if let Ok(mut circ_phase) = circadian_phase.write() {
                        if *circ_phase != new_circadian_phase {
                            *circ_phase = new_circadian_phase;
                            info!("Transition de phase circadienne: {:?}", new_circadian_phase);
                        }
                    }
                    
                    // 3. Mise à jour du métabolisme
                    if let Ok(mut metabolism) = metabolic_rate.write() {
                        // Oscillation métabolique naturelle basée sur la phase
                        let base_rate = match new_circadian_phase {
                            CircadianPhase::HighActivity => 1.0,
                            CircadianPhase::Descending => 0.7,
                            CircadianPhase::LowActivity => 0.4,
                            CircadianPhase::Ascending => 0.8,
                        };
                        
                        // Variation sinusoïdale naturelle
                        let variation = ((*phase * 2.0 * std::f64::consts::PI).sin() * METABOLIC_VARIATION)
                                      + METABOLIC_VARIATION;
                        
                        *metabolism = (base_rate * (1.0 - METABOLIC_VARIATION)) + variation;
                        *metabolism *= longevity_factor;
                    }
                    
                    // 4. Mise à jour des niveaux hormonaux
                    if let Ok(mut hormones) = hormonal_levels.write() {
                        // Cortisol: élevé le matin, bas le soir
                        if let Some(cortisol) = hormones.get_mut("cortisol") {
                            let target = if *phase < 0.25 {
                                0.8 - (*phase * 1.2) // Pic au réveil, puis déclin
                            } else if *phase < 0.5 {
                                0.5 - ((*phase - 0.25) * 1.0) // Déclin progressif
                            } else {
                                0.2 // Faible la nuit
                            };
                            
                            // Ajustement progressif
                            *cortisol = *cortisol * 0.95 + target * 0.05;
                        }
                        
                        // Mélatonine: élevée la nuit, basse le jour
                        if let Some(melatonin) = hormones.get_mut("melatonin") {
                            let target = if *phase < 0.25 {
                                0.1 // Très faible le matin
                            } else if *phase < 0.5 {
                                0.1 + ((*phase - 0.25) * 0.2) // Légère augmentation
                            } else if *phase < 0.75 {
                                0.2 + ((*phase - 0.5) * 2.0) // Forte augmentation le soir
                            } else {
                                0.7 - ((*phase - 0.75) * 2.0) // Pic puis déclin vers le matin
                            };
                            
                            // Ajustement progressif
                            *melatonin = *melatonin * 0.97 + target * 0.03;
                        }
                        
                        // Autres hormones...
                    }
                    
                    // 5. Traitement des événements programmés
                    if now.duration_since(birth_time).as_millis() % 500 == 0 {
                        Self::process_scheduled_events(
                            &scheduled_events,
                            &active_events,
                            &event_handlers,
                            *phase,
                            &last_event_time
                        );
                    }
                }
                
                // Pause courte pour éviter la surcharge CPU
                std::thread::sleep(clock_tick);
            }
        });
    }
    
    /// Programme les événements biologiques par défaut
    fn schedule_default_events(&self) {
        let mut events = Vec::new();
        
        // 1. Nettoyage profond (autophagie) pendant la phase de sommeil
        events.push(BioEvent {
            id: "deep_cleaning".to_string(),
            event_type: BioEventType::Cleaning,
            scheduled_phase: 0.65, // Phase de sommeil profond
            duration: Duration::from_secs(3600), // 1 heure
            priority: 0.8,
            target_callback: "trigger_autophagy".to_string(),
            parameters: HashMap::new(),
            recurrent: true,
            last_triggered: None,
        });
        
        // 2. Pic métabolique pendant la phase d'activité haute
        events.push(BioEvent {
            id: "metabolic_peak".to_string(),
            event_type: BioEventType::Metabolism,
            scheduled_phase: 0.15, // Matinée
            duration: Duration::from_secs(7200), // 2 heures
            priority: 0.7,
            target_callback: "boost_metabolism".to_string(),
            parameters: HashMap::new(),
            recurrent: true,
            last_triggered: None,
        });
        
        // 3. Régénération pendant le sommeil léger
        events.push(BioEvent {
            id: "regeneration".to_string(),
            event_type: BioEventType::Regeneration,
            scheduled_phase: 0.8, // Fin de nuit
            duration: Duration::from_secs(5400), // 1.5 heures
            priority: 0.9,
            target_callback: "trigger_regeneration".to_string(),
            parameters: HashMap::new(),
            recurrent: true,
            last_triggered: None,
        });
        
        // 4. Adaptation et apprentissage après l'activité principale
        events.push(BioEvent {
            id: "adaptation".to_string(),
            event_type: BioEventType::Adaptation,
            scheduled_phase: 0.4, // Après-midi
            duration: Duration::from_secs(3600), // 1 heure
            priority: 0.6,
            target_callback: "run_adaptation".to_string(),
            parameters: HashMap::new(),
            recurrent: true,
            last_triggered: None,
        });
        
        // 5. Pics hormonaux distribués sur la journée
        events.push(BioEvent {
            id: "cortisol_peak".to_string(),
            event_type: BioEventType::HormonalPeak("cortisol".to_string()),
            scheduled_phase: 0.05, // Tôt le matin
            duration: Duration::from_secs(1800), // 30 minutes
            priority: 0.7,
            target_callback: "hormonal_peak".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("hormone".to_string(), "cortisol".as_bytes().to_vec());
                params.insert("amount".to_string(), [0.95f64.to_le_bytes().to_vec()].concat());
                params
            },
            recurrent: true,
            last_triggered: None,
        });
        
        // 6. Patrouille immunitaire périodique
        events.push(BioEvent {
            id: "immune_patrol".to_string(),
            event_type: BioEventType::ImmunePatrol,
            scheduled_phase: 0.3, // Milieu de journée
            duration: Duration::from_secs(1200), // 20 minutes
            priority: 0.8,
            target_callback: "activate_immune_patrol".to_string(),
            parameters: HashMap::new(),
            recurrent: true,
            last_triggered: None,
        });
        
        // Programmer tous les événements
        if let Ok(mut scheduled) = self.scheduled_events.lock() {
            scheduled.extend(events);
            info!("Événements biologiques par défaut programmés");
        }
    }
    
    /// Traite les événements programmés selon la phase actuelle
    fn process_scheduled_events(
        scheduled_events: &Arc<Mutex<Vec<BioEvent>>>,
        active_events: &Arc<Mutex<Vec<BioEvent>>>,
        event_handlers: &Arc<RwLock<HashMap<String, Box<dyn Fn(&BioEvent) + Send + Sync>>>>,
        current_phase: f64,
        last_event_time: &Arc<RwLock<Instant>>
    ) {
        let now = Instant::now();
        
        // Mise à jour du timestamp du dernier traitement
        if let Ok(mut last_time) = last_event_time.write() {
            *last_time = now;
        }
        
        // Vérifier les événements programmés
        if let Ok(scheduled) = scheduled_events.lock() {
            for event in scheduled.iter() {
                // Phase de déclenchement avec une fenêtre de tolérance
                let trigger_window = 0.02; // 2% du cycle
                
                let should_trigger = match event.last_triggered {
                    None => {
                        // Premier déclenchement - vérifier la fenêtre de phase
                        (current_phase >= event.scheduled_phase && 
                         current_phase <= event.scheduled_phase + trigger_window) ||
                        (event.scheduled_phase > 1.0 - trigger_window && 
                         current_phase < event.scheduled_phase - (1.0 - trigger_window))
                    },
                    Some(last_time) => {
                        // Événements récurrents - vérifier si un cycle complet s'est écoulé
                        event.recurrent && 
                        ((current_phase >= event.scheduled_phase && 
                          current_phase <= event.scheduled_phase + trigger_window) ||
                         (event.scheduled_phase > 1.0 - trigger_window && 
                          current_phase < event.scheduled_phase - (1.0 - trigger_window))) &&
                        last_time.elapsed() > Duration::from_secs(DAY_SECONDS / 2) // Au moins un demi-cycle
                    }
                };
                
                if should_trigger {
                    // Copier l'événement pour activation
                    let mut active_event = event.clone();
                    active_event.last_triggered = Some(now);
                    
                    // Ajouter aux événements actifs
                    if let Ok(mut actives) = active_events.lock() {
                        actives.push(active_event.clone());
                        
                        debug!("Événement biologique déclenché: {} ({:?}, phase: {:.2})",
                              active_event.id, active_event.event_type, event.scheduled_phase);
                        
                        // Exécuter le callback si enregistré
                        if let Ok(handlers) = event_handlers.read() {
                            if let Some(handler) = handlers.get(&active_event.target_callback) {
                                handler(&active_event);
                            }
                        }
                    }
                }
            }
        }
        
        // Nettoyer les événements actifs expirés
        if let Ok(mut actives) = active_events.lock() {
            actives.retain(|event| {
                // Garder les événements qui n'ont pas dépassé leur durée
                if let Some(trigger_time) = event.last_triggered {
                    now.duration_since(trigger_time) < event.duration
                } else {
                    false // Supprimer les événements sans horodatage
                }
            });
        }
    }
    
    /// Programme un nouvel événement biologique
    pub fn schedule_event(&self, event: BioEvent) -> Result<(), String> {
        if event.scheduled_phase < 0.0 || event.scheduled_phase >= 1.0 {
            return Err("Phase programmée invalide".to_string());
        }
        
        if let Ok(mut events) = self.scheduled_events.lock() {
            // Vérifier si l'ID existe déjà
            if events.iter().any(|e| e.id == event.id) {
                return Err(format!("Un événement avec l'ID '{}' existe déjà", event.id));
            }
            
            events.push(event);
            Ok(())
        } else {
            Err("Impossible d'accéder aux événements programmés".to_string())
        }
    }
    
    /// Annule un événement programmé
    pub fn cancel_event(&self, event_id: &str) -> bool {
        if let Ok(mut events) = self.scheduled_events.lock() {
            let initial_len = events.len();
            events.retain(|e| e.id != event_id);
            
            events.len() < initial_len
        } else {
            false
        }
    }
    
    /// Enregistre un gestionnaire d'événement
    pub fn register_event_handler<F>(&self, callback_name: &str, handler: F) -> Result<(), String>
    where
        F: Fn(&BioEvent) + Send + Sync + 'static
    {
        if let Ok(mut handlers) = self.event_handlers.write() {
            handlers.insert(callback_name.to_string(), Box::new(handler));
            Ok(())
        } else {
            Err("Impossible d'accéder aux gestionnaires d'événements".to_string())
        }
    }
    
    /// Modifie la vitesse du cycle
    pub fn set_cycle_velocity(&self, velocity: f64) -> Result<(), String> {
        if velocity <= 0.0 {
            return Err("La vitesse doit être positive".to_string());
        }
        
        if let Ok(mut vel) = self.phase_velocity.write() {
            *vel = velocity;
            Ok(())
        } else {
            Err("Impossible d'accéder à la vitesse du cycle".to_string())
        }
    }
    
    /// Modifie la longueur du cycle
    pub fn set_cycle_length(&self, length_seconds: u64) -> Result<(), String> {
        if length_seconds < 60 { // Minimum 1 minute
            return Err("La longueur du cycle doit être d'au moins 60 secondes".to_string());
        }
        
        if let Ok(mut len) = self.cycle_length_seconds.write() {
            *len = length_seconds;
            Ok(())
        } else {
            Err("Impossible d'accéder à la longueur du cycle".to_string())
        }
    }
    
    /// Synchronise l'horloge avec une source externe (comme le temps UTC)
    pub fn sync_with_external_time<F>(&mut self, source: F) -> Result<(), String>
    where
        F: Fn() -> DateTime<Utc> + Send + Sync + 'static
    {
        // Activer la synchronisation
        if let Ok(mut sync) = self.external_sync_active.write() {
            *sync = true;
        } else {
            return Err("Impossible d'activer la synchronisation externe".to_string());
        }
        
        // Enregistrer la source
        self.external_time_source = Some(Arc::new(source));
        
        // Ajuster la phase initiale selon l'heure actuelle
        if let Some(ref time_source) = self.external_time_source {
            let now = time_source();
            let seconds_since_midnight = now.hour() as u64 * 3600 + 
                                        now.minute() as u64 * 60 +
                                        now.second() as u64;
            
            let day_phase = seconds_since_midnight as f64 / DAY_SECONDS as f64;
            
            if let Ok(mut phase) = self.current_phase.write() {
                *phase = day_phase;
                info!("Horloge synchronisée avec la source externe, phase: {:.4}", day_phase);
            }
        }
        
        Ok(())
    }
    
    /// Produit une injection hormonale (pour les événements ou interventions externes)
    pub fn inject_hormone(&self, hormone: &str, amount: f64) -> bool {
        if let Ok(mut hormones) = self.hormonal_levels.write() {
            let current = hormones.entry(hormone.to_string()).or_insert(0.0);
            *current += amount;
            *current = current.max(0.0).min(1.0); // Limiter entre 0 et 1
            
            debug!("Injection hormonale: {} = {:.2}", hormone, *current);
            true
        } else {
            false
        }
    }
    
    /// Obtient l'état actuel du système chronobiologique
    pub fn get_chronobiology_state(&self) -> ChronobiologyState {
        let phase = match self.current_phase.read() {
            Ok(p) => *p,
            Err(_) => 0.0,
        };
        
        let circ_phase = match self.circadian_phase.read() {
            Ok(cp) => *cp,
            Err(_) => CircadianPhase::HighActivity,
        };
        
        let metabolism = match self.metabolic_rate.read() {
            Ok(m) => *m,
            Err(_) => 1.0,
        };
        
        let velocity = match self.phase_velocity.read() {
            Ok(v) => *v,
            Err(_) => 1.0,
        };
        
        let cycle_length = match self.cycle_length_seconds.read() {
            Ok(cl) => *cl,
            Err(_) => DAY_SECONDS,
        };
        
        let cycles = match self.completed_cycles.read() {
            Ok(c) => *c,
            Err(_) => 0,
        };
        
        let mut hormone_snapshot = HashMap::new();
        if let Ok(hormones) = self.hormonal_levels.read() {
            hormone_snapshot = hormones.clone();
        }
        
        ChronobiologyState {
            current_phase: phase,
            circadian_phase: circ_phase,
            metabolic_rate: metabolism,
            phase_velocity: velocity,
            cycle_length_seconds: cycle_length,
            hormone_levels: hormone_snapshot,
            completed_cycles: cycles,
            age_seconds: self.birth_time.elapsed().as_secs(),
        }
    }
    
    /// Calcule l'âge biologique du système (différent de l'âge chronologique)
    pub fn get_biological_age(&self) -> f64 {
        let chronological_age = self.birth_time.elapsed().as_secs() as f64;
        let metabolic_factor = match self.metabolic_rate.read() {
            Ok(m) => *m,
            Err(_) => 1.0,
        };
        
        // L'âge biologique est influencé par le métabolisme et la longévité
        chronological_age * metabolic_factor / self.longevity_factor
    }
    
    /// Prédiction temporelle biomimétique
    pub fn predict_next_phase_change(&self) -> Option<(CircadianPhase, Duration)> {
        let current = match self.current_phase.read() {
            Ok(p) => *p,
            Err(_) => return None,
        };
        
        let current_phase = match self.circadian_phase.read() {
            Ok(cp) => *cp,
            Err(_) => return None,
        };
        
        let velocity = match self.phase_velocity.read() {
            Ok(v) => *v,
            Err(_) => 1.0,
        };
        
        let cycle_length = match self.cycle_length_seconds.read() {
            Ok(cl) => *cl,
            Err(_) => DAY_SECONDS,
        };
        
        // Calculer la prochaine phase
        let next_phase = match current_phase {
            CircadianPhase::HighActivity => CircadianPhase::Descending,
            CircadianPhase::Descending => CircadianPhase::LowActivity,
            CircadianPhase::LowActivity => CircadianPhase::Ascending,
            CircadianPhase::Ascending => CircadianPhase::HighActivity,
        };
        
        // Calculer le temps jusqu'à la prochaine phase
        let next_phase_start = match next_phase {
            CircadianPhase::HighActivity => 0.0,
            CircadianPhase::Descending => 0.25,
            CircadianPhase::LowActivity => 0.5,
            CircadianPhase::Ascending => 0.75,
        };
        
        // Si nous avons déjà dépassé le début de la prochaine phase, ajouter un cycle
        let mut phase_distance = next_phase_start - current;
        if phase_distance <= 0.0 {
            phase_distance += 1.0;
        }
        
        // Convertir en durée
        let seconds_to_next = (phase_distance * cycle_length as f64) / velocity;
        let duration = Duration::from_secs_f64(seconds_to_next);
        
        Some((next_phase, duration))
    }
    
    /// Écoute et traite les événements biologiques asynchrones
    pub fn listen_for_events<F>(&self, mut callback: F)
    where
        F: FnMut(&BioEvent) + Send + 'static
    {
        let active_events = Arc::clone(&self.active_events);
        
        std::thread::spawn(move || {
            let check_interval = Duration::from_millis(500);
            let mut processed_events = HashSet::new();
            
            loop {
                if let Ok(events) = active_events.lock() {
                    for event in events.iter() {
                        // Vérifier si nous avons déjà traité cet événement spécifique
                        let event_key = format!("{}:{:?}", 
                            event.id, 
                            event.last_triggered.unwrap_or(Instant::now())
                        );
                        
                        if !processed_events.contains(&event_key) {
                            callback(event);
                            processed_events.insert(event_key);
                            
                            // Limiter la taille de l'ensemble des événements traités
                            if processed_events.len() > 1000 {
                                processed_events.clear();
                            }
                        }
                    }
                }
                
                // Pause pour réduire l'utilisation CPU
                std::thread::sleep(check_interval);
            }
        });
    }
}

/// État complet du système chronobiologique
#[derive(Debug, Clone)]
pub struct ChronobiologyState {
    pub current_phase: f64,
    pub circadian_phase: CircadianPhase,
    pub metabolic_rate: f64,
    pub phase_velocity: f64,
    pub cycle_length_seconds: u64,
    pub hormone_levels: HashMap<String, f64>,
    pub completed_cycles: u64,
    pub age_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bios_time_creation() {
        let bios = BiosTime::new();
        let state = bios.get_chronobiology_state();
        
        assert!(state.current_phase >= 0.0 && state.current_phase < 1.0);
        assert_eq!(state.completed_cycles, 0);
    }
    
    #[test]
    fn test_hormone_injection() {
        let bios = BiosTime::new();
        
        // Niveau initial
        let initial_cortisol = bios.get_chronobiology_state().hormone_levels
            .get("cortisol").cloned().unwrap_or(0.0);
        
        // Injecter de la cortisol
        let result = bios.inject_hormone("cortisol", 0.3);
        assert!(result);
        
        // Vérifier l'augmentation
        let new_cortisol = bios.get_chronobiology_state().hormone_levels
            .get("cortisol").cloned().unwrap_or(0.0);
        
        assert!(new_cortisol > initial_cortisol);
    }
    
    #[test]
    fn test_event_scheduling() {
        let bios = BiosTime::new();
        
        let event = BioEvent {
            id: "test_event".to_string(),
            event_type: BioEventType::Custom("test".to_string()),
            scheduled_phase: 0.5,
            duration: Duration::from_secs(60),
            priority: 0.5,
            target_callback: "test_callback".to_string(),
            parameters: HashMap::new(),
            recurrent: false,
            last_triggered: None,
        };
        
        let result = bios.schedule_event(event);
        assert!(result.is_ok());
        
        // Vérifier que l'annulation fonctionne
        let cancelled = bios.cancel_event("test_event");
        assert!(cancelled);
    }
}
