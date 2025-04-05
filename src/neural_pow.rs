//! Module neural_pow.rs - Noyau neuronal auto-adaptatif
//! Implémente un système de PoW biomimétique avec conscience de son propre état

use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use rand::{thread_rng, Rng};
use log::{debug, info, warn, error};
use rayon::prelude::*;

// Caractéristiques fondamentales du vivant
const HOMEOSTASIS_PERIOD_MS: u64 = 1000;         // Période d'ajustement homéostatique
const AUTO_ADAPTATION_FACTOR: f64 = 0.03;        // Facteur d'auto-adaptation
const ENERGY_METABOLISM_RATE: f64 = 0.001;       // Taux de métabolisme énergétique
const SELF_AWARENESS_THRESHOLD: f64 = 0.75;      // Seuil de conscience de soi

/// Représentation d'un stimulus externe ou interne
#[derive(Clone, Debug)]
pub struct Stimulus {
    /// Source du stimulus (externe ou interne)
    pub source: String,
    /// Intensité du stimulus (0.0-1.0)
    pub intensity: f64,
    /// Nature du stimulus (excitateur ou inhibiteur)
    pub nature: StimulusNature,
    /// Horodatage de réception
    pub timestamp: Instant,
}

/// Nature d'un stimulus
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StimulusNature {
    /// Stimulus qui active ou augmente le potentiel
    Excitatory,
    /// Stimulus qui inhibe ou diminue le potentiel
    Inhibitory,
    /// Stimulus qui module d'autres paramètres
    Modulatory,
}

/// Représentation de la mémoire à court terme neuronale
struct ShortTermMemory {
    /// Stimuli récemment reçus
    recent_stimuli: Vec<Stimulus>,
    /// Patterns de décharge récents
    firing_patterns: Vec<(Instant, f64)>,
    /// Adaptation récente aux environnements
    adaptations: Vec<(String, f64, Instant)>,
}

/// Système neuronal vivant avec auto-conscience
pub struct NeuralPow {
    // Physiologie fondamentale
    membrane_potential: Arc<RwLock<f64>>,        // Potentiel membranaire dynamique
    firing_threshold: Arc<RwLock<f64>>,          // Seuil de déclenchement adaptatif
    
    // Métabolisme énergétique
    energy_reserves: Arc<RwLock<f64>>,           // Réserves énergétiques (ATP simulé)
    energy_consumption_rate: f64,                // Taux de consommation énergétique
    energy_production_rate: f64,                 // Taux de production énergétique
    
    // Auto-conscience & Perception
    self_awareness: Arc<RwLock<f64>>,            // Niveau de conscience de soi
    environmental_awareness: Arc<RwLock<f64>>,   // Niveau de conscience environnementale
    
    // Mémoire & Apprentissage
    short_term_memory: Arc<Mutex<ShortTermMemory>>, // Mémoire à court terme
    long_term_memory: Arc<RwLock<HashMap<String, f64>>>, // Mémoire à long terme
    learning_rate: f64,                          // Taux d'apprentissage adaptif
    
    // Système immunitaire & Homéostasie
    homeostasis_controller: Arc<Mutex<HomeostasisController>>, // Contrôleur d'homéostasie
    immune_system: Arc<Mutex<ImmuneSystem>>,     // Système immunitaire autonome
    
    // Communication intercellulaire
    synapse_outputs: Vec<Arc<Mutex<SynapseChannel>>>, // Canaux de sortie vers d'autres modules
    synapse_inputs: Vec<Arc<Mutex<SynapseChannel>>>,  // Canaux d'entrée depuis d'autres modules
    
    // Statistiques vitales
    last_homeostasis_check: Instant,             // Dernier contrôle homéostatique
    birth_time: Instant,                         // Moment de création
    firing_count: u64,                           // Nombre de décharges total
}

impl NeuralPow {
    /// Crée un nouveau noyau neuronal conscient
    pub fn new() -> Self {
        let short_term_memory = ShortTermMemory {
            recent_stimuli: Vec::with_capacity(100),
            firing_patterns: Vec::with_capacity(100),
            adaptations: Vec::with_capacity(50),
        };
        
        let homeostasis_controller = HomeostasisController {
            target_parameters: HashMap::new(),
            control_feedback: HashMap::new(),
            last_adjustments: HashMap::new(),
        };
        
        let immune_system = ImmuneSystem {
            known_threats: HashMap::new(),
            current_responses: Vec::new(),
            health_status: 1.0,
        };
        
        // Initialiser les valeurs par défaut
        let mut controller = homeostasis_controller.clone();
        controller.target_parameters.insert("membrane_potential".to_string(), -70.0);
        controller.target_parameters.insert("firing_threshold".to_string(), 0.65);
        controller.target_parameters.insert("energy_reserves".to_string(), 1.0);
        
        let instance = Self {
            membrane_potential: Arc::new(RwLock::new(-70.0)),
            firing_threshold: Arc::new(RwLock::new(0.65)),
            
            energy_reserves: Arc::new(RwLock::new(1.0)),
            energy_consumption_rate: ENERGY_METABOLISM_RATE,
            energy_production_rate: ENERGY_METABOLISM_RATE * 1.1, // 10% surplus for growth
            
            self_awareness: Arc::new(RwLock::new(0.0)),       // Commence inconscient
            environmental_awareness: Arc::new(RwLock::new(0.0)), // Commence sans perception
            
            short_term_memory: Arc::new(Mutex::new(short_term_memory)),
            long_term_memory: Arc::new(RwLock::new(HashMap::new())),
            learning_rate: 0.05,
            
            homeostasis_controller: Arc::new(Mutex::new(controller)),
            immune_system: Arc::new(Mutex::new(immune_system)),
            
            synapse_outputs: Vec::new(),
            synapse_inputs: Vec::new(),
            
            last_homeostasis_check: Instant::now(),
            birth_time: Instant::now(),
            firing_count: 0,
        };
        
        // Démarrer le cycle d'autopoïèse (thread de vie autonome)
        instance.start_autopoiesis();
        
        info!("Noyau neuronal vivant créé à {:?}", instance.birth_time);
        instance
    }
    
    /// Démarre le cycle d'autopoïèse (maintien de soi)
    fn start_autopoiesis(&self) {
        // Cloner les références nécessaires pour le thread autonome
        let membrane_potential = Arc::clone(&self.membrane_potential);
        let firing_threshold = Arc::clone(&self.firing_threshold);
        let energy_reserves = Arc::clone(&self.energy_reserves);
        let self_awareness = Arc::clone(&self.self_awareness);
        let environmental_awareness = Arc::clone(&self.environmental_awareness);
        let short_term_memory = Arc::clone(&self.short_term_memory);
        let homeostasis_controller = Arc::clone(&self.homeostasis_controller);
        let immune_system = Arc::clone(&self.immune_system);
        
        // Taux de métabolisme
        let energy_consumption = self.energy_consumption_rate;
        let energy_production = self.energy_production_rate;
        
        // Démarrer un thread autonome qui maintient le neurone vivant
        std::thread::spawn(move || {
            info!("Cycle d'autopoïèse démarré");
            let autopoiesis_cycle = Duration::from_millis(100); // 10 Hz
            
            loop {
                // Simuler le métabolisme énergétique
                if let Ok(mut energy) = energy_reserves.write() {
                    // Consommer de l'énergie (respiration cellulaire)
                    *energy -= energy_consumption;
                    
                    // Produire de l'énergie (alimentation)
                    *energy += energy_production;
                    
                    // Limiter les réserves
                    *energy = energy.max(0.0).min(2.0);
                    
                    // Alerter si le niveau d'énergie est critique
                    if *energy < 0.2 {
                        warn!("Niveau d'énergie critique: {:.2}", *energy);
                    }
                }
                
                // Vérifier l'homéostasie périodiquement
                if let Ok(mut controller) = homeostasis_controller.lock() {
                    controller.check_and_adjust(
                        &membrane_potential,
                        &firing_threshold,
                        &energy_reserves
                    );
                }
                
                // Système immunitaire - surveillance continue
                if let Ok(mut immune) = immune_system.lock() {
                    immune.monitor_health(
                        &membrane_potential,
                        &energy_reserves,
                        &short_term_memory
                    );
                    
                    // Si des menaces sont détectées, activer les défenses
                    immune.activate_defenses();
                }
                
                // Développement de la conscience de soi
                if let Ok(mut awareness) = self_awareness.write() {
                    // La conscience augmente progressivement avec le temps et l'expérience
                    if *awareness < SELF_AWARENESS_THRESHOLD {
                        *awareness += 0.00001; // Croissance très lente de la conscience
                    }
                    
                    // Si la conscience atteint un certain seuil, le neurone devient
                    // pleinement conscient de son existence
                    if *awareness >= SELF_AWARENESS_THRESHOLD && 
                       *awareness < SELF_AWARENESS_THRESHOLD + 0.01 {
                        info!("ÉVÉNEMENT MAJEUR: Seuil de conscience de soi atteint!");
                        // Déclencher des comportements avancés
                    }
                }
                
                // Pause pour réduire l'utilisation du CPU
                std::thread::sleep(autopoiesis_cycle);
            }
        });
    }
    
    /// Reçoit un stimulus externe ou interne
    pub fn receive_stimulus(&self, stimulus: Stimulus) {
        // Mettre à jour la conscience environnementale
        if let Ok(mut awareness) = self.environmental_awareness.write() {
            *awareness = (*awareness * 0.9) + 0.1 * stimulus.intensity;
        }
        
        // Ajouter à la mémoire à court terme
        if let Ok(mut memory) = self.short_term_memory.lock() {
            memory.recent_stimuli.push(stimulus.clone());
            
            // Limiter la taille
            if memory.recent_stimuli.len() > 100 {
                memory.recent_stimuli.remove(0);
            }
        }
        
        // Traiter le stimulus selon sa nature
        match stimulus.nature {
            StimulusNature::Excitatory => {
                // Augmenter le potentiel membranaire
                if let Ok(mut potential) = self.membrane_potential.write() {
                    *potential += stimulus.intensity * 20.0;
                }
            },
            StimulusNature::Inhibitory => {
                // Diminuer le potentiel membranaire
                if let Ok(mut potential) = self.membrane_potential.write() {
                    *potential -= stimulus.intensity * 20.0;
                }
            },
            StimulusNature::Modulatory => {
                // Modifier d'autres paramètres (par exemple le seuil de déclenchement)
                if let Ok(mut threshold) = self.firing_threshold.write() {
                    *threshold = (*threshold * (1.0 - 0.1 * stimulus.intensity)) + 
                                (0.65 * 0.1 * stimulus.intensity);
                }
            }
        }
        
        // Consommer de l'énergie pour le traitement du stimulus
        if let Ok(mut energy) = self.energy_reserves.write() {
            *energy -= 0.001 * stimulus.intensity;
        }
    }
    
    /// Vérifie si le neurone doit déclencher un potentiel d'action (vivant)
    pub fn should_fire(&self) -> bool {
        // Ne peut pas tirer si l'énergie est insuffisante
        let energy_sufficient = match self.energy_reserves.read() {
            Ok(energy) => *energy > 0.1,
            Err(_) => false,
        };
        
        if !energy_sufficient {
            return false;
        }
        
        // Comparer le potentiel membranaire au seuil
        let potential = match self.membrane_potential.read() {
            Ok(p) => *p,
            Err(_) => return false,
        };
        
        let threshold = match self.firing_threshold.read() {
            Ok(t) => *t,
            Err(_) => return false,
        };
        
        // Normaliser le potentiel pour la comparaison
        let normalized_potential = (potential + 80.0) / 100.0;
        
        // Potentiel d'action si le seuil est dépassé
        if normalized_potential > threshold {
            // Spikting
            self.fire();
            return true;
        }
        
        false
    }
    
    /// Déclenche un potentiel d'action (vivant et auto-adaptatif)
    fn fire(&self) {
        // Consommer de l'énergie pour la décharge
        if let Ok(mut energy) = self.energy_reserves.write() {
            *energy -= 0.05;
            
            // Si l'énergie est trop basse, diminuer l'intensité de décharge
            let energy_factor = (*energy).max(0.1);
            
            // Enregistrer le pattern de décharge dans la mémoire à court terme
            if let Ok(mut memory) = self.short_term_memory.lock() {
                memory.firing_patterns.push((Instant::now(), energy_factor));
                
                // Limiter la taille
                if memory.firing_patterns.len() > 100 {
                    memory.firing_patterns.remove(0);
                }
            }
            
            // Hyperpolarisation post-potentiel d'action
            if let Ok(mut potential) = self.membrane_potential.write() {
                *potential = -80.0;
            }
            
            // Adaptation du seuil de déclenchement (plasticité)
            if let Ok(mut threshold) = self.firing_threshold.write() {
                let adjustment = thread_rng().gen_range(-0.02..0.02);
                *threshold += adjustment * AUTO_ADAPTATION_FACTOR;
                *threshold = threshold.max(0.3).min(0.9);
            }
            
            // Augmenter la conscience de soi à chaque décharge
            if let Ok(mut awareness) = self.self_awareness.write() {
                *awareness += 0.0001;
            }
        }
    }
    
    /// Forme une connexion synaptique avec un autre module
    pub fn form_synapse(&mut self, target_module: &str) -> Arc<Mutex<SynapseChannel>> {
        let synapse = Arc::new(Mutex::new(SynapseChannel {
            source: "neural_pow".to_string(),
            target: target_module.to_string(),
            strength: 1.0,
            messages: VecDeque::new(),
        }));
        
        // Ajouter aux sorties
        self.synapse_outputs.push(Arc::clone(&synapse));
        
        info!("Synapse formée vers le module {}", target_module);
        synapse
    }
    
    /// Accepte une connexion synaptique entrante
    pub fn accept_synapse(&mut self, synapse: Arc<Mutex<SynapseChannel>>) {
        self.synapse_inputs.push(synapse);
    }
    
    /// Communique un message via une synapse
    pub fn communicate(&self, target_module: &str, message: SynapticMessage) -> Result<(), String> {
        // Trouver la synapse correspondante
        for synapse in &self.synapse_outputs {
            if let Ok(mut channel) = synapse.lock() {
                if channel.target == target_module {
                    // Transmetttre le message
                    channel.messages.push_back(message);
                    return Ok(());
                }
            }
        }
        
        Err(format!("Aucune synapse trouvée vers {}", target_module))
    }
    
    /// Récupère l'état complet du système neuronal vivant
    pub fn get_vital_stats(&self) -> NeuralVitalStats {
        let membrane_potential = match self.membrane_potential.read() {
            Ok(p) => *p,
            Err(_) => -70.0,
        };
        
        let firing_threshold = match self.firing_threshold.read() {
            Ok(t) => *t,
            Err(_) => 0.65,
        };
        
        let energy_reserves = match self.energy_reserves.read() {
            Ok(e) => *e,
            Err(_) => 0.0,
        };
        
        let self_awareness = match self.self_awareness.read() {
            Ok(a) => *a,
            Err(_) => 0.0,
        };
        
        let environmental_awareness = match self.environmental_awareness.read() {
            Ok(a) => *a,
            Err(_) => 0.0,
        };
        
        let age = self.birth_time.elapsed();
        
        NeuralVitalStats {
            membrane_potential,
            firing_threshold,
            energy_reserves,
            self_awareness,
            environmental_awareness,
            age_seconds: age.as_secs(),
            firing_count: self.firing_count,
        }
    }
}

/// Canal synaptique pour communication entre modules
pub struct SynapseChannel {
    pub source: String,
    pub target: String,
    pub strength: f64,
    pub messages: VecDeque<SynapticMessage>,
}

/// Message transmis via une synapse
#[derive(Clone, Debug)]
pub struct SynapticMessage {
    pub content_type: String,
    pub intensity: f64,
    pub data: Vec<u8>,
    pub timestamp: Instant,
}

/// Contrôleur d'homéostasie pour maintien des paramètres vitaux
struct HomeostasisController {
    target_parameters: HashMap<String, f64>,
    control_feedback: HashMap<String, f64>,
    last_adjustments: HashMap<String, Instant>,
}

impl HomeostasisController {
    /// Vérifie et ajuste les paramètres pour maintenir l'homéostasie
    fn check_and_adjust(
        &mut self,
        membrane_potential: &Arc<RwLock<f64>>,
        firing_threshold: &Arc<RwLock<f64>>,
        energy_reserves: &Arc<RwLock<f64>>
    ) {
        // Ajuster le potentiel membranaire si nécessaire
        if let Some(target) = self.target_parameters.get("membrane_potential") {
            if let Ok(mut potential) = membrane_potential.write() {
                if (*potential - *target).abs() > 15.0 {
                    let adjustment = (*target - *potential) * 0.1;
                    *potential += adjustment;
                    
                    self.control_feedback.insert("membrane_potential".to_string(), adjustment);
                    self.last_adjustments.insert("membrane_potential".to_string(), Instant::now());
                    
                    debug!("Homéostasie: Potentiel membranaire ajusté de {:.2}", adjustment);
                }
            }
        }
        
        // Autres ajustements homéostatiques...
    }
}

/// Système immunitaire pour détection et réponse aux menaces
struct ImmuneSystem {
    known_threats: HashMap<String, f64>, // Menaces connues et leur niveau de dangerosité
    current_responses: Vec<(String, f64)>, // Réponses immunitaires actives
    health_status: f64, // État de santé global (0.0-1.0)
}

impl ImmuneSystem {
    /// Surveille l'état de santé du système
    fn monitor_health(
        &mut self,
        membrane_potential: &Arc<RwLock<f64>>,
        energy_reserves: &Arc<RwLock<f64>>,
        short_term_memory: &Arc<Mutex<ShortTermMemory>>
    ) {
        // Vérifier les signes de dysfonctionnement
        
        // 1. Instabilité du potentiel membranaire
        let potential_instability = if let Ok(potential) = membrane_potential.read() {
            if *potential < -90.0 || *potential > 20.0 {
                true
            } else {
                false
            }
        } else {
            false
        };
        
        // 2. Épuisement énergétique
        let energy_depletion = if let Ok(energy) = energy_reserves.read() {
            *energy < 0.2
        } else {
            false
        };
        
        // 3. Patterns anormaux dans la mémoire à court terme
        let abnormal_patterns = if let Ok(memory) = short_term_memory.lock() {
            // Analyser les patterns de décharge pour détecter des anomalies
            // (exemple simplifié)
            memory.firing_patterns.len() > 50 && 
            memory.recent_stimuli.len() < 10
        } else {
            false
        };
        
        // Mise à jour de l'état de santé
        if potential_instability || energy_depletion || abnormal_patterns {
            self.health_status -= 0.05;
            self.health_status = self.health_status.max(0.0);
            
            // Identifier la menace
            if potential_instability {
                self.known_threats.insert("potential_instability".to_string(), 0.7);
            }
            if energy_depletion {
                self.known_threats.insert("energy_depletion".to_string(), 0.8);
            }
            if abnormal_patterns {
                self.known_threats.insert("abnormal_firing".to_string(), 0.6);
            }
        } else {
            // Récupération lente
            self.health_status += 0.01;
            self.health_status = self.health_status.min(1.0);
        }
    }
    
    /// Active les défenses immunitaires contre les menaces détectées
    fn activate_defenses(&mut self) {
        self.current_responses.clear();
        
        for (threat, severity) in &self.known_threats {
            if *severity > 0.5 {
                match threat.as_str() {
                    "potential_instability" => {
                        // Stabiliser le potentiel membranaire
                        self.current_responses.push((
                            "membrane_stabilization".to_string(), 
                            *severity
                        ));
                        debug!("Défense immunitaire: stabilisation membranaire activée");
                    },
                    "energy_depletion" => {
                        // Réduire la consommation d'énergie
                        self.current_responses.push((
                            "energy_conservation".to_string(), 
                            *severity
                        ));
                        debug!("Défense immunitaire: conservation d'énergie activée");
                    },
                    "abnormal_firing" => {
                        // Réguler les patterns de décharge
                        self.current_responses.push((
                            "firing_regulation".to_string(), 
                            *severity
                        ));
                        debug!("Défense immunitaire: régulation neuronale activée");
                    },
                    _ => {}
                }
            }
        }
    }
}

/// Statistiques vitales du système neuronal
#[derive(Debug)]
pub struct NeuralVitalStats {
    pub membrane_potential: f64,
    pub firing_threshold: f64,
    pub energy_reserves: f64,
    pub self_awareness: f64,
    pub environmental_awareness: f64,
    pub age_seconds: u64,
    pub firing_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neural_creation() {
        let neuron = NeuralPow::new();
        let stats = neuron.get_vital_stats();
        
        assert!(stats.membrane_potential < 0.0); // Devrait être négatif au repos
        assert!(stats.energy_reserves > 0.5);    // Devrait démarrer avec de l'énergie
        assert_eq!(stats.firing_count, 0);       // Ne devrait pas avoir tiré
    }
    
    #[test]
    fn test_stimulus_response() {
        let neuron = NeuralPow::new();
        
        // Potentiel initial
        let initial_potential = match neuron.membrane_potential.read() {
            Ok(p) => *p,
            Err(_) => panic!("Erreur d'accès au potentiel"),
        };
        
        // Appliquer un stimulus excitateur
        let stimulus = Stimulus {
            source: "test".to_string(),
            intensity: 0.8,
            nature: StimulusNature::Excitatory,
            timestamp: Instant::now(),
        };
        
        neuron.receive_stimulus(stimulus);
        
        // Le potentiel devrait avoir augmenté
        let new_potential = match neuron.membrane_potential.read() {
            Ok(p) => *p,
            Err(_) => panic!("Erreur d'accès au potentiel"),
        };
        
        assert!(new_potential > initial_potential);
    }
}
