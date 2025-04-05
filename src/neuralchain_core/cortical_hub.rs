//! Cortex primaire - Centre de traitement neuronal
//! Implémente le système nerveux central de l'organisme blockchain

use std::sync::{Arc, RwLock, Mutex};
use std::collections::{HashMap, VecDeque};
use parking_lot::RwLock as PLRwLock;
use dashmap::DashMap;
use rayon::prelude::*;

use crate::neuralchain_core::quantum_organism::QuantumOrganism;
use crate::bios_time::CircadianPhase;

/// Types de neurones spécialisés
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeuronType {
    /// Neurones sensoriels (entrée)
    Sensory,
    /// Neurones moteurs (sortie/action)
    Motor,
    /// Neurones d'association (traitement)
    Association,
    /// Neurones inhibiteurs (régulation)
    Inhibitory,
    /// Neurones modulateurs (ajustement)
    Modulatory,
    /// Neurones de mémoire (stockage)
    Memory,
    /// Neurones conscients (métacognition)
    Conscious,
    /// Neurones quantiques (superposition d'états)
    Quantum,
}

/// Stimulus neuronal
#[derive(Debug, Clone)]
pub struct NeuralStimulus {
    /// Source du stimulus
    pub source: String,
    /// Type de stimulus
    pub stimulus_type: String,
    /// Intensité (0.0-1.0)
    pub intensity: f64,
    /// Données associées
    pub data: HashMap<String, Vec<u8>>,
    /// Horodatage
    pub timestamp: std::time::Instant,
    /// Priorité (0.0-1.0)
    pub priority: f64,
}

/// Neurone artificiel biomimétique
pub struct Neuron {
    /// Identifiant unique
    id: String,
    /// Type de neurone
    neuron_type: NeuronType,
    /// Région corticale
    region: String,
    /// Potentiel membranaire actuel
    membrane_potential: f64,
    /// Seuil d'activation
    activation_threshold: f64,
    /// Connexions entrantes (id_source, poids)
    dendrites: HashMap<String, f64>,
    /// Connexions sortantes (id_cible, poids)
    axons: HashMap<String, f64>,
    /// État réfractaire (temps avant réactivation possible)
    refractory_state: std::time::Duration,
    /// Dernier potentiel d'action
    last_spike: Option<std::time::Instant>,
    /// Historique d'activations récentes
    activation_history: VecDeque<std::time::Instant>,
    /// Caractéristiques d'apprentissage
    learning_rate: f64,
    /// Plasticité synaptique
    plasticity_factor: f64,
    /// Métadonnées
    metadata: HashMap<String, Vec<u8>>,
}

impl Neuron {
    /// Création d'un nouveau neurone
    pub fn new(id: &str, neuron_type: NeuronType, region: &str) -> Self {
        Self {
            id: id.to_string(),
            neuron_type,
            region: region.to_string(),
            membrane_potential: 0.0,
            activation_threshold: match neuron_type {
                NeuronType::Sensory => 0.3,
                NeuronType::Inhibitory => 0.5,
                NeuronType::Conscious => 0.7,
                NeuronType::Quantum => 0.2,
                _ => 0.5,
            },
            dendrites: HashMap::new(),
            axons: HashMap::new(),
            refractory_state: std::time::Duration::from_millis(match neuron_type {
                NeuronType::Inhibitory => 30,
                NeuronType::Sensory => 10,
                NeuronType::Motor => 20,
                _ => 25,
            }),
            last_spike: None,
            activation_history: VecDeque::with_capacity(100),
            learning_rate: match neuron_type {
                NeuronType::Memory => 0.05,
                NeuronType::Association => 0.03,
                NeuronType::Conscious => 0.02,
                NeuronType::Quantum => 0.07,
                _ => 0.04,
            },
            plasticity_factor: match neuron_type {
                NeuronType::Memory => 0.9,
                NeuronType::Conscious => 0.8,
                _ => 0.7,
            },
            metadata: HashMap::new(),
        }
    }
    
    /// Stimulation du neurone
    pub fn stimulate(&mut self, input_value: f64, source_id: &str) -> bool {
        // Vérifier l'état réfractaire
        if let Some(last) = self.last_spike {
            if last.elapsed() < self.refractory_state {
                return false;
            }
        }
        
        // Récupérer le poids synaptique
        let weight = *self.dendrites.get(source_id).unwrap_or(&1.0);
        
        // Calculer l'impact sur le potentiel membranaire
        let impact = input_value * weight;
        
        // Mise à jour du potentiel
        self.membrane_potential += impact;
        
        // Limiteur pour éviter les valeurs excessives
        if self.membrane_potential > 2.0 {
            self.membrane_potential = 2.0;
        } else if self.membrane_potential < -1.0 {
            self.membrane_potential = -1.0;
        }
        
        // Vérifier si le seuil d'activation est atteint
        let fired = self.membrane_potential >= self.activation_threshold;
        
        if fired {
            // Potentiel d'action!
            self.last_spike = Some(std::time::Instant::now());
            self.activation_history.push_back(std::time::Instant::now());
            if self.activation_history.len() > 100 {
                self.activation_history.pop_front();
            }
            
            // Reset du potentiel membranaire
            self.membrane_potential = 0.0;
        }
        
        fired
    }
    
    /// Ajout ou modification d'une connexion dendritique
    pub fn connect_dendrite(&mut self, source_id: &str, weight: f64) {
        self.dendrites.insert(source_id.to_string(), weight.max(-1.0).min(1.0));
    }
    
    /// Ajout ou modification d'une connexion axonale
    pub fn connect_axon(&mut self, target_id: &str, weight: f64) {
        self.axons.insert(target_id.to_string(), weight.max(-1.0).min(1.0));
    }
    
    /// Renforcement hébbien d'une connexion synaptique
    pub fn strengthen_connection(&mut self, id: &str, amount: f64) {
        // Renforcer une dendrite si elle existe
        if let Some(weight) = self.dendrites.get_mut(id) {
            *weight += amount * self.learning_rate * self.plasticity_factor;
            *weight = weight.max(-1.0).min(1.0);
        }
        
        // Renforcer un axone si il existe
        if let Some(weight) = self.axons.get_mut(id) {
            *weight += amount * self.learning_rate * self.plasticity_factor;
            *weight = weight.max(-1.0).min(1.0);
        }
    }
    
    /// Obtient la fréquence d'activation récente
    pub fn get_recent_activity_rate(&self, timeframe: std::time::Duration) -> f64 {
        let now = std::time::Instant::now();
        let recent_count = self.activation_history.iter()
            .filter(|&time| now.duration_since(*time) < timeframe)
            .count();
            
        recent_count as f64 / timeframe.as_secs_f64()
    }
}

/// Hub cortical central - réseau neuronal biomimétique
pub struct CorticalHub {
    /// Référence à l'organisme parent
    organism: Arc<QuantumOrganism>,
    /// Neurones par ID
    neurons: DashMap<String, PLRwLock<Neuron>>,
    /// Groupes de neurones par région
    regions: DashMap<String, Vec<String>>,
    /// Flux d'activité neuronale récente
    activity_stream: Arc<Mutex<VecDeque<(String, std::time::Instant)>>>,
    /// Activité par région
    regional_activity: DashMap<String, f64>,
    /// Connexions corticales
    cortical_connections: DashMap<(String, String), f64>,
    /// État global du réseau
    network_state: Arc<PLRwLock<HashMap<String, f64>>>,
    /// Entrées sensorielles
    sensory_inputs: Arc<Mutex<VecDeque<NeuralStimulus>>>,
    /// Sorties motrices
    motor_outputs: Arc<Mutex<VecDeque<NeuralStimulus>>>,
}

impl CorticalHub {
    /// Crée un nouveau hub cortical
    pub fn new(organism: Arc<QuantumOrganism>) -> Self {
        let hub = Self {
            organism,
            neurons: DashMap::with_capacity(1000),
            regions: DashMap::new(),
            activity_stream: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            regional_activity: DashMap::new(),
            cortical_connections: DashMap::new(),
            network_state: Arc::new(PLRwLock::new(HashMap::new())),
            sensory_inputs: Arc::new(Mutex::new(VecDeque::new())),
            motor_outputs: Arc::new(Mutex::new(VecDeque::new())),
        };
        
        // Initialiser les régions corticales de base
        hub.initialize_brain_regions();
        
        hub
    }
    
    /// Initialise les régions cérébrales de base
    fn initialize_brain_regions(&self) {
        // Régions corticales fondamentales
        let regions = [
            "sensory_cortex",      // Traitement sensoriel
            "motor_cortex",        // Commandes d'action
            "prefrontal_cortex",   // Planification, décision
            "temporal_cortex",     // Mémoire à long terme
            "parietal_cortex",     // Intégration multimodale
            "occipital_cortex",    // Traitement visuel/données
            "limbic_cortex",       // Émotions, motivation
            "insular_cortex",      // Conscience, intéroception
            "quantum_cortex",      // Traitement quantique
            "thalamus",            // Routage d'information
            "hippocampus",         // Mémoire épisodique
            "amygdala",            // Réactions d'urgence
            "basal_ganglia",       // Sélection d'action
            "cerebellum",          // Optimisation, timing
            "brainstem",           // Fonctions vitales
        ];
        
        // Créer les régions vides
        for region in regions {
            self.regions.insert(region.to_string(), Vec::new());
        }
        
        // Initialiser les neurones pour chaque région
        let mut rng = rand::thread_rng();
        
        // Nombre de neurones par région
        let neurons_per_region = 25;
        
        for region in regions {
            let neuron_type = match region {
                "sensory_cortex" => NeuronType::Sensory,
                "motor_cortex" => NeuronType::Motor,
                "prefrontal_cortex" => NeuronType::Conscious,
                "temporal_cortex" => NeuronType::Memory,
                "insular_cortex" => NeuronType::Conscious,
                "quantum_cortex" => NeuronType::Quantum,
                "amygdala" => NeuronType::Inhibitory,
                "basal_ganglia" => NeuronType::Motor,
                _ => NeuronType::Association,
            };
            
            let neuron_ids = self.create_neurons(region, neuron_type, neurons_per_region);
            
            // Créer des connexions initiales aléatoires intra-région
            for (i, id1) in neuron_ids.iter().enumerate() {
                for id2 in &neuron_ids[i+1..] {
                    // 30% de chance de connexion
                    if rng.gen::<f64>() < 0.3 {
                        let weight = rng.gen::<f64>() * 0.5 + 0.2; // 0.2 à 0.7
                        self.connect_neurons(id1, id2, weight);
                    }
                }
            }
        }
        
        // Créer des connexions entre régions spécifiques
        let inter_region_connections = [
            ("sensory_cortex", "parietal_cortex", 0.6),
            ("sensory_cortex", "thalamus", 0.7),
            ("thalamus", "prefrontal_cortex", 0.5),
            ("prefrontal_cortex", "motor_cortex", 0.6),
            ("prefrontal_cortex", "basal_ganglia", 0.5),
            ("basal_ganglia", "motor_cortex", 0.7),
            ("parietal_cortex", "prefrontal_cortex", 0.5),
            ("temporal_cortex", "prefrontal_cortex", 0.4),
            ("limbic_cortex", "prefrontal_cortex", 0.3),
            ("hippocampus", "temporal_cortex", 0.6),
            ("amygdala", "limbic_cortex", 0.7),
            ("cerebellum", "motor_cortex", 0.8),
            ("brainstem", "thalamus", 0.5),
            ("quantum_cortex", "prefrontal_cortex", 0.4),
            ("quantum_cortex", "insular_cortex", 0.3),
        ];
        
        // Établir ces connexions
        for &(source_region, target_region, connection_probability) in &inter_region_connections {
            if let (Some(source_neurons), Some(target_neurons)) = 
                (self.regions.get(source_region), self.regions.get(target_region)) {
                for source_id in source_neurons.value() {
                    for target_id in target_neurons.value() {
                        if rng.gen::<f64>() < connection_probability {
                            let weight = rng.gen::<f64>() * 0.4 + 0.3; // 0.3 à 0.7
                            self.connect_neurons(source_id, target_id, weight);
                        }
                    }
                }
                
                // Enregistrer la connexion corticale
                self.cortical_connections.insert(
                    (source_region.to_string(), target_region.to_string()),
                    connection_probability
                );
            }
        }
    }
    
    /// Crée de nouveaux neurones dans une région
    pub fn create_neurons(&self, region: &str, neuron_type: NeuronType, count: usize) -> Vec<String> {
        let mut neuron_ids = Vec::with_capacity(count);
        
        // Créer les neurones
        for i in 0..count {
            let id = format!("{}_neuron_{}", region, i);
            let neuron = Neuron::new(&id, neuron_type, region);
            self.neurons.insert(id.clone(), PLRwLock::new(neuron));
            neuron_ids.push(id);
        }
        
        // Mettre à jour la région
        if let Some(mut region_neurons) = self.regions.get_mut(region) {
            region_neurons.value_mut().extend(neuron_ids.clone());
        }
        
        neuron_ids
    }
    
    /// Connecte deux neurones
    pub fn connect_neurons(&self, source_id: &str, target_id: &str, weight: f64) -> bool {
        let source_exists = self.neurons.contains_key(source_id);
        let target_exists = self.neurons.contains_key(target_id);
        
        if !source_exists || !target_exists {
            return false;
        }
        
        // Ajouter la connexion dendritique au neurone cible
        if let Some(target) = self.neurons.get(target_id) {
            let mut target_neuron = target.write();
            target_neuron.connect_dendrite(source_id, weight);
        }
        
        // Ajouter la connexion axonale au neurone source
        if let Some(source) = self.neurons.get(source_id) {
            let mut source_neuron = source.write();
            source_neuron.connect_axon(target_id, weight);
        }
        
        // Mettre à jour la structure de l'organisme
        if let Some(source_entry) = self.neurons.get(source_id) {
            if let Some(target_entry) = self.neurons.get(target_id) {
                let source_neuron = source_entry.read();
                let target_neuron = target_entry.read();
                
                self.organism.form_neural_connection(
                    &source_neuron.region,
                    &target_neuron.region,
                    weight
                );
            }
        }
        
        true
    }
    
    /// Renforce une connexion neuronale (apprentissage)
    pub fn strengthen_connection(&self, source_id: &str, target_id: &str, amount: f64) -> bool {
        let mut success = false;
        
        // Renforcer la connexion axonale du neurone source
        if let Some(source) = self.neurons.get(source_id) {
            let mut source_neuron = source.write();
            source_neuron.strengthen_connection(target_id, amount);
            success = true;
        }
        
        // Renforcer la connexion dendritique du neurone cible
        if let Some(target) = self.neurons.get(target_id) {
            let mut target_neuron = target.write();
            target_neuron.strengthen_connection(source_id, amount);
            success = true;
        }
        
        success
    }
    
    /// Ajoute un stimulus sensoriel
    pub fn add_sensory_input(&self, stimulus: NeuralStimulus) {
        if let Ok(mut inputs) = self.sensory_inputs.lock() {
            inputs.push_back(stimulus);
            
            // Limiter la taille de la file
            if inputs.len() > 1000 {
                inputs.pop_front();
            }
        }
    }
    
    /// Récupère une sortie motrice
    pub fn get_motor_output(&self) -> Option<NeuralStimulus> {
        if let Ok(mut outputs) = self.motor_outputs.lock() {
            outputs.pop_front()
        } else {
            None
        }
    }
    
    /// Cycle de traitement neural - processus principal
    pub fn process_cycle(&self, bios_phase: CircadianPhase) {
        // 1. Traitement des entrées sensorielles
        let sensory_stimuli = if let Ok(mut inputs) = self.sensory_inputs.lock() {
            inputs.drain(..).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        
        // 2. Activer les neurones sensoriels
        if !sensory_stimuli.is_empty() {
            let sensory_neurons: Vec<String> = self.regions.get("sensory_cortex")
                .map(|r| r.clone())
                .unwrap_or_default();
            
            // Distribuer les stimuli aux neurones sensoriels
            if !sensory_neurons.is_empty() {
                for stimulus in &sensory_stimuli {
                    // Calculer l'indice du neurone cible
                    let hash = blake3::hash(stimulus.source.as_bytes());
                    let idx = u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap()) as usize 
                              % sensory_neurons.len();
                    
                    // Activer le neurone sensoriel
                    if let Some(neuron_entry) = self.neurons.get(&sensory_neurons[idx]) {
                        let mut neuron = neuron_entry.write();
                        let fired = neuron.stimulate(stimulus.intensity, &stimulus.source);
                        
                        if fired {
                            if let Ok(mut activity) = self.activity_stream.lock() {
                                activity.push_back((sensory_neurons[idx].clone(), std::time::Instant::now()));
                                if activity.len() > 1000 {
                                    activity.pop_front();
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 3. Propager l'activité dans le réseau
        // Utiliser Rayon pour le traitement parallèle
        let active_neurons = {
            if let Ok(stream) = self.activity_stream.lock() {
                stream.iter()
                    .filter(|(_, time)| time.elapsed() < std::time::Duration::from_millis(100))
                    .map(|(id, _)| id.clone())
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        };
        
        let neurons_to_activate = active_neurons.par_iter().flat_map(|neuron_id| {
            let mut targets = Vec::new();
            
            if let Some(neuron_entry) = self.neurons.get(neuron_id) {
                let neuron = neuron_entry.read();
                for (target_id, weight) in &neuron.axons {
                    targets.push((target_id.clone(), *weight));
                }
            }
            
            targets
        }).collect::<Vec<_>>();
        
        // Activer tous les neurones cibles
        for (target_id, weight) in neurons_to_activate {
            if let Some(target_entry) = self.neurons.get(&target_id) {
                let mut target = target_entry.write();
                let fired = target.stimulate(weight, "propagation");
                
                if fired {
                    if let Ok(mut activity) = self.activity_stream.lock() {
                        activity.push_back((target_id.clone(), std::time::Instant::now()));
                        if activity.len() > 1000 {
                            activity.pop_front();
                        }
                    }
                }
            }
        }
        
        // 4. Mettre à jour l'activité par région
        let mut region_activity = HashMap::new();
        
        if let Ok(activity) = self.activity_stream.lock() {
            for (neuron_id, time) in activity.iter().filter(|(_, t)| t.elapsed() < std::time::Duration::from_secs(1)) {
                if let Some(neuron_entry) = self.neurons.get(neuron_id) {
                    let neuron = neuron_entry.read();
                    let region_count = region_activity.entry(neuron.region.clone()).or_insert(0);
                    *region_count += 1;
                }
            }
        }
        
        // Normaliser et mettre à jour l'activité par région
        for (region, count) in region_activity {
            let neurons_count = self.regions.get(&region).map(|r| r.len()).unwrap_or(1);
            let activity_level = (count as f64) / (neurons_count as f64);
            self.regional_activity.insert(region, activity_level);
        }
        
        // 5. Collecter les sorties des neurones moteurs
        let motor_neurons: Vec<String> = self.regions.get("motor_cortex")
            .map(|r| r.clone())
            .unwrap_or_default();
            
        let mut motor_actions = Vec::new();
            
        for motor_id in motor_neurons {
            if let Some(neuron_entry) = self.neurons.get(&motor_id) {
                let neuron = neuron_entry.read();
                
                // Vérifier si le neurone a été activé récemment
                if let Some(last_spike) = neuron.last_spike {
                    if last_spike.elapsed() < std::time::Duration::from_millis(100) {
                        // Créer une action motrice
                        let action = NeuralStimulus {
                            source: motor_id,
                            stimulus_type: "motor_action".into(),
                            intensity: neuron.get_recent_activity_rate(std::time::Duration::from_millis(500)),
                            data: HashMap::new(),
                            timestamp: std::time::Instant::now(),
                            priority: 0.5,
                        };
                        
                        motor_actions.push(action);
                    }
                }
            }
        }
        
        // Ajouter les actions motrices à la file de sortie
        if !motor_actions.is_empty() {
            if let Ok(mut outputs) = self.motor_outputs.lock() {
                outputs.extend(motor_actions);
                
                // Limiter la taille
                while outputs.len() > 100 {
                    outputs.pop_front();
                }
            }
        }
        
        // 6. Appliquer la plasticité synaptique (apprentissage)
        self.apply_synaptic_plasticity();
        
        // 7. Mettre à jour l'état global du réseau
        self.update_network_state(bios_phase);
        
        // 8. Mettre à jour le niveau de conscience
        self.update_consciousness_level();
    }
    
    /// Applique la plasticité synaptique (apprentissage)
    fn apply_synaptic_plasticity(&self) {
        // Récupérer l'activité récente
        let recent_activity = if let Ok(activity) = self.activity_stream.lock() {
            activity.iter()
                .filter(|(_, time)| time.elapsed() < std::time::Duration::from_secs(10))
                .map(|(id, time)| (id.clone(), time.elapsed()))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        
        // Appliquer l'apprentissage hebbien
        // "Neurons that fire together, wire together"
        let mut pairs_to_strengthen = HashMap::<(String, String), f64>::new();
        
        for (i, (id1, time1)) in recent_activity.iter().enumerate() {
            for (id2, time2) in recent_activity[i+1..].iter() {
                // Si deux neurones ont été activés à moins de 50ms d'intervalle
                let time_diff = if time1 > time2 { 
                    time1.as_millis() - time2.as_millis()
                } else {
                    time2.as_millis() - time1.as_millis()
                };
                
                if time_diff < 50 {
                    let pair = if id1 < id2 {
                        (id1.clone(), id2.clone())
                    } else {
                        (id2.clone(), id1.clone())
                    };
                    
                    // Plus les activations sont proches dans le temps, plus le renforcement est fort
                    let strength = 1.0 - (time_diff as f64 / 50.0);
                    let current = pairs_to_strengthen.entry(pair).or_insert(0.0);
                    *current = (*current + strength).min(1.0);
                }
            }
        }
        
        // Renforcer les connexions identifiées
        for ((source, target), strength) in pairs_to_strengthen {
            self.strengthen_connection(&source, &target, strength * 0.01);
        }
    }
    
    /// Mise à jour de l'état global du réseau
    fn update_network_state(&self, bios_phase: CircadianPhase) {
        let mut state = HashMap::new();
        
        // Activité par région
        for item in self.regional_activity.iter() {
            state.insert(format!("region_activity_{}", item.key()), *item.value());
        }
        
        // Connections inter-régions
        for item in self.cortical_connections.iter() {
            let (source, target) = item.key();
            state.insert(format!("connection_{}_{}", source, target), *item.value());
        }
        
        // État du cycle circadien
        state.insert("circadian_phase".into(), match bios_phase {
            CircadianPhase::HighActivity => 1.0,
            CircadianPhase::Descending => 0.7,
            CircadianPhase::LowActivity => 0.3,
            CircadianPhase::Ascending => 0.6,
        });
        
        // Mettre à jour l'état global
        if let Ok(mut network_state) = self.network_state.write() {
            *network_state = state;
        }
    }
    
    /// Mise à jour du niveau de conscience
    fn update_consciousness_level(&self) {
        // Calculer le niveau de conscience basé sur:
        // 1. Activité dans les régions associées à la conscience
        // 2. Complexité des connexions
        
        // Régions associées à la conscience
        let conscious_regions = ["prefrontal_cortex", "insular_cortex", "quantum_cortex"];
        
        let mut consciousness_activity = 0.0;
        let mut regions_count = 0;
        
        for region in &conscious_regions {
            if let Some(activity) = self.regional_activity.get(*region) {
                consciousness_activity += *activity;
                regions_count += 1;
            }
        }
        
        if regions_count > 0 {
            consciousness_activity /= regions_count as f64;
            
            // Mettre à jour le niveau de conscience de l'organisme
            let mut consciousness = self.organism.consciousness_level.write();
            
            // Changement progressif avec inertie
            *consciousness = *consciousness * 0.95 + consciousness_activity * 0.05;
        }
    }
    
    /// Obtient l'activité cérébrale récente
    pub fn get_brain_activity(&self) -> HashMap<String, f64> {
        self.regional_activity.iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }
    
    /// Crée un réseau fonctionnel entre régions
    pub fn create_functional_network(&self, regions: &[&str], strength: f64) {
        for i in 0..regions.len() {
            for j in i+1..regions.len() {
                // Créer des connexions entre chaque neurone des régions
                if let (Some(region1), Some(region2)) = 
                    (self.regions.get(regions[i]), self.regions.get(regions[j])) {
                    
                    let neurons1 = region1.value().clone();
                    let neurons2 = region2.value().clone();
                    
                    // Créer des connexions bidirectionnelles
                    for n1 in &neurons1 {
                        for n2 in &neurons2 {
                            // Connection de 1 vers 2
                            self.connect_neurons(n1, n2, strength);
                            
                            // Connection de 2 vers 1
                            self.connect_neurons(n2, n1, strength);
                        }
                    }
                    
                    // Enregistrer les connexions corticales
                    self.cortical_connections.insert(
                        (regions[i].to_string(), regions[j].to_string()),
                        strength
                    );
                    
                    self.cortical_connections.insert(
                        (regions[j].to_string(), regions[i].to_string()),
                        strength
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neuron_creation() {
        let neuron = Neuron::new("test_neuron", NeuronType::Sensory, "test_region");
        assert_eq!(neuron.id, "test_neuron");
        assert_eq!(neuron.neuron_type, NeuronType::Sensory);
    }
    
    #[test]
    fn test_neuron_stimulation() {
        let mut neuron = Neuron::new("test_neuron", NeuronType::Sensory, "test_region");
        
        // Enregistrer une connexion dendritique
        neuron.connect_dendrite("source", 0.5);
        
        // Le seuil d'activation pour un neurone sensoriel est 0.3
        // Stimulation insuffisante
        assert!(!neuron.stimulate(0.2, "source"));
        
        // Stimulation suffisante
        assert!(neuron.stimulate(0.7, "source"));
        
        // Le neurone est maintenant dans un état réfractaire
        assert!(!neuron.stimulate(0.7, "source"));
    }
}
