//! Module metasynapse.rs - Système de connexion neuroplastique
//! Inspiré de la synaptogenèse et plasticité synaptique du cerveau,
//! permettant à la blockchain de former, renforcer ou éliminer des connexions
//! entre modules de façon autonome et adaptative.

use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::time::{Duration, Instant};
use log::{debug, info, warn, error};
use rand::{thread_rng, Rng};
use rayon::prelude::*;

// Constantes d'optimisation synaptique
const MAX_SYNAPSES_PER_MODULE: usize = 100;     // Nombre maximal de synapses par module
const SYNAPSE_DECAY_RATE: f64 = 0.005;          // Taux de dégénérescence synaptique
const HEBBIAN_LEARNING_RATE: f64 = 0.02;        // Taux d'apprentissage hebbien
const PRUNING_THRESHOLD: f64 = 0.1;             // Seuil d'élagage synaptique
const SYNAPTOGENESIS_PROBABILITY: f64 = 0.01;   // Probabilité de formation spontanée
const INITIAL_SYNAPSE_STRENGTH: f64 = 0.3;      // Force initiale d'une nouvelle synapse

/// Type de message transmis via une synapse
#[derive(Debug, Clone)]
pub enum SynapticMessageType {
    /// Données brutes (octets)
    Data(Vec<u8>),
    /// Signal d'activation (scalaire)
    Activation(f64),
    /// Signal inhibiteur (scalaire négatif)
    Inhibition(f64),
    /// Message de contrôle (commande + paramètres)
    Control(String, HashMap<String, Vec<u8>>),
    /// Signal de modulation (hormonal)
    Modulation(String, f64),
    /// Message structurel (modification de topologie)
    Structural(String),
}

/// Message transmis via une connexion synaptique
#[derive(Debug, Clone)]
pub struct SynapticMessage {
    /// Type de message
    pub message_type: SynapticMessageType,
    /// Module source
    pub source_module: String,
    /// Module destinataire
    pub target_module: String,
    /// Identifiant unique du message
    pub message_id: u64,
    /// Priorité du message (0.0-1.0)
    pub priority: f64,
    /// Horodatage de création
    pub creation_time: Instant,
    /// Temps de vie (TTL)
    pub ttl: Duration,
}

/// Synapse connectant deux modules
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Identifiant unique
    pub id: String,
    /// Module source
    pub source: String,
    /// Module cible
    pub target: String,
    /// Force de la connexion (0.0-1.0)
    pub strength: f64,
    /// Type de synapse
    pub synapse_type: SynapseType,
    /// Taux d'activation récent
    pub recent_activation: f64,
    /// Taux d'efficacité
    pub efficacy: f64,
    /// Temps de création
    pub creation_time: Instant,
    /// Dernière activation
    pub last_activation: Instant,
    /// Paramètres spécifiques
    pub parameters: HashMap<String, f64>,
    /// Métadonnées
    pub metadata: HashMap<String, Vec<u8>>,
}

/// Type de connexion synaptique
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SynapseType {
    /// Synapse excitatrice
    Excitatory,
    /// Synapse inhibitrice
    Inhibitory,
    /// Synapse modulatrice
    Modulatory,
    /// Synapse structurelle
    Structural,
    /// Synapse bidirectionnelle
    Bidirectional,
    /// Synapse conditionnelle
    Conditional,
    /// Synapse temporaire
    Temporary,
}

/// Décrit un groupe de synapses avec des propriétés communes
#[derive(Debug, Clone)]
pub struct SynapticCluster {
    /// Identifiant unique
    pub id: String,
    /// Synapses membres
    pub synapses: Vec<String>,
    /// Fonction collective
    pub function: String,
    /// Force moyenne
    pub average_strength: f64,
    /// Temps de création
    pub creation_time: Instant,
}

/// Structure principale du module de métasynapse
pub struct MetaSynapse {
    // État du système
    active: Arc<RwLock<bool>>,
    plasticity_enabled: Arc<RwLock<bool>>,
    
    // Registre des modules, synapses et clusters
    modules: Arc<RwLock<HashSet<String>>>,
    synapses: Arc<RwLock<HashMap<String, Synapse>>>,
    synapse_clusters: Arc<RwLock<HashMap<String, SynapticCluster>>>,
    
    // Canaux de communication
    message_channels: Arc<RwLock<HashMap<String, Arc<Mutex<Vec<SynapticMessage>>>>>>,
    
    // Paramètres de neuroplasticité
    learning_rate: Arc<RwLock<f64>>,
    pruning_threshold: Arc<RwLock<f64>>,
    max_synapses_per_module: Arc<RwLock<usize>>,
    
    // Métriques et statistiques
    activation_history: Arc<Mutex<HashMap<String, VecDeque<(Instant, f64)>>>>,
    message_counters: Arc<RwLock<HashMap<String, u64>>>,
    
    // Gestion des événements neuronaux
    event_callbacks: Arc<RwLock<HashMap<String, Box<dyn Fn(&str, &str, f64) + Send + Sync>>>>,
    
    // Cache de routage pour optimisation
    routing_cache: Arc<RwLock<HashMap<(String, String), Vec<String>>>>,
    
    // Topologie et visualisation
    topology_graph: Arc<RwLock<HashMap<String, Vec<(String, f64)>>>>,
    
    // Paramètres système
    system_birth_time: Instant,
}

impl MetaSynapse {
    /// Crée une nouvelle instance du système de métasynapse
    pub fn new() -> Self {
        let instance = Self {
            active: Arc::new(RwLock::new(true)),
            plasticity_enabled: Arc::new(RwLock::new(true)),
            
            modules: Arc::new(RwLock::new(HashSet::new())),
            synapses: Arc::new(RwLock::new(HashMap::new())),
            synapse_clusters: Arc::new(RwLock::new(HashMap::new())),
            
            message_channels: Arc::new(RwLock::new(HashMap::new())),
            
            learning_rate: Arc::new(RwLock::new(HEBBIAN_LEARNING_RATE)),
            pruning_threshold: Arc::new(RwLock::new(PRUNING_THRESHOLD)),
            max_synapses_per_module: Arc::new(RwLock::new(MAX_SYNAPSES_PER_MODULE)),
            
            activation_history: Arc::new(Mutex::new(HashMap::new())),
            message_counters: Arc::new(RwLock::new(HashMap::new())),
            
            event_callbacks: Arc::new(RwLock::new(HashMap::new())),
            
            routing_cache: Arc::new(RwLock::new(HashMap::new())),
            
            topology_graph: Arc::new(RwLock::new(HashMap::new())),
            
            system_birth_time: Instant::now(),
        };
        
        // Démarrer le système de métasynapse
        instance.start_metasynapse_system();
        
        info!("Système de métasynapse créé à {:?}", instance.system_birth_time);
        instance
    }
    
    /// Démarre le système de métasynapse autonome
    fn start_metasynapse_system(&self) {
        // Cloner les références nécessaires
        let active = Arc::clone(&self.active);
        let plasticity_enabled = Arc::clone(&self.plasticity_enabled);
        let synapses = Arc::clone(&self.synapses);
        let modules = Arc::clone(&self.modules);
        let message_channels = Arc::clone(&self.message_channels);
        let activation_history = Arc::clone(&self.activation_history);
        let learning_rate = Arc::clone(&self.learning_rate);
        let pruning_threshold = Arc::clone(&self.pruning_threshold);
        let topology_graph = Arc::clone(&self.topology_graph);
        let routing_cache = Arc::clone(&self.routing_cache);
        
// Suite du code précédent...

        // Démarrer le thread autonome du système de métasynapse
        std::thread::spawn(move || {
            info!("Système de métasynapse démarré - neuroplasticité et synaptogenèse autonome");
            let cycle_interval = Duration::from_millis(25); // 40 Hz - optimisé pour réactivité
            let mut last_plasticity_cycle = Instant::now();
            let plasticity_interval = Duration::from_secs(10);
            let mut rng = thread_rng();
            
            // Utiliser un pool de threads préalloué pour maximiser les performances sur Windows
            // en exploitant efficacement les cœurs du processeur
            #[cfg(target_os = "windows")]
            let thread_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_cpus::get().min(16))
                .stack_size(2 * 1024 * 1024) // 2MB stack - optimisé pour Windows
                .thread_name(|i| format!("metasynapse_worker_{}", i))
                .build()
                .unwrap();
            
            loop {
                // Vérifier si le système est actif
                let is_active = match active.read() {
                    Ok(a) => *a,
                    Err(_) => true, // Par défaut actif en cas d'erreur
                };
                
                if !is_active {
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
                
                // 1. Traitement des messages dans tous les canaux (haute priorité)
                if let Ok(channels) = message_channels.read() {
                    for (module_id, channel) in channels.iter() {
                        // Traiter jusqu'à 10 messages par module par cycle
                        let mut processed_count = 0;
                        
                        if let Ok(mut messages) = channel.lock() {
                            // Trier les messages par priorité (optimisation pour traitement rapide)
                            messages.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
                            
                            let messages_to_process: Vec<_> = messages.drain(0..messages.len().min(10)).collect();
                            
                            // Libérer le verrou avant traitement pour réduire le temps de verrouillage
                            drop(messages);
                            
                            for message in messages_to_process {
                                // Vérifier que le message n'a pas expiré
                                if message.creation_time.elapsed() > message.ttl {
                                    continue;
                                }
                                
                                // Traiter le message
                                Self::process_synaptic_message(
                                    &message,
                                    &synapses,
                                    &activation_history,
                                    &message_channels,
                                );
                                
                                processed_count += 1;
                            }
                        }
                        
                        if processed_count > 0 {
                            debug!("Traitement de {} messages pour le module {}", processed_count, module_id);
                        }
                    }
                }
                
                // 2. Appliquer la plasticité synaptique (basse priorité, exécutée moins fréquemment)
                if last_plasticity_cycle.elapsed() > plasticity_interval {
                    last_plasticity_cycle = Instant::now();
                    
                    // Vérifier si la plasticité est activée
                    let plasticity_active = match plasticity_enabled.read() {
                        Ok(p) => *p,
                        Err(_) => false,
                    };
                    
                    if plasticity_active {
                        // Cloner les synapses pour traitement parallèle
                        let mut synapses_to_process = Vec::new();
                        if let Ok(syn_map) = synapses.read() {
                            synapses_to_process = syn_map.values().cloned().collect();
                        }
                        
                        if !synapses_to_process.is_empty() {
                            #[cfg(target_os = "windows")]
                            thread_pool.install(|| {
                                // Traiter les synapses en parallèle pour des performances optimales
                                let updated_synapses: Vec<_> = synapses_to_process.par_iter()
                                    .map(|synapse| {
                                        Self::apply_hebbian_learning(
                                            synapse,
                                            &activation_history,
                                            &learning_rate,
                                        )
                                    })
                                    .filter(|result| result.is_some())
                                    .map(|result| result.unwrap())
                                    .collect();
                                
                                // Mettre à jour les synapses modifiées
                                if !updated_synapses.is_empty() {
                                    if let Ok(mut syn_map) = synapses.write() {
                                        for synapse in updated_synapses {
                                            syn_map.insert(synapse.id.clone(), synapse);
                                        }
                                    }
                                }
                            });
                            
                            #[cfg(not(target_os = "windows"))]
                            {
                                // Version non-Windows (moins optimisée mais fonctionnelle)
                                let updated_synapses: Vec<_> = synapses_to_process.iter()
                                    .filter_map(|synapse| {
                                        Self::apply_hebbian_learning(
                                            synapse,
                                            &activation_history,
                                            &learning_rate,
                                        )
                                    })
                                    .collect();
                                
                                if !updated_synapses.is_empty() {
                                    if let Ok(mut syn_map) = synapses.write() {
                                        for synapse in updated_synapses {
                                            syn_map.insert(synapse.id.clone(), synapse);
                                        }
                                    }
                                }
                            }
                        }
                        
                        // 3. Élagage synaptique (suppression des synapses faibles)
                        Self::prune_weak_synapses(&synapses, &pruning_threshold, &modules);
                        
                        // 4. Synaptogenèse (création de nouvelles synapses)
                        if rng.gen::<f64>() < SYNAPTOGENESIS_PROBABILITY {
                            Self::attempt_synaptogenesis(
                                &modules,
                                &synapses,
                                &activation_history,
                                &message_channels,
                                &max_synapses_per_module,
                            );
                        }
                        
                        // 5. Mise à jour du graphe de topologie (pour visualisation et analyse)
                        Self::update_topology_graph(&synapses, &topology_graph);
                        
                        // 6. Invalidation du cache de routage pour refléter les changements de topologie
                        if let Ok(mut cache) = routing_cache.write() {
                            cache.clear();
                        }
                    }
                }
                
                // Pause optimisée pour réduire l'utilisation CPU
                // Ajustement spécial pour Windows pour éviter le "thread thrashing"
                #[cfg(target_os = "windows")]
                {
                    let start = std::time::Instant::now();
                    while start.elapsed() < cycle_interval {
                        // Spin wait court puis yield pour les systèmes Windows
                        if start.elapsed() > cycle_interval / 2 {
                            std::thread::yield_now();
                        }
                    }
                }
                
                #[cfg(not(target_os = "windows"))]
                {
                    std::thread::sleep(cycle_interval);
                }
            }
        });
    }
    
    /// Traite un message synaptique
    fn process_synaptic_message(
        message: &SynapticMessage,
        synapses: &Arc<RwLock<HashMap<String, Synapse>>>,
        activation_history: &Arc<Mutex<HashMap<String, VecDeque<(Instant, f64)>>>>,
        message_channels: &Arc<RwLock<HashMap<String, Arc<Mutex<Vec<SynapticMessage>>>>>>,
    ) {
        // Déterminer la force du stimulus basée sur le type de message
        let activation_strength = match &message.message_type {
            SynapticMessageType::Activation(strength) => *strength,
            SynapticMessageType::Inhibition(strength) => -*strength,
            SynapticMessageType::Data(data) => 0.3 + (data.len() as f64 * 0.01).min(0.7),
            SynapticMessageType::Control(_, _) => 0.5,
            SynapticMessageType::Modulation(_, intensity) => *intensity,
            SynapticMessageType::Structural(_) => 0.8,
        };
        
        // Enregistrer l'activation pour l'apprentissage hebbien
        if let Ok(mut history) = activation_history.lock() {
            // Source
            let source_history = history.entry(message.source_module.clone())
                .or_insert_with(|| VecDeque::with_capacity(100));
            source_history.push_back((Instant::now(), activation_strength));
            if source_history.len() > 100 {
                source_history.pop_front();
            }
            
            // Target
            let target_history = history.entry(message.target_module.clone())
                .or_insert_with(|| VecDeque::with_capacity(100));
            target_history.push_back((Instant::now(), activation_strength));
            if target_history.len() > 100 {
                target_history.pop_front();
            }
        }
        
        // Trouver les synapses correspondantes et les renforcer
        if let Ok(mut synapses_map) = synapses.write() {
            let synapse_key = format!("{}->{}", message.source_module, message.target_module);
            
            if let Some(synapse) = synapses_map.get_mut(&synapse_key) {
                // Renforcer immédiatement la synapse (feedback rapide)
                let age_factor = 1.0 - (-synapse.creation_time.elapsed().as_secs_f64() / 3600.0).exp(); // Facteur d'âge exponentiel
                let strength_delta = activation_strength * 0.01 * age_factor;
                
                synapse.strength += strength_delta;
                synapse.strength = synapse.strength.max(0.0).min(1.0);
                synapse.last_activation = Instant::now();
                synapse.recent_activation = (synapse.recent_activation * 0.9) + (activation_strength * 0.1);
                
                // Pour les synapses bidirectionnelles, créer un message inverse
                if synapse.synapse_type == SynapseType::Bidirectional {
                    // Créer un message de retour avec force réduite
                    let feedback_message = SynapticMessage {
                        message_type: match &message.message_type {
                            SynapticMessageType::Activation(s) => SynapticMessageType::Activation(s * 0.6),
                            SynapticMessageType::Inhibition(s) => SynapticMessageType::Inhibition(s * 0.6),
                            other => other.clone(),
                        },
                        source_module: message.target_module.clone(),
                        target_module: message.source_module.clone(),
                        message_id: message.message_id + 1000000, // ID différent
                        priority: message.priority * 0.8,
                        creation_time: Instant::now(),
                        ttl: message.ttl / 2, // TTL réduit pour éviter les loops
                    };
                    
                    // Placer dans la file du module destinataire
                    if let Ok(channels) = message_channels.read() {
                        if let Some(channel) = channels.get(&message.source_module) {
                            if let Ok(mut messages) = channel.lock() {
                                messages.push(feedback_message);
                            }
                        }
                    }
                }
            }
        }
        
        // Propagation du message aux cibles liées
        // (implémentation de la diffusion du signal aux modules connectés)
    }
    
    /// Applique l'apprentissage hebbien pour renforcer/affaiblir les synapses
    fn apply_hebbian_learning(
        synapse: &Synapse,
        activation_history: &Arc<Mutex<HashMap<String, VecDeque<(Instant, f64)>>>>,
        learning_rate: &Arc<RwLock<f64>>,
    ) -> Option<Synapse> {
        // Obtenir l'historique d'activation des modules source et cible
        let (source_activations, target_activations) = if let Ok(history) = activation_history.lock() {
            let source_hist = history.get(&synapse.source)
                .map(|h| h.iter().filter(|(t, _)| t.elapsed() < Duration::from_secs(60)).cloned().collect::<Vec<_>>())
                .unwrap_or_default();
                
            let target_hist = history.get(&synapse.target)
                .map(|h| h.iter().filter(|(t, _)| t.elapsed() < Duration::from_secs(60)).cloned().collect::<Vec<_>>())
                .unwrap_or_default();
                
            (source_hist, target_hist)
        } else {
            (Vec::new(), Vec::new())
        };
        
        if source_activations.is_empty() || target_activations.is_empty() {
            // Appliquer la dégénérescence synaptique pour les synapses inactives
            // Allocation sur la pile pour éviter les allocations mémoire inutiles
            if synapse.last_activation.elapsed() > Duration::from_secs(300) { // 5 minutes
                let mut new_synapse = synapse.clone();
                new_synapse.strength -= SYNAPSE_DECAY_RATE;
                new_synapse.strength = new_synapse.strength.max(0.0);
                return Some(new_synapse);
            }
            return None;
        }
        
        // Calculer la corrélation temporelle entre les activations
        let mut correlation: f64 = 0.0;
        let mut count = 0;
        
        for (source_time, source_strength) in &source_activations {
            for (target_time, target_strength) in &target_activations {
                // Si la source s'active juste avant la cible (0-100ms), renforcer la connexion
                let time_diff = target_time.duration_since(*source_time);
                if time_diff > Duration::from_millis(0) && time_diff < Duration::from_millis(100) {
                    // Plus les activations sont fortes et proches dans le temps, plus la corrélation est forte
                    let temporal_factor = (-time_diff.as_millis() as f64 / 100.0).exp();
                    correlation += source_strength * target_strength * temporal_factor;
                    count += 1;
                }
            }
        }
        
        if count == 0 {
            return None;
        }
        
        correlation /= count as f64;
        
        // Appliquer l'apprentissage hebbien
        if correlation != 0.0 {
            let learn_rate = learning_rate.read().unwrap_or(0.02);
            
            // Application de la règle d'apprentissage hebbien modifiée avec compression non-linéaire
            let mut new_synapse = synapse.clone();
            
            // Fonction sigmoïdale pour éviter les extrêmes et réduire le surajustement
            let strength_delta = learn_rate * correlation * (1.0 - new_synapse.strength.powf(2.0));
            new_synapse.strength += strength_delta;
            new_synapse.strength = new_synapse.strength.max(0.0).min(1.0);
            
            // Mise à jour de l'efficacité
            new_synapse.efficacy = (new_synapse.efficacy * 0.95) + (correlation.abs() * 0.05);
            
            return Some(new_synapse);
        }
        
        None
    }
    
    /// Élague les synapses faibles pour optimiser le réseau
    fn prune_weak_synapses(
        synapses: &Arc<RwLock<HashMap<String, Synapse>>>,
        pruning_threshold: &Arc<RwLock<f64>>,
        modules: &Arc<RwLock<HashSet<String>>>,
    ) {
        let threshold = match pruning_threshold.read() {
            Ok(t) => *t,
            Err(_) => PRUNING_THRESHOLD,
        };
        
        // Compter les synapses par module pour éviter l'élagage excessif
        let mut synapse_counts: HashMap<String, usize> = HashMap::new();
        
        // Collection pour stocker les synapses à éliminer
        let mut to_prune = Vec::new();
        
        // Premier passage: compter les synapses par module
        if let Ok(synapse_map) = synapses.read() {
            for synapse in synapse_map.values() {
                *synapse_counts.entry(synapse.source.clone()).or_insert(0) += 1;
                *synapse_counts.entry(synapse.target.clone()).or_insert(0) += 1;
            }
            
            // Deuxième passage: identifier les synapses à élaguer
            for (id, synapse) in synapse_map.iter() {
                // Conditions d'élagage:
                // 1. Force inférieure au seuil
                // 2. Pas une synapse temporaire (celles-ci ont leur propre gestion de durée de vie)
                // 3. Le module a plus qu'un minimum de synapses
                let source_has_enough = synapse_counts.get(&synapse.source).map_or(false, |&count| count > 3);
                let target_has_enough = synapse_counts.get(&synapse.target).map_or(false, |&count| count > 3);
                
                if synapse.strength < threshold && synapse.synapse_type != SynapseType::Temporary &&
                   source_has_enough && target_has_enough {
                    to_prune.push(id.clone());
                }
            }
        }
        
        // Troisième passage: supprimer les synapses identifiées
        if !to_prune.is_empty() {
            if let Ok(mut synapse_map) = synapses.write() {
                for id in to_prune {
                    synapse_map.remove(&id);
                }
            }
        }
    }
    
    /// Tente de créer de nouvelles synapses (synaptogenèse)
    fn attempt_synaptogenesis(
        modules: &Arc<RwLock<HashSet<String>>>,
        synapses: &Arc<RwLock<HashMap<String, Synapse>>>,
        activation_history: &Arc<Mutex<HashMap<String, VecDeque<(Instant, f64)>>>>,
        message_channels: &Arc<RwLock<HashMap<String, Arc<Mutex<Vec<SynapticMessage>>>>>>,
        max_synapses_per_module: &Arc<RwLock<usize>>,
    ) {
        let mut rng = thread_rng();
        
        // Obtenir la liste des modules
        let module_list = if let Ok(mods) = modules.read() {
            mods.iter().cloned().collect::<Vec<_>>()
        } else {
            return;
        };
        
        if module_list.len() < 2 {
            return; // Besoin d'au moins 2 modules
        }
        
        // Choisir aléatoirement un module source et cible différents
        let source_idx = rng.gen_range(0..module_list.len());
        let mut target_idx = rng.gen_range(0..module_list.len());
        
        // S'assurer que les modules sont différents
        while target_idx == source_idx && module_list.len() > 1 {
            target_idx = rng.gen_range(0..module_list.len());
        }
        
        let source = &module_list[source_idx];
        let target = &module_list[target_idx];
        
        // Vérifier les limites de synapses par module
        let max_per_module = if let Ok(max) = max_synapses_per_module.read() {
            *max
        } else {
            MAX_SYNAPSES_PER_MODULE
        };
        
        let mut source_count = 0;
        let mut target_count = 0;
        let mut connection_exists = false;
        
        // Vérifier les conditions actuelles
        if let Ok(synapse_map) = synapses.read() {
            for synapse in synapse_map.values() {
                if &synapse.source == source {
                    source_count += 1;
                }
                if &synapse.target == target {
                    target_count += 1;
                }
                if &synapse.source == source && &synapse.target == target {
                    connection_exists = true;
                    break;
                }
            }
        }
        
        // Si la connexion existe déjà ou si l'un des modules a atteint sa limite, annuler
        if connection_exists || source_count >= max_per_module || target_count >= max_per_module {
            return;
        }
        
        // Vérifier l'activité récente des modules pour déterminer s'ils sont candidats à la connexion
        let should_connect = {
            let mut connect = false;
            
            if let Ok(history) = activation_history.lock() {
                // Une synapse se forme si les deux modules ont été actifs récemment
                let source_activity = history.get(source).map_or(0.0, |h| {
                    h.iter()
                        .filter(|(t, _)| t.elapsed() < Duration::from_secs(300))
                        .map(|(_, s)| *s)
                        .sum::<f64>()
                });
                
                let target_activity = history.get(target).map_or(0.0, |h| {
                    h.iter()
                        .filter(|(t, _)| t.elapsed() < Duration::from_secs(300))
                        .map(|(_, s)| *s)
                        .sum::<f64>()
                });
                
                connect = source_activity > 0.5 && target_activity > 0.5;
            }
            
            connect
        };
        
        if should_connect {
            // Créer une nouvelle synapse
            let synapse_type = if rng.gen::<f64>() < 0.2 {
                // 20% de chance de créer une synapse inhibitrice
                SynapseType::Inhibitory
            } else if rng.gen::<f64>() < 0.15 {
                // 15% de chance de créer une synapse bidirectionnelle
                SynapseType::Bidirectional
            } else {
                // 65% de chance de créer une synapse excitatrice standard
                SynapseType::Excitatory
            };
            
            let synapse_id = format!("{}_{}_{}",
                                    source,
                                    target,
                                    chrono::Utc::now().timestamp_nanos());
                                    
            let synapse_key = format!("{}->{}", source, target);
            
            let new_synapse = Synapse {
                id: synapse_id,
                source: source.clone(),
                target: target.clone(),
                strength: INITIAL_SYNAPSE_STRENGTH,
                synapse_type,
                recent_activation: 0.0,
                efficacy: 0.5,
                creation_time: Instant::now(),
                last_activation: Instant::now(),
                parameters: HashMap::new(),
                metadata: HashMap::new(),
            };
            
            // Ajouter la synapse au registre
            if let Ok(mut synapse_map) = synapses.write() {
                synapse_map.insert(synapse_key, new_synapse.clone());
                
                debug!("Nouvelle synapse créée: {} -> {} (type: {:?}, force: {:.2})", 
                      source, target, synapse_type, INITIAL_SYNAPSE_STRENGTH);
            }
            
            // Assurer que les deux modules ont des canaux de messages
            if let Ok(mut channels) = message_channels.write() {
                // Créer le canal pour la source s'il n'existe pas
                if !channels.contains_key(source) {
                    channels.insert(source.clone(), Arc::new(Mutex::new(Vec::new())));
                }
                
                // Créer le canal pour la cible s'il n'existe pas
                if !channels.contains_key(target) {
                    channels.insert(target.clone(), Arc::new(Mutex::new(Vec::new())));
                }
            }
        }
    }
    
    /// Met à jour le graphe de topologie pour visualisation
    fn update_topology_graph(
        synapses: &Arc<RwLock<HashMap<String, Synapse>>>,
        topology_graph: &Arc<RwLock<HashMap<String, Vec<(String, f64)>>>>,
    ) {
        // Structure intermédiaire pour construire le graphe
        let mut graph: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        
        // Construire le graphe à partir des synapses
        if let Ok(synapse_map) = synapses.read() {
            for synapse in synapse_map.values() {
                // Ajouter une arête dirigée de la source vers la cible
                graph.entry(synapse.source.clone())
                    .or_insert_with(Vec::new)
                    .push((synapse.target.clone(), synapse.strength));
                
                // Pour les synapses bidirectionnelles, ajouter également l'arête inverse
                if synapse.synapse_type == SynapseType::Bidirectional {
                    graph.entry(synapse.target.clone())
                        .or_insert_with(Vec::new)
                        .push((synapse.source.clone(), synapse.strength));
                }
            }
        }
        
        // Mettre à jour le graphe global
        if let Ok(mut topo_graph) = topology_graph.write() {
            *topo_graph = graph;
        }
    }
    
    /// Enregistre un nouveau module dans le système de métasynapse
    pub fn register_module(&self, module_id: &str) -> Result<(), String> {
        if module_id.is_empty() {
            return Err("L'identifiant du module ne peut pas être vide".to_string());
        }
        
        // Ajouter le module au registre
        if let Ok(mut mods) = self.modules.write() {
            mods.insert(module_id.to_string());
            
            // Créer un canal de messages pour ce module
            if let Ok(mut channels) = self.message_channels.write() {
                channels.insert(module_id.to_string(), Arc::new(Mutex::new(Vec::new())));
            }
            
            info!("Module enregistré dans le système de métasynapse: {}", module_id);
            Ok(())
        } else {
            Err("Impossible d'accéder au registre des modules".to_string())
        }
    }
    
    /// Crée manuellement une synapse entre deux modules
    pub fn create_synapse(
        &self,
        source: &str,
        target: &str,
        synapse_type: SynapseType,
        initial_strength: f64,
    ) -> Result<String, String> {
        // Vérifier que les deux modules sont enregistrés
        let modules_exist = if let Ok(mods) = self.modules.read() {
            mods.contains(source) && mods.contains(target)
        } else {
            false
        };
        
        if !modules_exist {
            return Err("Un ou plusieurs modules n'existent pas".to_string());
        }
        
        // Vérifier si une synapse existe déjà entre ces modules
        let synapse_key = format!("{}->{}", source, target);
        
        let exists = if let Ok(synapse_map) = self.synapses.read() {
            synapse_map.contains_key(&synapse_key)
        } else {
            false
        };
        
        if exists {
            return Err("Une synapse existe déjà entre ces modules".to_string());
        }
        
        // Créer la synapse
        let synapse_id = format!("{}_{}_{}",
                                source,
                                target,
                                chrono::Utc::now().timestamp_nanos());
        
        let strength = initial_strength.max(0.0).min(1.0);
        
        let new_synapse = Synapse {
            id: synapse_id.clone(),
            source: source.to_string(),
            target: target.to_string(),
            strength,
            synapse_type,
            recent_activation: 0.0,
            efficacy: 0.5,
            creation_time: Instant::now(),
            last_activation: Instant::now(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        // Ajouter la synapse au registre
        if let Ok(mut synapse_map) = self.synapses.write() {
            synapse_map.insert(synapse_key, new_synapse);
            
            // Invalider le cache de routage
            if let Ok(mut cache) = self.routing_cache.write() {
                cache.clear();
            }
            
            info!("Synapse créée manuellement: {} -> {} (type: {:?}, force: {:.2})", 
                  source, target, synapse_type, strength);
            
            Ok(synapse_id)
        } else {
            Err("Impossible d'accéder au registre des synapses".to_string())
        }
    }
    
    /// Envoie un message à travers le réseau synaptique
    pub fn send_message(
        &self,
        source: &str,
        target: &str,
        message_type: SynapticMessageType,
        priority: f64,
        ttl: Duration,
    ) -> Result<u64, String> {
        // Générer un ID de message unique basé sur le timestamp nano
        let message_id = chrono::Utc::now().timestamp_nanos() as u64;
        
        // Créer le message
        let message = SynapticMessage {
            message_type,
            source_module: source.to_string(),
            target_module: target.to_string(),
            message_id,
            priority: priority.max(0.0).min(1.0),
            creation_time: Instant::now(),
            ttl,
        };
        
        // Trouver le canal du module cible
        if let Ok(channels) = self.message_channels.read() {
            if let Some(channel) = channels.get(target) {
                if let Ok(mut messages) = channel.lock() {
                    messages.push(message);
                    
                    // Mettre à jour les compteurs
                    if let Ok(mut counters) = self.message_counters.write() {
                        let counter = counters.entry(target.to_string()).or_insert(0);
                        *counter += 1;
                    }
                    
                    return Ok(message_id);
                }
            }
            
            Err(format!("Module cible non trouvé: {}", target))
        } else {
            Err("Impossible d'accéder aux canaux de messages".to_string())
        }
    }
    
    /// Trouve un chemin optimal entre deux modules
    pub fn find_path(&self, source: &str, target: &str) -> Option<Vec<String>> {
        // Vérifier d'abord le cache de routage
        if let Ok(cache) = self.routing_cache.read() {
            if let Some(path) = cache.get(&(source.to_string(), target.to_string())) {
                return Some(path.clone());
            }
        }
        
        // Construire le graphe de topologie
        let graph = if let Ok(topo) = self.topology_graph.read() {
            topo.clone()
        } else {
            return None;
        };
        
        // Algorithme A* pour trouver le chemin optimal
        let path = Self::a_star_search(&graph, source, target);
        
        // Mettre en cache le résultat si trouvé
        if let Some(ref found_path) = path {
            if let Ok(mut cache) = self.routing_cache.write() {
                cache.insert(
                    (source.to_string(), target.to_string()),
                    found_path.clone()
                );
            }
        }
        
        path
    }
    
    /// Implémentation optimisée de l'algorithme A* pour la recherche de chemins
    fn a_star_search(
        graph: &HashMap<String, Vec<(String, f64)>>,
        start: &str,
        goal: &str
    ) -> Option<Vec<String>> {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;
        
        // Structure pour représenter un nœud dans la file de priorité
        #[derive(Clone, Eq)]
        struct Node {
            id: String,
            cost: u32,
            priority: u32,
        }
        
        impl PartialEq for Node {
            fn eq(&self, other: &Self) -> bool {
                self.priority == other.priority
            }
        }
        
        impl Ord for Node {
            fn cmp(&self, other: &Self) -> Ordering {
                other.priority.cmp(&self.priority) // Priorité inversée pour un tas min
            }
        }
        
        impl PartialOrd for Node {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        
        // Ensemble des nœuds visités
        let mut visited = HashSet::new();
        
        // File de priorité des nœuds à explorer
        let mut priority_queue = BinaryHeap::new();
        
        // Carte des coûts du meilleur chemin connu pour chaque nœud
        let mut costs = HashMap::new();
        
        // Carte des parents pour reconstruire le chemin
        let mut came_from = HashMap::new();
        
        // Initialiser avec le nœud de départ
        priority_queue.push(Node {
            id: start.to_string(),
            cost: 0,
            priority: 0,
        });
        
        costs.insert(start.to_string(), 0);
        
        while let Some(current) = priority_queue.pop() {
            // Si nous atteignons l'objectif, reconstruire et renvoyer le chemin
            if current.id == goal {
                let mut path = Vec::new();
                let mut current_id = goal.to_string();
                
                while current_id != start {
                    path.push(current_id.clone());
                    current_id = came_from.get(&current_id).unwrap().clone();
                }
                
                path.push(start.to_string());
                path.reverse();
                return Some(path);
            }
            
            // Sauter si déjà visité
            if visited.contains(&current.id) {
                continue;
            }
            
            visited.insert(current.id.clone());
            
            // Explorer les voisins
            if let Some(neighbors) = graph.get(&current.id) {
                for (neighbor, synapse_strength) in neighbors {
                    if visited.contains(neighbor) {
                        continue;
                    }
                    
                    // Calculer le coût pour atteindre le voisin
                    // Coût inversement proportionnel à la force synaptique
                    let edge_cost = ((1.0 - *synapse_strength) * 100.0).ceil() as u32;
                    let new_cost = current.cost + edge_cost;
                    
                    // Si c'est un meilleur chemin, le mettre à jour
                    let is_better = match costs.get(neighbor) {
                        None => true,
                        Some(&cost) => new_cost < cost,
                    };
                    
                    if is_better {
                        costs.insert(neighbor.clone(), new_cost);
                        
                        // Estimer l'heuristique (distance topologique)
                        // Comme la distance réelle est difficile à estimer, utiliser 0
                        let priority = new_cost;
                        
                        priority_queue.push(Node {
                            id: neighbor.clone(),
                            cost: new_cost,
                            priority,
                        });
                        
                        came_from.insert(neighbor.clone(), current.id.clone());
                    }
                }
            }
        }
        
        None // Aucun chemin trouvé
    }
    
    /// Crée un cluster synaptique pour des synapses liées fonctionnellement
    pub fn create_cluster(&self, cluster_id: &str, synapses: &[String], function: &str) -> Result<(), String> {
        if synapses.is_empty() {
            return Err("Le cluster doit contenir au moins une synapse".to_string());
        }
        
        // Vérifier que toutes les synapses existent
        let all_exist = if let Ok(synapse_map) = self.synapses.read() {
            synapses.iter().all(|id| {
                synapse_map.values().any(|s| &s.id == id)
            })
        } else {
            false
        };
        
        if !all_exist {
            return Err("Une ou plusieurs synapses n'existent pas".to_string());
        }
        
        // Calculer la force moyenne
        let avg_strength = if let Ok(synapse_map) = self.synapses.read() {
            let mut sum = 0.0;
            let mut count = 0;
            
            for id in synapses {
                for s in synapse_map.values() {
                    if &s.id == id {
                        sum += s.strength;
                        count += 1;
                        break;
                    }
                }
            }
            
            if count > 0 { sum / count as f64 } else { 0.0 }
        } else {
            0.0
        };
        
        // Créer le cluster
        let cluster = SynapticCluster {
            id: cluster_id.to_string(),
            synapses: synapses.to_vec(),
            function: function.to_string(),
            average_strength: avg_strength,
            creation_time: Instant::now(),
        };
        
        // Enregistrer le cluster
        if let Ok(mut clusters) = self.synapse_clusters.write() {
            clusters.insert(cluster_id.to_string(), cluster);
            
            info!("Cluster synaptique créé: {} avec {} synapses (fonction: {})", 
                  cluster_id, synapses.len(), function);
            
            Ok(())
        } else {
            Err("Impossible d'accéder au registre des clusters".to_string())
        }
    }
    
    /// Active ou désactive la plasticité synaptique
    pub fn set_plasticity_enabled(&self, enabled: bool) -> Result<(), String> {
        if let Ok(mut plasticity) = self.plasticity_enabled.write() {
            *plasticity = enabled;
            
            info!("Plasticité synaptique {}", if enabled { "activée" } else { "désactivée" });
            
            Ok(())
        } else {
            Err("Impossible d'accéder à l'état de plasticité".to_string())
        }
    }
    
    /// Règle le taux d'apprentissage hebbien
    pub fn set_learning_rate(&self, rate: f64) -> Result<(), String> {
        if rate < 0.0 || rate > 1.0 {
            return Err("Le taux d'apprentissage doit être entre 0.0 et 1.0".to_string());
        }
        
        if let Ok(mut lr) = self.learning_rate.write() {
            *lr = rate;
            
            info!("Taux d'apprentissage hebbien réglé à {:.3}", rate);
            
            Ok(())
        } else {
            Err("Impossible d'accéder au taux d'apprentissage".to_string())
        }
    }
    
    /// Obtient les statistiques du système de métasynapse
    pub fn get_stats(&self) -> MetaSynapseStats {
        let modules_count = match self.modules.read() {
            Ok(mods) => mods.len(),
            Err(_) => 0,
        };
        
        let synapses_count = match self.synapses.read() {
            Ok(syns) => syns.len(),
            Err(_) => 0,
        };
        
        let clusters_count = match self.synapse_clusters.read() {
            Ok(clusters) => clusters.len(),
            Err(_) => 0,
        };
        
        let plasticity_enabled = match self.plasticity_enabled.read() {
            Ok(p) => *p,
            Err(_) => false,
        };
        
        let learning_rate = match self.learning_rate.read() {
            Ok(lr) => *lr,
            Err(_) => 0.0,
        };
        
        let mut high_strength_synapses = 0;
        let mut avg_strength = 0.0;
        
        if let Ok(syns) = self.synapses.read() {
            let mut sum = 0.0;
            
            for synapse in syns.values() {
                sum += synapse.strength;
                
                if synapse.strength > 0.7 {
                    high_strength_synapses += 1;
                }
            }
            
            if !syns.is_empty() {
                avg_strength = sum / syns.len() as f64;
            }
        }
        
        let message_count = match self.message_counters.read() {
            Ok(counters) => counters.values().sum(),
            Err(_) => 0,
        };
        
        MetaSynapseStats {
            modules_count,
            synapses_count,
            clusters_count,
            plasticity_enabled,
            learning_rate,
            avg_synapse_strength: avg_strength,
            high_strength_synapses,
            message_count,
            age_seconds: self.system_birth_time.elapsed().as_secs(),
        }
    }
    
    /// Visualise le réseau synaptique sous forme de graphe DOT
    pub fn visualize_network(&self) -> String {
        let mut dot = String::from("digraph SynapticNetwork {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=circle, style=filled];\n");
        
        // Ajouter les nœuds (modules)
        if let Ok(mods) = self.modules.read() {
            for module in mods.iter() {
                dot.push_str(&format!("  \"{}\" [fillcolor=lightblue];\n", module));
            }
        }
        
        // Ajouter les arêtes (synapses)
        if let Ok(syns) = self.synapses.read() {
            for synapse in syns.values() {
                // Couleur basée sur le type de synapse
                let color = match synapse.synapse_type {
                    SynapseType::Excitatory => "green",
                    SynapseType::Inhibitory => "red",
                    SynapseType::Modulatory => "purple",
                    SynapseType::Structural => "blue",
                    SynapseType::Bidirectional => "orange",
                    SynapseType::Conditional => "brown",
                    SynapseType::Temporary => "gray",
                };
                
                // Épaisseur basée sur la force
                let width = 1.0 + (synapse.strength * 4.0);
                
                // Style en pointillés pour les synapses faibles
                let style = if synapse.strength < 0.3 {
                    "dashed"
                } else {
                    "solid"
                };
                
                dot.push_str(&format!(
                    "  \"{}\" -> \"{}\" [color=\"{}\", penwidth={:.1}, style=\"{}\", label=\"{:.2}\"];\n",
                    synapse.source,
                    synapse.target,
                    color,
                    width,
                    style,
                    synapse.strength
                ));
            }
        }
        
        dot.push_str("}\n");
        dot
    }
}

/// Statistiques du système de métasynapse
#[derive(Debug, Clone)]
pub struct MetaSynapseStats {
    pub modules_count: usize,
    pub synapses_count: usize,
    pub clusters_count: usize,
    pub plasticity_enabled: bool,
    pub learning_rate: f64,
    pub avg_synapse_strength: f64,
    pub high_strength_synapses: usize,
    pub message_count: u64,
    pub age_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metasynapse_creation() {
        let meta = MetaSynapse::new();
        let stats = meta.get_stats();
        
        assert_eq!(stats.modules_count, 0);
        assert_eq!(stats.synapses_count, 0);
        assert!(stats.plasticity_enabled); // Par défaut activé
    }
    
    #[test]
    fn test_module_registration() {
        let meta = MetaSynapse::new();
        
        let result = meta.register_module("test_module");
        assert!(result.is_ok());
        
        let stats = meta.get_stats();
        assert_eq!(stats.modules_count, 1);
    }
    
    #[test]
    fn test_synapse_creation() {
        let meta = MetaSynapse::new();
        
        // Enregistrer deux modules
        meta.register_module("module_a").unwrap();
        meta.register_module("module_b").unwrap();
        
        // Créer une synapse entre eux
        let result = meta.create_synapse(
            "module_a",
            "module_b",
            SynapseType::Excitatory,
            0.5
        );
        
        assert!(result.is_ok());
        
        let stats = meta.get_stats();
        assert_eq!(stats.synapses_count, 1);
    }
}
