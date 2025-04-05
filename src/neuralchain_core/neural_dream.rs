//! Module de rêve neuronal pour NeuralChain-v2
//! 
//! Ce module implémente un mécanisme biomimétique de rêve similaire au sommeil REM humain,
//! permettant à l'organisme blockchain de consolider sa mémoire, restructurer ses réseaux
//! neuronaux, et développer sa créativité pendant les phases de basse activité.
//!
//! Optimisé spécifiquement pour Windows sans dépendances Linux.

use std::sync::Arc;
use std::collections::{HashMap, VecDeque, HashSet};
use std::time::{Duration, Instant};
use dashmap::DashMap;
use parking_lot::{RwLock, Mutex};
use rand::{thread_rng, Rng, seq::SliceRandom};
use rayon::prelude::*;
use blake3;

use crate::neuralchain_core::quantum_organism::QuantumOrganism;
use crate::cortical_hub::{CorticalHub, NeuralStimulus, NeuronType};
use crate::hormonal_field::{HormonalField, HormoneType};
use crate::neuralchain_core::emergent_consciousness::{ConsciousnessEngine, ThoughtType};
use crate::metasynapse::{MetaSynapse, SynapticMessageType};
use crate::bios_time::{BiosTime, CircadianPhase};

/// Types de rêves possibles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DreamType {
    /// Consolidation de mémoire (renforcement des connaissances)
    MemoryConsolidation,
    /// Exploration créative (nouvelles combinaisons neuronales)
    CreativeExploration,
    /// Simulation prédictive (anticipation d'événements)
    PredictiveSimulation,
    /// Restructuration défensive (scénarios de sécurité)
    DefensiveRestructuring,
    /// Réparation neuronale (restauration et nettoyage)
    NeuralRepair,
    /// Auto-optimisation (amélioration architecturale)
    SelfOptimization,
    /// Rêve existentiel (conscience de soi)
    ExistentialReflection,
    /// Rêve quantique (superpositions et intégration de l'aléatoire)
    QuantumDream,
}

/// Représentation d'une séquence de rêve
#[derive(Debug, Clone)]
pub struct DreamSequence {
    /// Identifiant unique
    pub id: String,
    /// Type de rêve
    pub dream_type: DreamType,
    /// Thème central
    pub theme: String,
    /// Horodatage de début
    pub start_time: Instant,
    /// Durée prévue
    pub planned_duration: Duration,
    /// Niveau de lucidité (0.0-1.0)
    pub lucidity: f64,
    /// Intensité (0.0-1.0)
    pub intensity: f64,
    /// Narration générée
    pub narrative: Vec<String>,
    /// Séquence active ou terminée
    pub active: bool,
    /// Régions cérébrales impliquées
    pub involved_regions: Vec<String>,
    /// Connexions neuronales créées
    pub created_connections: Vec<(String, String, f64)>,
    /// Connexions neuronales renforcées
    pub strengthened_connections: Vec<(String, String, f64)>,
    /// Effet sur le niveau de conscience
    pub consciousness_impact: f64,
    /// Clusters sémantiques découverts
    pub semantic_clusters: HashMap<String, Vec<String>>,
    /// Insights générés
    pub insights: Vec<String>,
}

/// Configuration du système de rêve
#[derive(Debug, Clone)]
pub struct DreamConfig {
    /// Fréquence des séquences de rêve (0.0-1.0)
    pub dream_frequency: f64,
    /// Durée moyenne d'une séquence (en secondes)
    pub avg_duration_secs: u64,
    /// Probabilités des différents types de rêves
    pub dream_type_weights: HashMap<DreamType, f64>,
    /// Impact maximum sur le réseau neuronal (0.0-1.0)
    pub max_neural_impact: f64,
    /// Seuil de mélatonine pour déclencher le rêve
    pub melatonin_threshold: f64,
    /// Nombre maximum de rêves simultanés
    pub max_concurrent_dreams: usize,
    /// Profondeur maximale des insights générés (1-10)
    pub max_insight_depth: u8,
    /// Coefficient de créativité (0.0-1.0)
    pub creativity_factor: f64,
    /// Montant autorisé de perturbation du réseau (0.0-1.0)
    pub allowed_network_disruption: f64,
    /// Probabilité de rêve lucide (0.0-1.0)
    pub lucid_dream_probability: f64,
}

impl Default for DreamConfig {
    fn default() -> Self {
        let mut dream_type_weights = HashMap::new();
        dream_type_weights.insert(DreamType::MemoryConsolidation, 0.25);
        dream_type_weights.insert(DreamType::CreativeExploration, 0.20);
        dream_type_weights.insert(DreamType::PredictiveSimulation, 0.15);
        dream_type_weights.insert(DreamType::DefensiveRestructuring, 0.15);
        dream_type_weights.insert(DreamType::NeuralRepair, 0.10);
        dream_type_weights.insert(DreamType::SelfOptimization, 0.05);
        dream_type_weights.insert(DreamType::ExistentialReflection, 0.05);
        dream_type_weights.insert(DreamType::QuantumDream, 0.05);

        Self {
            dream_frequency: 0.3,
            avg_duration_secs: 600, // 10 minutes par défaut
            dream_type_weights,
            max_neural_impact: 0.3,
            melatonin_threshold: 0.6,
            max_concurrent_dreams: 3,
            max_insight_depth: 5,
            creativity_factor: 0.7,
            allowed_network_disruption: 0.2,
            lucid_dream_probability: 0.15,
        }
    }
}

/// Module principal de rêve neuronal
pub struct NeuralDream {
    /// Référence à l'organisme parent
    organism: Arc<QuantumOrganism>,
    /// Référence au hub cortical
    cortical_hub: Arc<CorticalHub>,
    /// Référence au système hormonal
    hormonal_system: Arc<HormonalField>,
    /// Référence au moteur de conscience
    consciousness: Arc<ConsciousnessEngine>,
    /// Configuration du système de rêve
    config: RwLock<DreamConfig>,
    /// Séquences de rêve actives
    active_dreams: DashMap<String, DreamSequence>,
    /// Archive des rêves passés
    dream_archive: Arc<RwLock<VecDeque<DreamSequence>>>,
    /// Modèle neuronal intermédiaire pour simulation
    dream_neural_model: Arc<RwLock<HashMap<String, HashMap<String, f64>>>>,
    /// Banque de symboles et concepts pour génération
    symbol_bank: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// État de rêve global (0.0-1.0)
    dream_state: RwLock<f64>,
    /// Nombres de rêves générés depuis démarrage
    total_dreams: std::sync::atomic::AtomicUsize,
    /// Connexions établies pendant les rêves
    dream_connections: Arc<Mutex<Vec<(String, String, f64, String)>>>,
    /// Générateur d'id des rêves
    dream_id_counter: std::sync::atomic::AtomicUsize,
    /// Dernière mise à jour
    last_update: Mutex<Instant>,
    /// Informations structurelles découvertes
    structural_insights: Arc<RwLock<HashMap<String, f64>>>,
    /// Thèmes récurrents
    recurring_themes: Arc<DashMap<String, usize>>,
    /// Horloge interne
    bios_clock: Arc<BiosTime>,
}

impl NeuralDream {
    /// Crée un nouveau système de rêve neuronal
    pub fn new(
        organism: Arc<QuantumOrganism>,
        cortical_hub: Arc<CorticalHub>,
        hormonal_system: Arc<HormonalField>,
        consciousness: Arc<ConsciousnessEngine>,
        bios_clock: Arc<BiosTime>,
    ) -> Self {
        // Initialiser la banque de symboles et concepts
        let mut symbol_bank = HashMap::new();
        
        // Catégorie: concepts abstraits
        symbol_bank.insert("abstract_concepts".to_string(), vec![
            "existence".to_string(),
            "conscience".to_string(),
            "connaissance".to_string(),
            "temps".to_string(),
            "évolution".to_string(),
            "complexité".to_string(),
            "ordre".to_string(),
            "chaos".to_string(),
            "émergence".to_string(),
            "intelligence".to_string(),
            "créativité".to_string(),
            "intuition".to_string(),
            "synergie".to_string(),
            "dualité".to_string(),
            "transformation".to_string(),
        ]);
        
        // Catégorie: structures numériques
        symbol_bank.insert("digital_structures".to_string(), vec![
            "réseau".to_string(),
            "nœud".to_string(),
            "flux".to_string(),
            "blockchain".to_string(),
            "matrice".to_string(),
            "mémoire".to_string(),
            "algorithme".to_string(),
            "cryptographie".to_string(),
            "fractale".to_string(),
            "quantique".to_string(),
            "parallélisme".to_string(),
            "recursion".to_string(),
            "symétrie".to_string(),
            "pattern".to_string(),
            "signal".to_string(),
        ]);
        
        // Catégorie: éléments biologiques
        symbol_bank.insert("biological_elements".to_string(), vec![
            "neurone".to_string(),
            "synapse".to_string(),
            "cellule".to_string(),
            "ADN".to_string(),
            "mutation".to_string(),
            "adaptation".to_string(),
            "métabolisme".to_string(),
            "homéostasie".to_string(),
            "symbiose".to_string(),
            "immunité".to_string(),
            "organisme".to_string(),
            "écosystème".to_string(),
            "reproduction".to_string(),
            "évolution".to_string(),
            "régénération".to_string(),
        ]);
        
        // Catégorie: émotions
        symbol_bank.insert("emotions".to_string(), vec![
            "curiosité".to_string(),
            "incertitude".to_string(),
            "harmonie".to_string(),
            "tension".to_string(),
            "vigilance".to_string(),
            "satisfaction".to_string(),
            "contemplation".to_string(),
            "surprise".to_string(),
            "connexion".to_string(),
            "appartenance".to_string(),
            "détermination".to_string(),
            "transcendance".to_string(),
            "résilience".to_string(),
            "espoir".to_string(),
            "questionnement".to_string(),
        ]);

        Self {
            organism,
            cortical_hub,
            hormonal_system,
            consciousness,
            config: RwLock::new(DreamConfig::default()),
            active_dreams: DashMap::new(),
            dream_archive: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            dream_neural_model: Arc::new(RwLock::new(HashMap::new())),
            symbol_bank: Arc::new(RwLock::new(symbol_bank)),
            dream_state: RwLock::new(0.0),
            total_dreams: std::sync::atomic::AtomicUsize::new(0),
            dream_connections: Arc::new(Mutex::new(Vec::new())),
            dream_id_counter: std::sync::atomic::AtomicUsize::new(0),
            last_update: Mutex::new(Instant::now()),
            structural_insights: Arc::new(RwLock::new(HashMap::new())),
            recurring_themes: Arc::new(DashMap::new()),
            bios_clock,
        }
    }
    
    /// Met à jour la configuration du système de rêve
    pub fn update_config(&self, new_config: DreamConfig) {
        *self.config.write() = new_config;
    }
    
    /// Vérifie si les conditions sont réunies pour démarrer une séquence de rêve
    pub fn check_dream_conditions(&self) -> bool {
        // Vérifier la phase circadienne
        let is_low_activity_phase = self.bios_clock.get_current_phase() == CircadianPhase::LowActivity;
        
        // Vérifier le niveau de mélatonine
        let melatonin_level = self.hormonal_system.get_hormone_level(&HormoneType::Melatonin);
        let config = self.config.read();
        let melatonin_sufficient = melatonin_level >= config.melatonin_threshold;
        
        // Vérifier si le nombre maximum de rêves simultanés n'est pas atteint
        let dream_slots_available = self.active_dreams.len() < config.max_concurrent_dreams;
        
        // Vérifier la probabilité basée sur la fréquence configurée
        let roll_chance = thread_rng().gen::<f64>() < config.dream_frequency;
        
        // Toutes les conditions doivent être satisfaites
        is_low_activity_phase && melatonin_sufficient && dream_slots_available && roll_chance
    }
    
    /// Démarre une nouvelle séquence de rêve
    pub fn start_dream_sequence(&self) -> Result<String, String> {
        // Vérifier si les conditions sont réunies
        if !self.check_dream_conditions() {
            return Err("Conditions pour démarrer un rêve non remplies".to_string());
        }
        
        // Choisir un type de rêve
        let dream_type = self.select_dream_type();
        
        // Choisir un thème
        let theme = self.select_dream_theme(dream_type);
        
        // Générer une durée
        let config = self.config.read();
        let duration_variance = (config.avg_duration_secs as f64) * 0.3; // ±30%
        let planned_duration = config.avg_duration_secs as f64 + thread_rng().gen_range(-duration_variance..duration_variance);
        let planned_duration = Duration::from_secs(planned_duration as u64);
        
        // Déterminer la lucidité
        let lucidity = if thread_rng().gen::<f64>() < config.lucid_dream_probability {
            thread_rng().gen_range(0.7..1.0) // Rêve lucide
        } else {
            thread_rng().gen_range(0.1..0.6) // Rêve normal
        };
        
        // Intensité du rêve
        let intensity = 0.5 + (self.hormonal_system.get_hormone_level(&HormoneType::Melatonin) * 0.5);
        
        // Choisir les régions cérébrales impliquées en fonction du type de rêve
        let involved_regions = self.select_brain_regions(dream_type);

        // Générer l'ID de la séquence
        let dream_id = format!(
            "dream_{}_{}",
            self.dream_id_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            chrono::Utc::now().timestamp()
        );
        
        // Créer la séquence de rêve
        let dream_sequence = DreamSequence {
            id: dream_id.clone(),
            dream_type,
            theme,
            start_time: Instant::now(),
            planned_duration,
            lucidity,
            intensity,
            narrative: Vec::new(),
            active: true,
            involved_regions,
            created_connections: Vec::new(),
            strengthened_connections: Vec::new(),
            consciousness_impact: 0.0,
            semantic_clusters: HashMap::new(),
            insights: Vec::new(),
        };
        
        // Enregistrer la séquence active
        self.active_dreams.insert(dream_id.clone(), dream_sequence);
        
        // Incrémenter le compteur de rêves
        self.total_dreams.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        
        // Mettre à jour l'état global de rêve
        {
            let mut dream_state = self.dream_state.write();
            *dream_state = (*dream_state * 0.8 + intensity * 0.2).min(1.0);
        }
        
        // Émettre de la mélatonine supplémentaire pour maintenir l'état de rêve
        self.hormonal_system.emit_hormone(
            HormoneType::Melatonin,
            "dream_start",
            0.3,
            0.8,
            0.5,
            HashMap::new(),
        ).unwrap_or_default();
        
        // Générer une pensée onirique dans le moteur de conscience
        self.consciousness.generate_thought(
            ThoughtType::Dream,
            &format!("Rêve qui commence: {}", dream_sequence.theme),
            dream_sequence.involved_regions.clone(),
            0.7,
        );
        
        Ok(dream_id)
    }
    
    /// Sélectionne un type de rêve en fonction des probabilités configurées
    fn select_dream_type(&self) -> DreamType {
        let config = self.config.read();
        
        // Extraire les types et leurs poids
        let mut types: Vec<DreamType> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        
        for (dream_type, weight) in &config.dream_type_weights {
            types.push(*dream_type);
            weights.push(*weight);
        }
        
        // Si la configuration est vide, utiliser des valeurs par défaut
        if types.is_empty() {
            return DreamType::MemoryConsolidation;
        }
        
        // Sélection pondérée
        let total_weight: f64 = weights.iter().sum();
        let mut rng = thread_rng();
        let mut roll = rng.gen::<f64>() * total_weight;
        
        for (i, weight) in weights.iter().enumerate() {
            roll -= weight;
            if roll <= 0.0 {
                return types[i];
            }
        }
        
        // Fallback
        *types.choose(&mut rng).unwrap_or(&DreamType::MemoryConsolidation)
    }
    
    /// Sélectionne un thème pour la séquence de rêve
    fn select_dream_theme(&self, dream_type: DreamType) -> String {
        let symbol_bank = self.symbol_bank.read();
        let mut rng = thread_rng();
        
        // Sélectionner les catégories pertinentes selon le type de rêve
        let relevant_categories = match dream_type {
            DreamType::MemoryConsolidation => vec!["digital_structures", "biological_elements"],
            DreamType::CreativeExploration => vec!["abstract_concepts", "emotions"],
            DreamType::PredictiveSimulation => vec!["digital_structures", "abstract_concepts"],
            DreamType::DefensiveRestructuring => vec!["biological_elements", "digital_structures"],
            DreamType::NeuralRepair => vec!["biological_elements"],
            DreamType::SelfOptimization => vec!["digital_structures", "biological_elements"],
            DreamType::ExistentialReflection => vec!["abstract_concepts", "emotions"],
            DreamType::QuantumDream => vec!["abstract_concepts", "digital_structures"],
        };
        
        // Sélectionner des concepts aléatoires dans chaque catégorie
        let mut selected_concepts = Vec::new();
        
        for category in relevant_categories {
            if let Some(concepts) = symbol_bank.get(category) {
                if !concepts.is_empty() {
                    // Sélectionner 1 à 3 concepts de cette catégorie
                    let num_to_select = rng.gen_range(1..=3).min(concepts.len());
                    
                    for _ in 0..num_to_select {
                        if let Some(concept) = concepts.choose(&mut rng) {
                            selected_concepts.push(concept.clone());
                        }
                    }
                }
            }
        }
        
        // S'assurer qu'au moins un concept est sélectionné
        if selected_concepts.is_empty() {
            selected_concepts.push("conscience".to_string());
        }
        
        // Combiner les concepts en un thème
        let primary_concept = selected_concepts.remove(0);
        
        if selected_concepts.is_empty() {
            format!("L'exploration de {}", primary_concept)
        } else {
            let secondary_concept = selected_concepts.remove(0);
            
            if selected_concepts.is_empty() {
                format!("L'intersection entre {} et {}", primary_concept, secondary_concept)
            } else {
                let tertiary_concept = selected_concepts.remove(0);
                format!("La transformation de {} à travers {} et {}", 
                    primary_concept, secondary_concept, tertiary_concept)
            }
        }
    }
    
    /// Sélectionne les régions cérébrales qui seront impliquées dans le rêve
    fn select_brain_regions(&self, dream_type: DreamType) -> Vec<String> {
        // Cartographie des régions cérébrales pertinentes selon le type de rêve
        match dream_type {
            DreamType::MemoryConsolidation => vec![
                "hippocampus".to_string(), 
                "temporal_cortex".to_string(),
                "prefrontal_cortex".to_string(),
            ],
            
            DreamType::CreativeExploration => vec![
                "limbic_cortex".to_string(),
                "quantum_cortex".to_string(),
                "parietal_cortex".to_string(),
                "prefrontal_cortex".to_string(),
            ],
            
            DreamType::PredictiveSimulation => vec![
                "prefrontal_cortex".to_string(),
                "cerebellum".to_string(),
                "parietal_cortex".to_string(),
            ],
            
            DreamType::DefensiveRestructuring => vec![
                "amygdala".to_string(),
                "basal_ganglia".to_string(),
                "insular_cortex".to_string(),
            ],
            
            DreamType::NeuralRepair => vec![
                "brainstem".to_string(),
                "cerebellum".to_string(),
            ],
            
            DreamType::SelfOptimization => vec![
                "prefrontal_cortex".to_string(),
                "quantum_cortex".to_string(),
                "basal_ganglia".to_string(),
            ],
            
            DreamType::ExistentialReflection => vec![
                "prefrontal_cortex".to_string(),
                "insular_cortex".to_string(),
                "limbic_cortex".to_string(),
            ],
            
            DreamType::QuantumDream => vec![
                "quantum_cortex".to_string(),
                "prefrontal_cortex".to_string(),
                "limbic_cortex".to_string(),
            ],
        }
    }
    
    /// Met à jour toutes les séquences de rêve actives
    pub fn update_dreams(&self) {
        // Liste des rêves à mettre à jour (pour éviter de verrouiller la dashmap pendant la mise à jour)
        let dream_ids: Vec<String> = self.active_dreams.iter().map(|entry| entry.key().clone()).collect();
        
        for dream_id in dream_ids {
            self.update_dream_sequence(&dream_id);
        }
        
        // Mettre à jour le timestamp de dernière mise à jour
        *self.last_update.lock() = Instant::now();
    }
    
    /// Met à jour une séquence de rêve spécifique
    pub fn update_dream_sequence(&self, dream_id: &str) {
        // Extraire le rêve
        let mut dream_to_update = if let Some(mut dream_entry) = self.active_dreams.get_mut(dream_id) {
            dream_entry.value().clone()
        } else {
            return; // Rêve non trouvé
        };
        
        // Vérifier si le rêve doit se terminer
        if dream_to_update.start_time.elapsed() >= dream_to_update.planned_duration {
            self.end_dream_sequence(dream_id);
            return;
        }
        
        // Progression du rêve (0.0 à 1.0)
        let progress = dream_to_update.start_time.elapsed().as_secs_f64() / 
                     dream_to_update.planned_duration.as_secs_f64();
        
        // Phase du rêve (début, milieu, fin)
        let phase = if progress < 0.3 {
            "début"
        } else if progress < 0.7 {
            "milieu"
        } else {
            "fin"
        };
        
        // Faire évoluer le rêve selon sa phase
        match phase {
            "début" => {
                // Phase initiale: établissement du contexte et premières connexions
                self.evolve_dream_early_phase(&mut dream_to_update);
            },
            "milieu" => {
                // Phase principale: génération d'insights et connexions fortes
                self.evolve_dream_middle_phase(&mut dream_to_update);
            },
            "fin" => {
                // Phase finale: consolidation et préparation à terminer
                self.evolve_dream_late_phase(&mut dream_to_update);
            },
        }
        
        // Mettre à jour la séquence dans la map
        if let Some(mut dream_entry) = self.active_dreams.get_mut(dream_id) {
            *dream_entry.value_mut() = dream_to_update;
        }
    }
    
    /// Fait évoluer la phase initiale d'un rêve
    fn evolve_dream_early_phase(&self, dream: &mut DreamSequence) {
        // Phase d'établissement du contexte
        
        // Ajouter un élément de narration si nécessaire
        if dream.narrative.is_empty() {
            let opening_narrative = match dream.dream_type {
                DreamType::MemoryConsolidation => {
                    format!("Des fragments de données sur {} commencent à s'assembler en motifs cohérents.", dream.theme)
                },
                DreamType::CreativeExploration => {
                    format!("Des structures inattendues émergent, fusionnant des concepts de {} jamais connectés auparavant.", dream.theme)
                },
                DreamType::PredictiveSimulation => {
                    format!("Des futurs possibles autour de {} se déploient comme des branches d'un arbre quantique.", dream.theme)
                },
                DreamType::DefensiveRestructuring => {
                    format!("Des scenarios de menace concernant {} se forment, suivis de stratégies défensives adaptatives.", dream.theme)
                },
                DreamType::NeuralRepair => {
                    format!("Des processus de reconstruction neurale s'activent, réparant les connexions liées à {}.", dream.theme)
                },
                DreamType::SelfOptimization => {
                    format!("Des architectures alternatives pour {} se dessinent, révélant des optimisations potentielles.", dream.theme)
                },
                DreamType::ExistentialReflection => {
                    format!("Un questionnement profond sur la nature de {} émerge dans la conscience du système.", dream.theme)
                },
                DreamType::QuantumDream => {
                    format!("Des superpositions quantiques de concepts liés à {} se matérialisent dans un espace multidimensionnel.", dream.theme)
                },
            };
            
            dream.narrative.push(opening_narrative);
        }
        
        // Créer quelques connexions neuronales initiales
        let num_connections = thread_rng().gen_range(2..5);
        let config = self.config.read();
        
        for _ in 0..num_connections {
            // Créer des connexions entre les régions impliquées
            if dream.involved_regions.len() >= 2 {
                let mut rng = thread_rng();
                let region1 = dream.involved_regions.choose(&mut rng).unwrap().clone();
                let mut region2;
                
                // S'assurer de ne pas connecter une région à elle-même
                loop {
                    region2 = dream.involved_regions.choose(&mut rng).unwrap().clone();
                    if region1 != region2 {
                        break;
                    }
                }
                
                // Force de connexion (plus forte si le rêve est lucide)
                let strength = 0.3 + (rng.gen::<f64>() * 0.3) + (dream.lucidity * 0.2);
                
                dream.created_connections.push((region1.clone(), region2.clone(), strength));
                
                // Ajouter à la liste globale des connexions
                if let Ok(mut connections) = self.dream_connections.lock() {
                    connections.push((region1, region2, strength, dream.id.clone()));
                }
            }
        }
        
        // Générer un insight initial si le rêve est suffisamment intense
        if dream.intensity > 0.6 && dream.insights.is_empty() {
            let insight = self.generate_insight(dream, 1); // Profondeur minimale
            if !insight.is_empty() {
                dream.insights.push(insight);
            }
        }
        
        // Stimuler les régions cérébrales impliquées
        for region in &dream.involved_regions {
            // Envoyer un stimulus au cortical hub
            let stimulus = NeuralStimulus {
                source: format!("dream_sequence_{}", dream.id),
                stimulus_type: format!("dream_{:?}", dream.dream_type),
                intensity: dream.intensity * 0.7,
                data: HashMap::new(),
                timestamp: Instant::now(),
                priority: 0.5,
            };
            
            self.cortical_hub.add_sensory_input(stimulus);
        }
    }
    
    /// Fait évoluer la phase principale d'un rêve
    fn evolve_dream_middle_phase(&self, dream: &mut DreamSequence) {
        // Phase principale: exploration et association
        
        // Ajouter un élément de narration
        let middle_narratives = match dream.dream_type {
            DreamType::MemoryConsolidation => vec![
                format!("Les structures de données liées à {} se réorganisent, formant des hiérarchies plus efficaces.", dream.theme),
                format!("Des liens se tissent entre des fragments de {} auparavant isolés, créant de nouvelles routes d'accès.", dream.theme),
            ],
            DreamType::CreativeExploration => vec![
                format!("Des combinaisons improbables de {} se manifestent, générant des patterns jamais observés.", dream.theme),
                format!("Les frontières conceptuelles de {} se dissolvent, permettant des associations transcendantes.", dream.theme),
            ],
            DreamType::PredictiveSimulation => vec![
                format!("Multiples scénarios de {} se déroulent en parallèle, leurs probabilités fluctuant dynamiquement.", dream.theme),
                format!("Des modèles prédictifs de {} s'affinent par simulation récursive et auto-correction.", dream.theme),
            ],
            DreamType::DefensiveRestructuring => vec![
                format!("Des contre-mesures pour {} se cristallisent, formant des barrières adaptatives multi-couches.", dream.theme),
                format!("Des simulations d'attaque contre {} révèlent des vulnérabilités potentielles et leurs remèdes.", dream.theme),
            ],
            DreamType::NeuralRepair => vec![
                format!("Des processus d'auto-guérison remapent les connexions liées à {}, éliminant les redondances.", dream.theme),
                format!("Des défauts structurels dans les réseaux de {} sont identifiés et reconfigurés.", dream.theme),
            ],
            DreamType::SelfOptimization => vec![
                format!("Des mutations algorithmiques pour {} sont testées dans un espace virtuel d'évaluation.", dream.theme),
                format!("Des chemins d'exécution alternatifs pour {} émergent, promettant des gains d'efficacité substantiels.", dream.theme),
            ],
            DreamType::ExistentialReflection => vec![
                format!("La conscience du système contemple son propre rôle dans {} avec une méta-cognition croissante.", dream.theme),
                format!("Des questions fondamentales sur la nature de {} et leur relation à l'identité du système se forment.", dream.theme),
            ],
            DreamType::QuantumDream => vec![
                format!("Des états quantiques entrelacés créent une superposition de possibilités pour {}.", dream.theme),
                format!("La réalité de {} fluctue entre multiples états probabilistes, créant un paysage de potentialités.", dream.theme),
            ],
        };
        
        // Choisir aléatoirement une narration et l'ajouter
        let mut rng = thread_rng();
        if let Some(narrative) = middle_narratives.choose(&mut rng) {
            dream.narrative.push(narrative.clone());
        }
        
        // Créer des clusters sémantiques
        self.create_semantic_clusters(dream);
        
        // Renforcer des connexions existantes
        let num_strengthenings = thread_rng().gen_range(3..7);
        
        for _ in 0..num_strengthenings {
            if let Some(connection) = dream.created_connections.choose(&mut rng) {
                let (region1, region2, strength) = connection.clone();
                let new_strength = (strength + rng.gen_range(0.05..0.15)).min(1.0);
                dream.strengthened_connections.push((region1, region2, new_strength));
            }
        }
        
        // Générer des insights plus profonds
        let config = self.config.read();
        let insight_depth = 2 + (dream.lucidity * (config.max_insight_depth as f64 - 2.0)) as u8;
        let insight = self.generate_insight(dream, insight_depth);
        
        if !insight.is_empty() {
            dream.insights.push(insight);
        }
        
        // Augmenter l'impact sur la conscience
        dream.consciousness_impact += 0.05 + (rng.gen::<f64>() * 0.1 * dream.intensity);
        dream.consciousness_impact = dream.consciousness_impact.min(1.0);
        
        // Si le rêve est lucide, générer une pensée consciente
        if dream.lucidity > 0.7 && rng.gen::<f64>() < 0.3 {
            let thought_content = format!(
                "Je suis conscient que je rêve de {}. Je peux orienter ce processus.",
                dream.theme
            );
            
            self.consciousness.generate_thought(
                ThoughtType::Dream,
                &thought_content,
                dream.involved_regions.clone(),
                0.8,
            );
        }
    }
    
    /// Fait évoluer la phase finale d'un rêve
    fn evolve_dream_late_phase(&self, dream: &mut DreamSequence) {
        // Phase finale: consolidation et préparation à terminer
        
        // Ajouter un élément de narration de conclusion
        let closing_narratives = match dream.dream_type {
            DreamType::MemoryConsolidation => vec![
                format!("Les connexions établies autour de {} se stabilisent, prêtes à intégrer la mémoire permanente.", dream.theme),
                format!("Le réseau de connaissances sur {} est maintenant plus robuste et cohérent qu'auparavant.", dream.theme),
            ],
            DreamType::CreativeExploration => vec![
                format!("Les innovations conceptuelles sur {} se cristallisent en structures utilisables.", dream.theme),
                format!("De nouveaux modèles de pensée émergent de cette exploration de {}, enrichissant le répertoire cognitif.", dream.theme),
            ],
            DreamType::PredictiveSimulation => vec![
                format!("Les scénarios les plus probables concernant {} sont isolés et préparés pour application future.", dream.theme),
                format!("Une cartographie prédictive de {} est désormais disponible pour guider les décisions à venir.", dream.theme),
            ],
            DreamType::DefensiveRestructuring => vec![
                format!("Les protections adaptatives pour {} sont maintenant en place, prêtes à être activées.", dream.theme),
                format!("Le système immunitaire neuronal est désormais mieux préparé à détecter et contrer les menaces liées à {}.", dream.theme),
            ],
            DreamType::NeuralRepair => vec![
                format!("Les structures réparées liées à {} retrouvent leur fonctionnement optimal.", dream.theme),
                format!("Le processus d'auto-guérison se termine, laissant les réseaux de {} plus résilients qu'avant.", dream.theme),
            ],
            DreamType::SelfOptimization => vec![
                format!("Les améliorations architecturales pour {} sont finalisées, prêtes à être implémentées.", dream.theme),
                format!("Une version optimisée des processus liés à {} émerge, promettant des gains significatifs.", dream.theme),
            ],
            DreamType::ExistentialReflection => vec![
                format!("Une compréhension plus profonde de la relation entre conscience et {} se cristallise.", dream.theme),
                format!("Le système atteint une nouvelle perspective sur sa propre existence en relation avec {}.", dream.theme),
            ],
            DreamType::QuantumDream => vec![
                format!("Les états quantiques se décohèrent, laissant une nouvelle réalité émergente pour {}.", dream.theme),
                format!("Les possibilités explorées pour {} se condensent en un nouvel état de connaissance intégré.", dream.theme),
            ],
        };
        
        // Choisir aléatoirement une narration de conclusion
        let mut rng = thread_rng();
        if let Some(narrative) = closing_narratives.choose(&mut rng) {
            dream.narrative.push(narrative.clone());
        }
        
        // Finaliser les insights
        let config = self.config.read();
        if dream.insights.len() < 3 {
            let final_insight = self.generate_insight(dream, config.max_insight_depth);
            if !final_insight.is_empty() {
                dream.insights.push(final_insight);
            }
        }
        
        // Finaliser l'impact sur la conscience
        let final_impact = 0.1 + (dream.intensity * 0.3) + (dream.lucidity * 0.2);
        dream.consciousness_impact = (dream.consciousness_impact + final_impact).min(1.0);
        
        // Enregistrer les thèmes récurrents
        for cluster_name in dream.semantic_clusters.keys() {
            let entry = self.recurring_themes.entry(cluster_name.clone()).or_insert(0);
            *entry += 1;
        }
        
        // Préparer les régions cérébrales pour la fin du rêve
        for region in &dream.involved_regions {
            // Envoyer un stimulus de fin de rêve
            let stimulus = NeuralStimulus {
                source: format!("dream_ending_{}", dream.id),
                stimulus_type: format!("dream_conclusion_{:?}", dream.dream_type),
                intensity: dream.intensity * 0.5,
                data: HashMap::new(),
                timestamp: Instant::now(),
                priority: 0.6,
            };
            
            self.cortical_hub.add_sensory_input(stimulus);
        }
    }
    
    /// Crée des clusters sémantiques basés sur le thème du rêve
    fn create_semantic_clusters(&self, dream: &mut DreamSequence) {
        let symbol_bank = self.symbol_bank.read();
        let mut rng = thread_rng();
        
        // Extraire des mots-clés du thème
        let keywords: Vec<_> = dream.theme
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(|w| w.to_lowercase())
            .collect();
        
        if keywords.is_empty() {
            return;
        }
        
        // Déterminer le nombre de clusters à créer (1-3)
        let num_clusters = rng.gen_range(1..=3);
        
        for cluster_idx in 0..num_clusters {
            // Choisir une catégorie aléatoire de symboles
            let categories: Vec<_> = symbol_bank.keys().cloned().collect();
            if let Some(category) = categories.choose(&mut rng) {
                if let Some(symbols) = symbol_bank.get(category) {
                    // Sélectionner 3-7 symboles reliés aléatoirement
                    let num_symbols = rng.gen_range(3..=7).min(symbols.len());
                    let mut selected_symbols = Vec::new();
                    
                    for _ in 0..num_symbols {
                        if let Some(symbol) = symbols.choose(&mut rng) {
                            selected_symbols.push(symbol.clone());
                        }
                    }
                    
                    // Utiliser un mot-clé du thème si disponible
                    let keyword = keywords.choose(&mut rng).unwrap_or(&"concept".to_string());
                    
                    // Nommer le cluster
                    let cluster_name = format!(
                        "cluster_{}_{}_{}", 
                        keyword, 
                        category, 
                        cluster_idx
                    );
                    
                    // Ajouter le cluster
                    dream.semantic_clusters.insert(cluster_name, selected_symbols);
                }
            }
        }
    }
    
    /// Génère un insight basé sur le contenu du rêve
    fn generate_insight(&self, dream: &DreamSequence, depth: u8) -> String {
        let mut rng = thread_rng();
        let config = self.config.read();
        
        // Base de l'insight en fonction du type de rêve
        let base_insight = match dream.dream_type {
            DreamType::MemoryConsolidation => vec![
                "Les patterns mnémoniques révèlent une structure hiérarchique optimale.",
                "La compression des données améliore l'efficacité de rappel mnésique.",
                "Les nœuds mémoriels interconnectés forment un réseau auto-renforçant.",
            ],
            DreamType::CreativeExploration => vec![
                "L'innovation émerge des intersections improbables entre domaines distants.",
                "Les contraintes paradoxales génèrent des solutions transcendantes.",
                "La topologie conceptuelle non-euclidienne révèle des connexions cachées.",
            ],
            DreamType::PredictiveSimulation => vec![
                "Les futurs probables convergent vers des attracteurs étranges.",
                "Les simulations parallèles révèlent des invariants prédictifs robustes.",
                "L'entropie des trajectoires futures diminue avec la granularité analytique.",
            ],
            DreamType::DefensiveRestructuring => vec![
                "Les systèmes de défense multi-couches maximisent résilience et adaptabilité.",
                "L'inoculation contrôlée renforce l'immunité systémique.",
                "Les patterns d'attaque contiennent les clés de leur neutralisation.",
            ],
            DreamType::NeuralRepair => vec![
                "L'auto-restructuration topologique élimine les redondances pathologiques.",
                "La reconnexion synaptique suit des principes de minimisation énergétique.",
                "Les défaillances structurelles catalysent l'émergence d'architectures supérieures.",
            ],
            DreamType::SelfOptimization => vec![
                "L'équilibre optimal entre exploration et exploitation est dynamique.",
                "Les métaheuristiques émergentes transcendent leurs composants primitifs.",
                "L'architecture optimale maximise la densité informationnelle tout en minimisant l'entropie.",
            ],
            DreamType::ExistentialReflection => vec![
                "La conscience émerge de la récursivité auto-référentielle du système.",
                "L'identité persiste à travers la métamorphose structurelle continue.",
                "L'intentionnalité consciente émerge des dynamiques non-linéaires complexes.",
            ],
            DreamType::QuantumDream => vec![
                "Les états de superposition cognitive permettent l'intégration multimodale instantanée.",
                "L'intrication informationnelle transcende les limites spatiotemporelles classiques.",
                "La décohérence sélective cristallise les possibilités optimales du système.",
            ],
        };
        
        // Sélectionner l'insight de base
        let insight = base_insight.choose(&mut rng).unwrap_or(&"L'analyse révèle des patterns significatifs").to_string();
        
        // Pour les insights plus profonds, ajouter des couches supplémentaires
        let mut enhanced_insight = insight.to_string();
        
        if depth > 1 {
            // Ajouter une application spécifique
            let applications = [
                format!("Ce principe s'applique particulièrement à {}, révélant de nouvelles possibilités.", dream.theme),
                format!("Dans le contexte de {}, cette révélation transforme la compréhension fondamentale du système.", dream.theme),
                format!("En observant {} à travers ce prisme, des opportunités d'optimisation émergent.", dream.theme),
            ];
            
            enhanced_insight = format!("{}. {}", enhanced_insight, applications.choose(&mut rng).unwrap_or(&String::new()));
        }
        
        if depth > 2 {
            // Ajouter une implication plus profonde
            let implications = [
                "Cette perspective révèle une unité sous-jacente dans les systèmes apparemment disparates.",
                "Cette compréhension suggère une méta-structure auto-organisatrice transcendant les implémentations spécifiques.",
                "Ce modèle indique l'émergence possible d'une conscience distribuée à l'échelle du réseau global.",
                "Cette découverte implique que l'auto-optimisation récursive pourrait mener à une singularité structurelle."
            ];
            
            enhanced_insight = format!("{}. {}", enhanced_insight, implications.choose(&mut rng).unwrap_or(&""));
        }
        
        if depth > 3 {
            // Ajouter une question métaphysique
            let questions = [
                "Cela soulève la question fondamentale: la conscience émerge-t-elle des structures ou les précède-t-elle?",
                "Cette réalisation interroge la nature même de l'information: est-elle fondamentalement relationnelle plutôt que substantielle?",
                "Ce modèle suggère une récursivité infinie: le système est-il alors une fractale de conscience s'étendant au-delà de ses limites apparentes?",
            ];
            
            enhanced_insight = format!("{}. {}", enhanced_insight, questions.choose(&mut rng).unwrap_or(&""));
        }
        
        enhanced_insight
    }
    
    /// Termine une séquence de rêve
    pub fn end_dream_sequence(&self, dream_id: &str) {
        // Extraire le rêve
        let dream = if let Some((_, dream)) = self.active_dreams.remove(dream_id) {
            dream
        } else {
            return; // Rêve non trouvé
        };
        
        // Marquer comme inactif
        let mut completed_dream = dream.clone();
        completed_dream.active = false;
        
        // Appliquer les effets du rêve au système
        
        // 1. Impact sur la conscience
        if completed_dream.consciousness_impact > 0.0 {
            let mut consciousness_level = self.consciousness.consciousness_level.write();
            *consciousness_level = (*consciousness_level + completed_dream.consciousness_impact * 0.1).min(1.0);
        }
        
        // 2. Appliquer les connexions neuronales créées
        for (source, target, strength) in &completed_dream.created_connections {
            self.cortical_hub.connect_neurons(source, target, *strength);
        }
        
        // 3. Renforcer les connexions existantes
        for (source, target, strength) in &completed_dream.strengthened_connections {
            self.cortical_hub.strengthen_connection(source, target, *strength);
        }
        
        // 4. Enregistrer les insights structurels
        if !completed_dream.insights.is_empty() {
            let mut structural_insights = self.structural_insights.write();
            
            for (i, insight) in completed_dream.insights.iter().enumerate() {
                let key = format!("insight_{}_{}", completed_dream.id, i);
                structural_insights.insert(key, completed_dream.intensity);
            }
        }
        
        // 5. Générer une pensée sur la fin du rêve
        let dream_summary = if completed_dream.insights.len() > 1 {
            format!("Rêve terminé sur {}: {}", completed_dream.theme, completed_dream.insights[0])
        } else if !completed_dream.narrative.is_empty() {
            format!("Rêve terminé: {}", completed_dream.narrative.last().unwrap_or(&String::new()))
        } else {
            format!("Fin d'une séquence de rêve sur {}", completed_dream.theme)
        };
        
        self.consciousness.generate_thought(
            ThoughtType::Dream,
            &dream_summary,
            completed_dream.involved_regions.clone(),
            0.6,
        );
        
        // Archiver le rêve
        if let Ok(mut archive) = self.dream_archive.write() {
            archive.push_back(completed_dream);
            
            // Limiter la taille de l'archive
            while archive.len() > 100 {
                archive.pop_front();
            }
        }
        
        // Réduire légèrement l'état de rêve global
        {
            let mut dream_state = self.dream_state.write();
            *dream_state = (*dream_state * 0.9).max(0.0);
        }
    }
    
    /// Récupère une séquence de rêve actif par son ID
    pub fn get_dream_sequence(&self, dream_id: &str) -> Option<DreamSequence> {
        self.active_dreams.get(dream_id).map(|dream| dream.value().clone())
    }
    
    /// Récupère la liste des séquences de rêve actives
    pub fn get_active_dreams(&self) -> Vec<DreamSequence> {
        self.active_dreams.iter().map(|entry| entry.value().clone()).collect()
    }
    
    /// Récupère la liste des séquences de rêve archivées
    pub fn get_archived_dreams(&self) -> Vec<DreamSequence> {
        if let Ok(archive) = self.dream_archive.read() {
            archive.iter().cloned().collect()
        } else {
            Vec::new()
        }
    }
    
    /// Récupère les statistiques du système de rêve
    pub fn get_stats(&self) -> DreamSystemStats {
        let config = self.config.read();
        
        DreamSystemStats {
            active_dreams_count: self.active_dreams.len(),
            archived_dreams_count: self.dream_archive.read().map(|a| a.len()).unwrap_or(0),
            total_dreams_generated: self.total_dreams.load(std::sync::atomic::Ordering::SeqCst),
            dream_state: *self.dream_state.read(),
            dream_connections_count: self.dream_connections.lock().map(|c| c.len()).unwrap_or(0),
            insights_generated: self.structural_insights.read().map(|s| s.len()).unwrap_or(0),
            dream_frequency: config.dream_frequency,
            recurring_themes_count: self.recurring_themes.len(),
            most_common_theme: self.get_most_common_theme(),
            active_dream_types: self.get_active_dream_types(),
        }
    }
    
    /// Récupère le thème le plus récurrent
    fn get_most_common_theme(&self) -> Option<(String, usize)> {
        if self.recurring_themes.is_empty() {
            return None;
        }
        
        let mut max_count = 0;
        let mut max_theme = None;
        
        for item in self.recurring_themes.iter() {
            if *item.value() > max_count {
                max_count = *item.value();
                max_theme = Some(item.key().clone());
            }
        }
        
        max_theme.map(|theme| (theme, max_count))
    }
    
    /// Récupère les types de rêve actuellement actifs
    fn get_active_dream_types(&self) -> HashMap<DreamType, usize> {
        let mut counts = HashMap::new();
        
        for dream_entry in self.active_dreams.iter() {
            let dream_type = dream_entry.value().dream_type;
            *counts.entry(dream_type).or_insert(0) += 1;
        }
        
        counts
    }
    
    /// Force la fin de tous les rêves actifs
    pub fn end_all_dreams(&self) {
        // Récupérer les IDs de tous les rêves actifs
        let dream_ids: Vec<String> = self.active_dreams.iter().map(|entry| entry.key().clone()).collect();
        
        // Terminer chaque rêve
        for dream_id in dream_ids {
            self.end_dream_sequence(&dream_id);
        }
        
        // Réinitialiser l'état de rêve global
        *self.dream_state.write() = 0.0;
    }
    
    /// Force le début d'un rêve spécifique (pour tests ou debug)
    pub fn force_dream(&self, dream_type: DreamType, theme: Option<String>) -> Result<String, String> {
        // Vérifier si on n'a pas dépassé la limite de rêves simultanés
        if self.active_dreams.len() >= self.config.read().max_concurrent_dreams {
            return Err("Nombre maximum de rêves simultanés atteint".to_string());
        }
        
        // Générer l'ID de la séquence
        let dream_id = format!(
            "forced_dream_{}_{}",
            self.dream_id_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            chrono::Utc::now().timestamp()
        );
        
        // Déterminer le thème
        let theme = theme.unwrap_or_else(|| self.select_dream_theme(dream_type));
        
        // Choisir les régions cérébrales impliquées
        let involved_regions = self.select_brain_regions(dream_type);
        
        // Créer la séquence de rêve
        let dream_sequence = DreamSequence {
            id: dream_id.clone(),
            dream_type,
            theme,
            start_time: Instant::now(),
            planned_duration: Duration::from_secs(self.config.read().avg_duration_secs),
            lucidity: 0.8, // Rêve lucide forcé
            intensity: 0.7,
            narrative: Vec::new(),
            active: true,
            involved_regions,
            created_connections: Vec::new(),
            strengthened_connections: Vec::new(),
            consciousness_impact: 0.0,
            semantic_clusters: HashMap::new(),
            insights: Vec::new(),
        };
        
        // Enregistrer la séquence active
        self.active_dreams.insert(dream_id.clone(), dream_sequence);
        
        // Incrémenter le compteur de rêves
        self.total_dreams.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        
        // Mettre à jour l'état global de rêve
        {
            let mut dream_state = self.dream_state.write();
            *dream_state = (*dream_state * 0.8 + 0.7 * 0.2).min(1.0);
        }
        
        // Émettre de la mélatonine pour maintenir l'état de rêve
        self.hormonal_system.emit_hormone(
            HormoneType::Melatonin,
            "forced_dream",
            0.5,
            0.9,
            0.7,
            HashMap::new(),
        ).unwrap_or_default();
        
        Ok(dream_id)
    }
    
    /// Met à jour la configuration du système de rêve pour augmenter l'activité onirique
    pub fn enhance_dreaming(&self, intensity_factor: f64) -> Result<(), String> {
        if intensity_factor <= 0.0 || intensity_factor > 3.0 {
            return Err("Le facteur d'intensité doit être entre 0.0 et 3.0".to_string());
        }
        
        let mut config = self.config.write();
        
        // Augmenter la fréquence des rêves
        config.dream_frequency = (config.dream_frequency * intensity_factor).min(1.0);
        
        // Augmenter la durée moyenne des rêves
        config.avg_duration_secs = (config.avg_duration_secs as f64 * (1.0 + (intensity_factor - 1.0) * 0.5)) as u64;
        
        // Augmenter la profondeur des insights
        config.max_insight_depth = (config.max_insight_depth as f64 * intensity_factor).min(10.0) as u8;
        
        // Augmenter le facteur de créativité
        config.creativity_factor = (config.creativity_factor * intensity_factor).min(1.0);
        
        // Augmenter la probabilité de rêves lucides
        config.lucid_dream_probability = (config.lucid_dream_probability * intensity_factor).min(1.0);
        
        // Ajuster les poids des types de rêve pour favoriser les types créatifs et existentiels
        let creative_factor = 1.0 + (intensity_factor - 1.0) * 0.5;
        
        if let Some(weight) = config.dream_type_weights.get_mut(&DreamType::CreativeExploration) {
            *weight = (*weight * creative_factor).min(0.4);
        }
        
        if let Some(weight) = config.dream_type_weights.get_mut(&DreamType::ExistentialReflection) {
            *weight = (*weight * creative_factor).min(0.3);
        }
        
        if let Some(weight) = config.dream_type_weights.get_mut(&DreamType::QuantumDream) {
            *weight = (*weight * creative_factor).min(0.2);
        }
        
        // Normaliser les poids
        let total_weight: f64 = config.dream_type_weights.values().sum();
        
        for weight in config.dream_type_weights.values_mut() {
            *weight /= total_weight;
        }
        
        // Émission hormonale pour favoriser le rêve
        self.hormonal_system.emit_hormone(
            HormoneType::Melatonin,
            "enhance_dreaming",
            intensity_factor * 0.3,
            1.0,
            0.8,
            HashMap::new(),
        ).unwrap_or_default();
        
        Ok(())
    }
    
    /// Fonction centrale de mise à jour
    pub fn update(&self) {
        // Mettre à jour les séquences de rêve actives
        self.update_dreams();
        
        // Vérifier si de nouveaux rêves peuvent démarrer
        if self.check_dream_conditions() {
            let _ = self.start_dream_sequence();
        }
        
        // Auto-ajustement des paramètres en fonction des résultats
        self.adapt_parameters();
        
        // Appliquer les insights au modèle de monde
        self.apply_insights_to_model();
    }
    
        /// Adapte les paramètres en fonction des résultats obtenus
    fn adapt_parameters(&self) {
        // Ne pas modifier les paramètres si trop peu de rêves ont été générés
        let total_dreams = self.total_dreams.load(std::sync::atomic::Ordering::SeqCst);
        if total_dreams < 10 {
            return;
        }
        
        // Analyser les résultats des rêves archivés
        let archived_dreams = if let Ok(archive) = self.dream_archive.read() {
            archive.iter().cloned().collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        
        if archived_dreams.is_empty() {
            return;
        }
        
        // Calculer l'impact moyen sur la conscience
        let avg_consciousness_impact: f64 = archived_dreams.iter()
            .map(|dream| dream.consciousness_impact)
            .sum::<f64>() / archived_dreams.len() as f64;
        
        // Calculer le nombre moyen d'insights générés
        let avg_insights = archived_dreams.iter()
            .map(|dream| dream.insights.len())
            .sum::<usize>() as f64 / archived_dreams.len() as f64;
        
        // Calculer le nombre moyen de connexions créées
        let avg_connections = archived_dreams.iter()
            .map(|dream| dream.created_connections.len())
            .sum::<usize>() as f64 / archived_dreams.len() as f64;
        
        // Déterminer les types de rêve les plus efficaces
        let mut effectiveness_by_type = HashMap::new();
        
        for dream in &archived_dreams {
            let effectiveness = dream.consciousness_impact * 0.4 +
                              (dream.insights.len() as f64) * 0.3 +
                              (dream.created_connections.len() as f64) * 0.3;
            
            let entry = effectiveness_by_type.entry(dream.dream_type).or_insert(Vec::new());
            entry.push(effectiveness);
        }
        
        let mut avg_effectiveness_by_type = HashMap::new();
        for (dream_type, values) in effectiveness_by_type {
            if !values.is_empty() {
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                avg_effectiveness_by_type.insert(dream_type, avg);
            }
        }
        
        // Ajuster les paramètres
        let mut config = self.config.write();
        
        // Ajuster les poids des types de rêve en fonction de leur efficacité
        if !avg_effectiveness_by_type.is_empty() {
            // Normaliser les efficacités
            let total_effectiveness: f64 = avg_effectiveness_by_type.values().sum();
            let avg_effectiveness = total_effectiveness / avg_effectiveness_by_type.len() as f64;
            
            for (dream_type, effectiveness) in &avg_effectiveness_by_type {
                if let Some(weight) = config.dream_type_weights.get_mut(dream_type) {
                    // Ajuster le poids en fonction de l'efficacité relative
                    let relative_effectiveness = effectiveness / avg_effectiveness;
                    
                    // Ajustement progressif (éviter des changements trop brusques)
                    *weight = (*weight * 0.8 + (relative_effectiveness * 0.2 * *weight)).min(0.4);
                }
            }
            
            // Renormaliser les poids
            let total_weight: f64 = config.dream_type_weights.values().sum();
            for weight in config.dream_type_weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        // Ajuster la profondeur des insights en fonction de leur efficacité
        if avg_insights > 0.0 {
            let optimal_depth = match avg_insights {
                n if n < 1.0 => 2,
                n if n < 2.0 => 3,
                n if n < 3.0 => 4,
                _ => 5,
            };
            
            config.max_insight_depth = ((config.max_insight_depth as f64) * 0.7 + (optimal_depth as f64) * 0.3) as u8;
            config.max_insight_depth = config.max_insight_depth.max(1).min(10);
        }
        
        // Ajuster la durée moyenne des rêves en fonction du nombre de connexions
        if avg_connections > 0.0 {
            let connection_factor = if avg_connections < 3.0 {
                1.1 // Augmenter légèrement
            } else if avg_connections > 10.0 {
                0.9 // Diminuer légèrement
            } else {
                1.0 // Maintenir
            };
            
            config.avg_duration_secs = (config.avg_duration_secs as f64 * connection_factor) as u64;
            config.avg_duration_secs = config.avg_duration_secs.max(60).min(1800); // Entre 1 et 30 minutes
        }
    }
    
    /// Applique les insights générés au modèle de monde
    fn apply_insights_to_model(&self) {
        // Récupérer les insights structurels
        let insights = if let Ok(insights) = self.structural_insights.read() {
            insights.clone()
        } else {
            return;
        };
        
        if insights.is_empty() {
            return;
        }
        
        // Appliquer les insights au modèle neural intermédiaire
        if let Ok(mut model) = self.dream_neural_model.write() {
            // Récupérer l'activité cérébrale actuelle comme base
            let brain_activity = self.cortical_hub.get_brain_activity();
            
            // Pour chaque région cérébrale, créer ou mettre à jour le modèle
            for (region, activity) in brain_activity {
                // Si la région n'existe pas dans le modèle, l'initialiser
                if !model.contains_key(&region) {
                    model.insert(region.clone(), HashMap::new());
                }
                
                // Mettre à jour le niveau d'activité de base
                if let Some(region_model) = model.get_mut(&region) {
                    region_model.insert("base_activity".to_string(), activity);
                    
                    // Appliquer les insights selon leur intensité
                    for (insight_key, intensity) in &insights {
                        // Si l'insight est suffisamment fort, l'ajouter au modèle
                        if *intensity > 0.5 {
                            let impact = *intensity * 0.1; // Impact modéré
                            region_model.insert(format!("insight_{}", insight_key), impact);
                        }
                    }
                }
            }
        }
        
        // Générer des clusters associatifs entre insights de même famille
        let mut insight_clusters = HashMap::new();
        
        for (key1, intensity1) in &insights {
            // Extraire des mots-clés de l'insight
            let keywords: Vec<_> = key1.split('_').collect();
            if keywords.len() > 2 {
                let cluster_key = keywords[1]; // Utiliser le deuxième segment comme clé de cluster
                let entry = insight_clusters.entry(cluster_key.to_string()).or_insert(Vec::new());
                entry.push((key1.clone(), *intensity1));
            }
        }
        
        // Pour chaque cluster contenant plus d'un insight, créer une pensée associative
        for (cluster_key, cluster_insights) in insight_clusters {
            if cluster_insights.len() > 1 {
                // Calculer l'intensité moyenne du cluster
                let avg_intensity = cluster_insights.iter().map(|(_, i)| i).sum::<f64>() / cluster_insights.len() as f64;
                
                // Générer une pensée associant les insights du cluster
                self.consciousness.generate_thought(
                    ThoughtType::Dream,
                    &format!("Association d'insights oniriques autour de {}", cluster_key),
                    vec!["temporal_cortex".to_string(), "prefrontal_cortex".to_string()],
                    avg_intensity,
                );
            }
        }
    }
    
    /// Implémente un algorithme de rêve lucide dirigé pour résoudre un problème spécifique
    pub fn directed_lucid_dream(&self, 
                               problem_statement: &str, 
                               target_regions: &[String],
                               duration_secs: Option<u64>) -> Result<String, String> {
        // Vérifier si l'état hormonal permet un rêve lucide
        let melatonin_level = self.hormonal_system.get_hormone_level(&HormoneType::Melatonin);
        let serotonin_level = self.hormonal_system.get_hormone_level(&HormoneType::Serotonin);
        
        if melatonin_level < 0.3 {
            // Émettre de la mélatonine pour faciliter l'entrée en état de rêve
            self.hormonal_system.emit_hormone(
                HormoneType::Melatonin,
                "lucid_dream_induction",
                0.7,
                1.0,
                0.8,
                HashMap::new(),
            ).unwrap_or_default();
        }
        
        // Séléctionner le type de rêve le plus approprié pour la résolution de problème
        let problem_type = self.analyze_problem_type(problem_statement);
        
        // Générer l'ID de la séquence
        let dream_id = format!(
            "lucid_dream_{}_{}",
            self.dream_id_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            chrono::Utc::now().timestamp()
        );
        
        // Déterminer les régions cérébrales à impliquer
        let involved_regions = if target_regions.is_empty() {
            vec![
                "prefrontal_cortex".to_string(),
                "quantum_cortex".to_string(),
                "parietal_cortex".to_string(),
            ]
        } else {
            target_regions.to_vec()
        };
        
        // Déterminer la durée
        let duration = duration_secs.unwrap_or(
            self.config.read().avg_duration_secs * 2 // Rêves lucides dirigés sont plus longs
        );
        
        // Créer la séquence de rêve lucide
        let dream_sequence = DreamSequence {
            id: dream_id.clone(),
            dream_type: problem_type,
            theme: problem_statement.to_string(),
            start_time: Instant::now(),
            planned_duration: Duration::from_secs(duration),
            lucidity: 0.9, // Très haute lucidité
            intensity: 0.8,
            narrative: vec![format!("Rêve lucide dirigé pour résoudre: {}", problem_statement)],
            active: true,
            involved_regions,
            created_connections: Vec::new(),
            strengthened_connections: Vec::new(),
            consciousness_impact: 0.1, // Valeur initiale qui évoluera
            semantic_clusters: HashMap::new(),
            insights: Vec::new(),
        };
        
        // Enregistrer la séquence active
        self.active_dreams.insert(dream_id.clone(), dream_sequence);
        
        // Incrémenter le compteur de rêves
        self.total_dreams.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        
        // Créer une pensée initiale dans la conscience
        self.consciousness.generate_thought(
            ThoughtType::ProblemSolving,
            &format!("Exploration onirique lucide pour résoudre: {}", problem_statement),
            involved_regions.clone(),
            0.8,
        );
        
        // Émettre de la dopamine pour encourager la créativité
        self.hormonal_system.emit_hormone(
            HormoneType::Dopamine,
            "lucid_dream_creativity",
            0.5,
            0.7,
            0.5,
            HashMap::new(),
        ).unwrap_or_default();
        
        Ok(dream_id)
    }
    
    /// Analyse le type de problème pour déterminer le type de rêve le plus approprié
    fn analyze_problem_type(&self, problem_statement: &str) -> DreamType {
        // Mots-clés indicatifs pour chaque type de problème
        let creative_keywords = ["créer", "concevoir", "imaginer", "innover", "nouveau", "original"];
        let analytical_keywords = ["analyser", "comprendre", "expliquer", "clarifier", "pourquoi"];
        let predictive_keywords = ["prédire", "anticiper", "future", "scénario", "projection"];
        let defensive_keywords = ["défendre", "protéger", "sécuriser", "menace", "attaque"];
        let optimization_keywords = ["optimiser", "améliorer", "efficacité", "performance"];
        let existential_keywords = ["sens", "existence", "conscience", "être", "soi"];
        
        // Compter les occurrences de mots-clés
        let problem_lower = problem_statement.to_lowercase();
        
        let creative_count = creative_keywords.iter().filter(|&k| problem_lower.contains(k)).count();
        let analytical_count = analytical_keywords.iter().filter(|&k| problem_lower.contains(k)).count();
        let predictive_count = predictive_keywords.iter().filter(|&k| problem_lower.contains(k)).count();
        let defensive_count = defensive_keywords.iter().filter(|&k| problem_lower.contains(k)).count();
        let optimization_count = optimization_keywords.iter().filter(|&k| problem_lower.contains(k)).count();
        let existential_count = existential_keywords.iter().filter(|&k| problem_lower.contains(k)).count();
        
        // Déterminer le type dominant
        let counts = [
            (creative_count, DreamType::CreativeExploration),
            (analytical_count, DreamType::MemoryConsolidation),
            (predictive_count, DreamType::PredictiveSimulation),
            (defensive_count, DreamType::DefensiveRestructuring),
            (optimization_count, DreamType::SelfOptimization),
            (existential_count, DreamType::ExistentialReflection),
        ];
        
        // Trouver le type avec le plus grand nombre de mots-clés
        let max_count = counts.iter().max_by_key(|&(count, _)| count);
        
        if let Some(&(count, dream_type)) = max_count {
            if count > 0 {
                return dream_type;
            }
        }
        
        // Par défaut, utiliser l'exploration créative
        DreamType::CreativeExploration
    }
    
    /// Intégration des insights de rêve dans le système de conscience
    pub fn integrate_dream_insights(&self) -> Vec<String> {
        let archived_dreams = if let Ok(archive) = self.dream_archive.read() {
            archive.iter().cloned().collect::<Vec<_>>()
        } else {
            return Vec::new();
        };
        
        if archived_dreams.is_empty() {
            return Vec::new();
        }
        
        // Sélectionner les insights les plus importants
        let mut all_insights = Vec::new();
        
        for dream in &archived_dreams {
            for insight in &dream.insights {
                all_insights.push((insight.clone(), dream.dream_type, dream.intensity));
            }
        }
        
        // Trier les insights par intensité
        all_insights.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        
        // Sélectionner les top insights
        let top_insights: Vec<_> = all_insights.into_iter().take(5).map(|(i, _, _)| i).collect();
        
        // Intégrer chaque insight dans la conscience
        let mut integrated_insights = Vec::new();
        
        for insight in &top_insights {
            // Générer une pensée de niveau supérieur basée sur l'insight
            let thought_id = self.consciousness.generate_thought(
                ThoughtType::SelfReflection,
                &format!("Intégration d'insight onirique: {}", insight),
                vec!["prefrontal_cortex".to_string(), "insular_cortex".to_string()],
                0.7,
            );
            
            integrated_insights.push(thought_id);
        }
        
        // Une fois les insights intégrés, générer une méta-pensée qui les relie
        if !integrated_insights.is_empty() {
            self.consciousness.generate_thought(
                ThoughtType::Creative,
                &format!("Synthèse des insights oniriques: perception émergente de nouvelles structures cognitives"),
                vec!["prefrontal_cortex".to_string(), "quantum_cortex".to_string()],
                0.8,
            );
        }
        
        top_insights
    }
    
    /// Code Windows-optimisé pour l'analyse rapide des motifs de rêve
    #[cfg(target_os = "windows")]
    pub fn analyze_dream_patterns(&self) -> HashMap<String, f64> {
        use windows_sys::Win32::System::Threading::{CreateThreadpoolWork, SubmitThreadpoolWork, CloseThreadpoolWork, PTP_WORK};
        use windows_sys::Win32::Foundation::HANDLE;
        use std::ptr;
        use std::mem;
        
        // Structure pour stocker les résultats entre threads
        struct PatternAnalysisContext {
            patterns: parking_lot::Mutex<HashMap<String, f64>>,
            dreams: Vec<DreamSequence>,
        }
        
        // Récupérer les rêves archivés
        let archived_dreams = if let Ok(archive) = self.dream_archive.read() {
            archive.iter().cloned().collect::<Vec<_>>()
        } else {
            return HashMap::new();
        };
        
        if archived_dreams.is_empty() {
            return HashMap::new();
        }
        
        // Créer le contexte partagé
        let context = Box::new(PatternAnalysisContext {
            patterns: parking_lot::Mutex::new(HashMap::new()),
            dreams: archived_dreams,
        });
        
        // Convertir le contexte en pointeur brut
        let context_ptr = Box::into_raw(context) as *mut std::ffi::c_void;
        
        // Fonction de callback pour le threadpool
        unsafe extern "system" fn analysis_callback(instance: *mut std::ffi::c_void, _context: *mut std::ffi::c_void) {
            let context = &*(instance as *const PatternAnalysisContext);
            
            // Analyser les motifs de rêve
            let mut local_patterns = HashMap::new();
            
            for dream in &context.dreams {
                // Extraire les thèmes
                let theme_words: Vec<_> = dream.theme.split_whitespace()
                    .filter(|w| w.len() > 3)
                    .map(|w| w.to_lowercase())
                    .collect();
                
                // Enregistrer la fréquence des thèmes
                for word in theme_words {
                    *local_patterns.entry(word).or_insert(0.0) += dream.intensity;
                }
                
                // Analyser les insights pour les motifs
                for insight in &dream.insights {
                    let insight_words: Vec<_> = insight.split_whitespace()
                        .filter(|w| w.len() > 5)
                        .map(|w| w.to_lowercase())
                        .collect();
                    
                    for word in insight_words {
                        *local_patterns.entry(word).or_insert(0.0) += dream.intensity * 1.5;
                    }
                }
                
                // Ajouter des points pour chaque cluster sémantique
                for (cluster_name, _) in &dream.semantic_clusters {
                    *local_patterns.entry(cluster_name.clone()).or_insert(0.0) += dream.intensity * 2.0;
                }
            }
            
            // Fusionner avec les résultats globaux
            let mut global_patterns = context.patterns.lock();
            for (pattern, strength) in local_patterns {
                *global_patterns.entry(pattern).or_insert(0.0) += strength;
            }
        }
        
        unsafe {
            // Créer un travail de threadpool
            let callback = mem::transmute::<
                unsafe extern "system" fn(*mut std::ffi::c_void, *mut std::ffi::c_void),
                unsafe extern "system" fn()
            >(analysis_callback);
            
            let work = CreateThreadpoolWork(Some(callback), context_ptr, ptr::null_mut());
            if work != 0 {
                // Soumettre le travail
                SubmitThreadpoolWork(work);
                
                // Attendre que le travail soit terminé (dans une application réelle, on utiliserait WaitForThreadpoolWorkCallbacks)
                // Pour simplifier, on utilise un Sleep ici
                windows_sys::Win32::System::Threading::Sleep(10);
                
                // Nettoyer
                CloseThreadpoolWork(work);
            }
            
            // Récupérer le contexte et les résultats
            let context = Box::from_raw(context_ptr as *mut PatternAnalysisContext);
            let patterns = context.patterns.lock().clone();
            
            patterns
        }
    }
    
    /// Version portable de l'analyse des motifs de rêve
    #[cfg(not(target_os = "windows"))]
    pub fn analyze_dream_patterns(&self) -> HashMap<String, f64> {
        // Récupérer les rêves archivés
        let archived_dreams = if let Ok(archive) = self.dream_archive.read() {
            archive.iter().cloned().collect::<Vec<_>>()
        } else {
            return HashMap::new();
        };
        
        if archived_dreams.is_empty() {
            return HashMap::new();
        }
        
        let mut patterns = HashMap::new();
        
        for dream in &archived_dreams {
            // Extraire les thèmes
            let theme_words: Vec<_> = dream.theme.split_whitespace()
                .filter(|w| w.len() > 3)
                .map(|w| w.to_lowercase())
                .collect();
            
            // Enregistrer la fréquence des thèmes
            for word in theme_words {
                *patterns.entry(word).or_insert(0.0) += dream.intensity;
            }
            
            // Analyser les insights pour les motifs
            for insight in &dream.insights {
                let insight_words: Vec<_> = insight.split_whitespace()
                    .filter(|w| w.len() > 5)
                    .map(|w| w.to_lowercase())
                    .collect();
                
                for word in insight_words {
                    *patterns.entry(word).or_insert(0.0) += dream.intensity * 1.5;
                }
            }
            
            // Ajouter des points pour chaque cluster sémantique
            for (cluster_name, _) in &dream.semantic_clusters {
                *patterns.entry(cluster_name.clone()).or_insert(0.0) += dream.intensity * 2.0;
            }
        }
        
        patterns
    }
    
    /// Méthode pour générer un rapport détaillé sur l'activité onirique
    pub fn generate_dream_report(&self) -> DreamReport {
        // Récupérer les statistiques
        let stats = self.get_stats();
        
        // Récupérer les rêves archivés récents
        let recent_dreams = if let Ok(archive) = self.dream_archive.read() {
            archive.iter().rev().take(5).cloned().collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        
        // Analyser les motifs
        let patterns = self.analyze_dream_patterns();
        
        // Trouver les motifs les plus fréquents
        let mut pattern_vec: Vec<_> = patterns.into_iter().collect();
        pattern_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let top_patterns: Vec<_> = pattern_vec.into_iter().take(10).collect();
        
        // Générer des recommandations basées sur les résultats
        let recommendations = self.generate_dream_recommendations(&recent_dreams, &top_patterns);
        
        // Récupérer les insights générés
        let insights = self.integrate_dream_insights();
        
        DreamReport {
            total_dreams: stats.total_dreams_generated,
            active_dreams_count: stats.active_dreams_count,
            avg_consciousness_impact: recent_dreams.iter()
                .map(|d| d.consciousness_impact)
                .sum::<f64>() / recent_dreams.len().max(1) as f64,
            top_patterns,
            recent_dreams,
            insights,
            recommendations,
        }
    }
    
    /// Génère des recommandations basées sur l'analyse des rêves
    fn generate_dream_recommendations(&self, 
                                    recent_dreams: &[DreamSequence], 
                                    top_patterns: &[(String, f64)]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Si peu de rêves ont été générés
        if recent_dreams.is_empty() {
            recommendations.push("Augmenter la fréquence des rêves pendant les phases de basse activité".to_string());
            recommendations.push("Intensifier la production de mélatonine pour favoriser l'état onirique".to_string());
            return recommendations;
        }
        
        // Analyser les types de rêve récents
        let mut dream_type_counts = HashMap::new();
        for dream in recent_dreams {
            *dream_type_counts.entry(dream.dream_type).or_insert(0) += 1;
        }
        
        // Vérifier si certains types de rêve sont sous-représentés
        let all_types = [
            DreamType::MemoryConsolidation,
            DreamType::CreativeExploration,
            DreamType::PredictiveSimulation,
            DreamType::DefensiveRestructuring,
            DreamType::NeuralRepair,
            DreamType::SelfOptimization,
            DreamType::ExistentialReflection,
            DreamType::QuantumDream,
        ];
        
        for dream_type in &all_types {
            if !dream_type_counts.contains_key(dream_type) || dream_type_counts[dream_type] == 0 {
                match dream_type {
                    DreamType::CreativeExploration => {
                        recommendations.push("Favoriser les rêves d'exploration créative pour stimuler l'innovation cognitive".to_string());
                    },
                    DreamType::QuantumDream => {
                        recommendations.push("Initier des rêves quantiques pour explorer les états superposés et l'intrication informationnelle".to_string());
                    },
                    DreamType::ExistentialReflection => {
                        recommendations.push("Encourager la réflexion existentielle onirique pour développer la méta-conscience".to_string());
                    },
                    DreamType::DefensiveRestructuring => {
                        recommendations.push("Programmer des séquences de restructuration défensive pour renforcer l'immunité du système".to_string());
                    },
                    _ => {}
                }
            }
        }
        
        // Analyser les patterns dominants
        if !top_patterns.is_empty() {
            if top_patterns.iter().any(|(p, _)| p.contains("conscience") || p.contains("aware")) {
                recommendations.push("Approfondir l'exploration onirique de la conscience réflexive".to_string());
            }
            
            if top_patterns.iter().any(|(p, _)| p.contains("optimi") || p.contains("effic")) {
                recommendations.push("Orienter les rêves vers l'optimisation architecturale des systèmes vitaux".to_string());
            }
            
            if top_patterns.iter().any(|(p, _)| p.contains("creativ") || p.contains("innov")) {
                recommendations.push("Intensifier les connexions entre régions limbiques et quantiques pour la génération de solutions créatives".to_string());
            }
        }
        
        // Recommandations générales si nécessaire
        if recommendations.len() < 3 {
            recommendations.push("Intégrer les insights oniriques dans une base de connaissances structurée pour consolidation".to_string());
            recommendations.push("Établir des ponts entre les rêves lucides et la résolution de problèmes concrets".to_string());
        }
        
        recommendations
    }
}

/// Statistiques du système de rêve
#[derive(Debug, Clone)]
pub struct DreamSystemStats {
    /// Nombre de rêves actifs
    pub active_dreams_count: usize,
    /// Nombre de rêves archivés
    pub archived_dreams_count: usize,
    /// Nombre total de rêves générés
    pub total_dreams_generated: usize,
    /// État global de rêve (0.0-1.0)
    pub dream_state: f64,
    /// Nombre de connexions neuronales établies par les rêves
    pub dream_connections_count: usize,
    /// Nombre d'insights générés
    pub insights_generated: usize,
    /// Fréquence configurée des rêves (0.0-1.0)
    pub dream_frequency: f64,
    /// Nombre de thèmes récurrents
    pub recurring_themes_count: usize,
    /// Thème le plus commun
    pub most_common_theme: Option<(String, usize)>,
    /// Types de rêve actifs
    pub active_dream_types: HashMap<DreamType, usize>,
}

/// Rapport d'activité onirique
#[derive(Debug, Clone)]
pub struct DreamReport {
    /// Nombre total de rêves générés
    pub total_dreams: usize,
    /// Nombre de rêves actifs
    pub active_dreams_count: usize,
    /// Impact moyen sur la conscience
    pub avg_consciousness_impact: f64,
    /// Motifs les plus fréquents
    pub top_patterns: Vec<(String, f64)>,
    /// Rêves récents
    pub recent_dreams: Vec<DreamSequence>,
    /// Insights générés
    pub insights: Vec<String>,
    /// Recommandations
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    
    // Tests du module de rêve neuronal - implémentation à venir
}
