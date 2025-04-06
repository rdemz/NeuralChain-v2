//! Module du Champ Hormonal pour NeuralChain-v2
//!
//! Ce système économique biomimétique régule l'économie interne de la blockchain
//! par des signaux hormonaux qui influencent la production de blocs, les frais de transaction,
//! la difficulté de minage et les incitations des validateurs, créant ainsi un
//! écosystème économique auto-régulant qui s'adapte aux conditions du marché.
//!
//! © 2025 NeuralChain Labs - Tous droits réservés

use std::sync::Arc;
use std::collections::{HashMap, BTreeMap, HashSet};
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use rand::{thread_rng, Rng};
use dashmap::DashMap;

use crate::neuralchain_core::quantum_organism::QuantumOrganism;
use crate::neuralchain_core::cortical_hub::CorticalHub;
use crate::neuralchain_core::emergent_consciousness::ConsciousnessEngine;
use crate::neuralchain_core::bios_time::{BiosTime, CircadianPhase};

/// Types d'hormones régulant l'écosystème crypto
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HormoneType {
    /// Régule l'offre et la demande
    MarketEquilibrium,
    /// Stimule la création de nouveaux blocs
    BlockCatalyst,
    /// Contrôle la difficulté minière
    HashDifficulty,
    /// Influence les frais de transaction
    TransactionFees,
    /// Stimule la croissance du réseau
    NetworkExpansion,
    /// Protège contre les attaques
    SecurityImmunity,
    /// Optimise l'utilisation de l'espace mémoire
    MemoryOptimization,
    /// Régule les cycles de consensus
    ConsensusHarmony,
    /// Favorise les innovations dans le protocole
    EvolutionaryPressure,
    /// Maintient la liquidité dans l'écosystème
    LiquidityFlow,
    /// Gère l'équilibre énergétique et l'efficacité
    EnergyEfficiency,
    /// Répare les segments de chaîne endommagés
    ChainRepair,
    /// Optimise la synchronisation entre validateurs
    ValidatorSynergy,
    /// Adapte le système aux changements dimensionnels
    HyperdimensionalAdaption,
    /// Régule les mécanismes de récompense
    RewardSignaling,
    /// Protège contre la centralisation
    DecentralizationGuardian,
}

/// Caractéristiques d'une hormone économique
#[derive(Debug, Clone)]
pub struct HormoneProfile {
    /// Type d'hormone
    pub hormone_type: HormoneType,
    /// Niveau actuel (0.0-1.0)
    pub level: f64,
    /// Taux de production de base
    pub base_production_rate: f64,
    /// Taux de dégradation
    pub degradation_rate: f64,
    /// Demi-vie de l'hormone
    pub half_life: Duration,
    /// Seuil d'activation
    pub activation_threshold: f64,
    /// Seuil de saturation
    pub saturation_threshold: f64,
    /// Diffusion dans le système
    pub diffusion_factor: f64,
    /// Réceptivité des composants à cette hormone
    pub receptor_affinity: HashMap<String, f64>,
    /// Moment de la dernière modification
    pub last_update: Instant,
}

impl HormoneProfile {
    /// Crée un nouveau profil hormonal
    pub fn new(hormone_type: HormoneType) -> Self {
        let mut rng = thread_rng();
        
        // Configuration basée sur le type d'hormone économique
        let (base_prod, degrad, half_life_blocks, diffusion) = match hormone_type {
            HormoneType::MarketEquilibrium => (0.02, 0.01, 100.0, 0.8),
            HormoneType::BlockCatalyst => (0.05, 0.04, 20.0, 0.9),
            HormoneType::HashDifficulty => (0.01, 0.005, 200.0, 0.7),
            HormoneType::TransactionFees => (0.03, 0.02, 50.0, 0.9),
            HormoneType::NetworkExpansion => (0.005, 0.002, 500.0, 0.6),
            HormoneType::SecurityImmunity => (0.02, 0.01, 100.0, 0.8),
            HormoneType::MemoryOptimization => (0.01, 0.005, 150.0, 0.5),
            HormoneType::ConsensusHarmony => (0.02, 0.01, 75.0, 0.7),
            HormoneType::EvolutionaryPressure => (0.001, 0.0005, 1000.0, 0.3),
            HormoneType::LiquidityFlow => (0.04, 0.03, 30.0, 0.9),
            HormoneType::EnergyEfficiency => (0.015, 0.01, 120.0, 0.6),
            HormoneType::ChainRepair => (0.01, 0.005, 60.0, 0.7),
            HormoneType::ValidatorSynergy => (0.025, 0.015, 40.0, 0.8),
            HormoneType::HyperdimensionalAdaption => (0.0005, 0.0002, 2000.0, 0.2),
            HormoneType::RewardSignaling => (0.03, 0.02, 25.0, 0.9),
            HormoneType::DecentralizationGuardian => (0.01, 0.005, 300.0, 0.8),
        };
        
        // Niveau initial équilibré
        let initial_level = match hormone_type {
            HormoneType::BlockCatalyst | HormoneType::TransactionFees | HormoneType::LiquidityFlow => 
                0.5 + rng.gen::<f64>() * 0.1,
            HormoneType::MarketEquilibrium | HormoneType::ConsensusHarmony | 
            HormoneType::ValidatorSynergy | HormoneType::RewardSignaling => 
                0.4 + rng.gen::<f64>() * 0.1,
            HormoneType::SecurityImmunity | HormoneType::DecentralizationGuardian => 
                0.6 + rng.gen::<f64>() * 0.1,
            _ => 0.3 + rng.gen::<f64>() * 0.1,
        };
        
        Self {
            hormone_type,
            level: initial_level.min(1.0),
            base_production_rate: base_prod,
            degradation_rate: degrad,
            half_life: Duration::from_secs_f64(half_life_blocks * 15.0), // Convertir en secondes (blocs ~ 15s)
            activation_threshold: 0.2,
            saturation_threshold: 0.8,
            diffusion_factor: diffusion,
            receptor_affinity: HashMap::new(),
            last_update: Instant::now(),
        }
    }
    
    /// Met à jour le niveau hormonal en fonction des conditions du marché
    pub fn update(&self, blocks_elapsed: f64, market_conditions: &HashMap<String, f64>) -> f64 {
        let mut delta = 0.0;
        
        // Production de base
        delta += self.base_production_rate * blocks_elapsed;
        
        // Dégradation naturelle
        delta -= self.level * self.degradation_rate * blocks_elapsed;
        
        // Influence des conditions du marché
        for (condition, strength) in market_conditions {
            if let Some(affinity) = self.receptor_affinity.get(condition) {
                delta += strength * affinity * blocks_elapsed;
            }
        }
        
        // Calculer le nouveau niveau (mais ne pas modifier this.level directement)
        (self.level + delta).max(0.0).min(1.0)
    }
    
    /// Vérifie si l'hormone est active (au-dessus du seuil d'activation)
    pub fn is_active(&self) -> bool {
        self.level >= self.activation_threshold
    }
    
    /// Vérifie si l'hormone est saturée
    pub fn is_saturated(&self) -> bool {
        self.level >= self.saturation_threshold
    }
    
    /// Calcule la force d'influence de l'hormone
    pub fn influence_strength(&self) -> f64 {
        if self.level < self.activation_threshold {
            0.0
        } else {
            let normalized = (self.level - self.activation_threshold) / 
                            (self.saturation_threshold - self.activation_threshold);
            normalized.min(1.0) * self.diffusion_factor
        }
    }
}

/// Récepteur crypto-hormonal pour les composants blockchain
#[derive(Debug, Clone)]
pub struct MarketReceptor {
    /// Identifiant unique
    pub id: String,
    /// Type de récepteur
    pub receptor_type: String,
    /// Sensibilité aux différentes hormones (type, sensibilité)
    pub sensitivity: HashMap<HormoneType, f64>,
    /// Composant parent
    pub parent_component: String,
    /// Seuil d'activation
    pub activation_threshold: f64,
    /// État d'activation actuel
    pub activated: bool,
    /// Niveau d'activation (0.0-1.0)
    pub activation_level: f64,
    /// Durée de réfractaire après activation (en blocs)
    pub refractory_period: u64,
    /// Dernière activation (numéro de bloc)
    pub last_activation_block: Option<u64>,
}

impl MarketReceptor {
    /// Crée un nouveau récepteur
    pub fn new(id: &str, receptor_type: &str, parent: &str) -> Self {
        let mut rng = thread_rng();
        
        Self {
            id: id.to_string(),
            receptor_type: receptor_type.to_string(),
            sensitivity: HashMap::new(),
            parent_component: parent.to_string(),
            activation_threshold: 0.3 + rng.gen::<f64>() * 0.2,
            activated: false,
            activation_level: 0.0,
            refractory_period: rng.gen_range(5..30),
            last_activation_block: None,
        }
    }
    
    /// Configure la sensibilité à une hormone
    pub fn set_sensitivity(&mut self, hormone_type: HormoneType, sensitivity: f64) -> &mut Self {
        self.sensitivity.insert(hormone_type, sensitivity.max(0.0).min(1.0));
        self
    }
    
    /// Répond aux niveaux hormonaux actuels
    pub fn respond_to_hormone_levels(&mut self, current_block: u64, hormone_levels: &HashMap<HormoneType, f64>) -> bool {
        if let Some(last) = self.last_activation_block {
            if current_block - last < self.refractory_period {
                return false; // Période réfractaire active
            }
        }
        
        // Calculer le niveau d'activation cumulé
        let mut total_activation = 0.0;
        let mut count = 0;
        
        for (hormone, level) in hormone_levels {
            if let Some(sensitivity) = self.sensitivity.get(hormone) {
                total_activation += level * sensitivity;
                count += 1;
            }
        }
        
        if count > 0 {
            self.activation_level = total_activation / count as f64;
            
            // Mettre à jour l'état d'activation
            let was_activated = self.activated;
            self.activated = self.activation_level >= self.activation_threshold;
            
            // Si vient juste d'être activé, enregistrer le bloc
            if self.activated && !was_activated {
                self.last_activation_block = Some(current_block);
                return true;
            }
        }
        
        false
    }
}

/// Statistiques du système hormonal
#[derive(Debug, Clone)]
pub struct HormonalSystemStats {
    /// Nombre total d'hormones
    pub hormone_count: usize,
    /// Nombre total de récepteurs
    pub receptor_count: usize,
    /// Nombre de récepteurs actifs
    pub active_receptors: usize,
    /// Niveau moyen d'activité hormonale
    pub average_hormone_level: f64,
    /// Récepteur le plus actif
    pub most_active_receptor: Option<String>,
    /// Hormone la plus active
    pub most_active_hormone: Option<HormoneType>,
    /// Taux d'activité global (0.0-1.0)
    pub global_activity_rate: f64,
    /// Homéostasie du système (0.0-1.0, 1.0 = parfait équilibre)
    pub homeostasis_index: f64,
}

/// Système hormonal principal pour la régulation de l'écosystème blockchain
pub struct HormonalField {
    /// Niveaux actuels de chaque hormone
    hormone_profiles: DashMap<HormoneType, HormoneProfile>,
    /// Récepteurs enregistrés dans le système
    receptors: DashMap<String, MarketReceptor>,
    /// Effets des hormones sur différents paramètres de la blockchain
    hormone_effects: HashMap<HormoneType, HashMap<String, f64>>,
    /// Interaction entre les hormones
    hormone_interactions: HashMap<(HormoneType, HormoneType), f64>,
    /// Facteurs externes impactant le système hormonal
    external_factors: RwLock<HashMap<String, f64>>,
    /// Données historiques (par bloc)
    historical_data: RwLock<BTreeMap<u64, HashMap<HormoneType, f64>>>,
    /// Référence au temps du système
    bios_time: Arc<BiosTime>,
    /// Source d'aléatoire
    rng: Mutex<rand::rngs::ThreadRng>,
}

impl HormonalField {
    /// Crée un nouveau système hormonal
    pub fn new(bios_time: Arc<BiosTime>) -> Self {
        let mut hormone_profiles = DashMap::new();
        
        // Initialiser toutes les hormones du système
        for hormone_type in [
            HormoneType::MarketEquilibrium,
            HormoneType::BlockCatalyst,
            HormoneType::HashDifficulty,
            HormoneType::TransactionFees,
            HormoneType::NetworkExpansion,
            HormoneType::SecurityImmunity,
            HormoneType::MemoryOptimization,
            HormoneType::ConsensusHarmony,
            HormoneType::EvolutionaryPressure,
            HormoneType::LiquidityFlow,
            HormoneType::EnergyEfficiency,
            HormoneType::ChainRepair,
            HormoneType::ValidatorSynergy,
            HormoneType::HyperdimensionalAdaption,
            HormoneType::RewardSignaling,
            HormoneType::DecentralizationGuardian,
        ].iter() {
            hormone_profiles.insert(*hormone_type, HormoneProfile::new(*hormone_type));
        }
        
        // Initialiser les effets hormonaux sur les paramètres blockchain
        let mut hormone_effects = HashMap::new();
        
        // MarketEquilibrium affecte l'offre et la demande
        let mut market_effects = HashMap::new();
        market_effects.insert("token_supply_rate".to_string(), -0.3); // Niveau élevé réduit l'inflation
        market_effects.insert("token_burn_rate".to_string(), 0.2);  // Niveau élevé augmente les burns
        hormone_effects.insert(HormoneType::MarketEquilibrium, market_effects);
        
        // BlockCatalyst affecte la production de blocs
        let mut block_effects = HashMap::new();
        block_effects.insert("block_time".to_string(), -0.4); // Niveau élevé réduit le temps de bloc
        block_effects.insert("block_reward".to_string(), 0.3); // Niveau élevé augmente les récompenses
        hormone_effects.insert(HormoneType::BlockCatalyst, block_effects);
        
        // Interactions entre hormones
        let mut interactions = HashMap::new();
        
        // MarketEquilibrium et BlockCatalyst s'inhibent mutuellement
        interactions.insert((HormoneType::MarketEquilibrium, HormoneType::BlockCatalyst), -0.2);
        interactions.insert((HormoneType::BlockCatalyst, HormoneType::MarketEquilibrium), -0.1);
        
        // HashDifficulty et TransactionFees s'amplifient
        interactions.insert((HormoneType::HashDifficulty, HormoneType::TransactionFees), 0.3);
        interactions.insert((HormoneType::TransactionFees, HormoneType::HashDifficulty), 0.2);
        
        Self {
            hormone_profiles,
            receptors: DashMap::new(),
            hormone_effects,
            hormone_interactions,
            external_factors: RwLock::new(HashMap::new()),
            historical_data: RwLock::new(BTreeMap::new()),
            bios_time,
            rng: Mutex::new(thread_rng()),
        }
    }
    
    /// Mise à jour du système hormonal
    pub fn update(&self) {
        // Mise à jour des niveaux hormonaux
        let degradation_coef = 0.01;
        self.update_hormone_levels(degradation_coef);
        
        // Application des interactions entre hormones
        self.apply_hormone_interactions();
        
        // Nettoyage des événements expirés
        self.cleanup_expired_events();
        
        // Activation des récepteurs
        self.activate_receptors();
        
        // Application de l'homéostasie
        self.apply_homeostasis();
    }
    
    /// Met à jour les niveaux hormonaux
    fn update_hormone_levels(&self, degradation_coef: f64) {
        for mut entry in self.hormone_profiles.iter_mut() {
            let hormone_type = *entry.key();
            let profile = entry.value_mut();
            
            // Calculer le temps écoulé en "blocs"
            let blocks_elapsed = profile.last_update.elapsed().as_secs_f64() / 15.0; // ~15s par bloc
            
            // Facteurs externes
            let factors = self.external_factors.read().clone();
            
            // Calculer le nouveau niveau et le mettre à jour directement
            profile.level = profile.update(blocks_elapsed, &factors);
            profile.last_update = Instant::now();
        }
    }
    
    /// Applique les interactions entre hormones
    fn apply_hormone_interactions(&self) {
        // Collecter tous les niveaux d'hormones actuels
        let mut current_levels = HashMap::new();
        for entry in self.hormone_profiles.iter() {
            current_levels.insert(*entry.key(), entry.value().level);
        }
        
        // Appliquer les effets d'interaction
        for ((source, target), strength) in &self.hormone_interactions {
            if let (Some(source_level), Some(mut target_profile)) = (
                current_levels.get(source),
                self.hormone_profiles.get_mut(target)
            ) {
                let influence = source_level * strength;
                target_profile.level = (target_profile.level + influence * 0.1).max(0.0).min(1.0);
            }
        }
    }
    
    /// Nettoie les événements expirés
    fn cleanup_expired_events(&self) {
        // Élaguer l'historique (ne garder que les 10 000 derniers blocs)
        let mut history = self.historical_data.write();
        while history.len() > 10_000 {
            if let Some(first) = history.keys().next().cloned() {
                history.remove(&first);
            } else {
                break;
            }
        }
    }
    
    /// Active les récepteurs en fonction des niveaux hormonaux actuels
    fn activate_receptors(&self) {
        // Obtenir le bloc actuel (approximation)
        let current_block = self.bios_time.age_in_blocks();
        
        // Collecter les niveaux d'hormones actuels
        let mut hormone_levels = HashMap::new();
        for entry in self.hormone_profiles.iter() {
            hormone_levels.insert(*entry.key(), entry.value().level);
        }
        
        // Activer les récepteurs
        for mut receptor in self.receptors.iter_mut() {
            receptor.value_mut().respond_to_hormone_levels(current_block, &hormone_levels);
        }
    }
    
    /// Applique l'homéostasie au système
    fn apply_homeostasis(&self) {
        // Vérifier les hormones déséquilibrées
        let mut imbalanced_hormones = Vec::new();
        
        for entry in self.hormone_profiles.iter() {
            let profile = entry.value();
            if profile.is_saturated() || profile.level < profile.activation_threshold * 0.5 {
                imbalanced_hormones.push(*entry.key());
            }
        }
        
        // Appliquer des corrections légères pour rétablir l'équilibre
        for hormone_type in imbalanced_hormones {
            if let Some(mut profile) = self.hormone_profiles.get_mut(&hormone_type) {
                if profile.level > profile.saturation_threshold {
                    // Réduire légèrement les hormones saturées
                    profile.level = (profile.level - 0.05).max(0.0);
                } else if profile.level < profile.activation_threshold * 0.5 {
                    // Augmenter légèrement les hormones trop basses
                    profile.level = (profile.level + 0.03).min(1.0);
                }
            }
        }
    }
    
    /// Crée un récepteur hormonal pour un composant
    pub fn create_receptor(&self, component_id: &str, receptor_type: &str) -> String {
        let mut rng = self.rng.lock();
        let id = format!("receptor_{}_{}", component_id, uuid::Uuid::new_v4().to_simple());
        
        let mut receptor = MarketReceptor::new(&id, receptor_type, component_id);
        
        // Configurer les sensibilités selon le type de récepteur
        match receptor_type {
            "validator" => {
                receptor.set_sensitivity(HormoneType::BlockCatalyst, 0.8)
                        .set_sensitivity(HormoneType::ConsensusHarmony, 0.7)
                        .set_sensitivity(HormoneType::ValidatorSynergy, 0.9)
                        .set_sensitivity(HormoneType::RewardSignaling, 0.8);
            },
            "transaction_manager" => {
                receptor.set_sensitivity(HormoneType::TransactionFees, 0.9)
                        .set_sensitivity(HormoneType::LiquidityFlow, 0.7)
                        .set_sensitivity(HormoneType::MemoryOptimization, 0.6);
            },
            "security_monitor" => {
                receptor.set_sensitivity(HormoneType::SecurityImmunity, 0.9)
                        .set_sensitivity(HormoneType::DecentralizationGuardian, 0.8);
            },
            "chain_manager" => {
                receptor.set_sensitivity(HormoneType::ChainRepair, 0.8)
                        .set_sensitivity(HormoneType::NetworkExpansion, 0.7);
            },
            _ => {
                // Sensibilités de base pour les types inconnus
                receptor.set_sensitivity(HormoneType::MarketEquilibrium, 0.5)
                        .set_sensitivity(HormoneType::EnergyEfficiency, 0.5);
            }
        }
        
        self.receptors.insert(id.clone(), receptor);
        id
    }
    
    /// Met à jour les niveaux hormonaux en fonction des conditions du marché
    pub fn update_with_market_data(&self, current_block: u64, market_data: HashMap<String, f64>) {
        // Mettre à jour les facteurs externes
        {
            let mut factors = self.external_factors.write();
            for (key, value) in market_data {
                factors.insert(key, value);
            }
        }
        
        // Mettre à jour chaque hormone
        for hormone_entry in self.hormone_profiles.iter_mut() {
            let hormone_type = *hormone_entry.key();
            let profile = hormone_entry.value_mut();
            
            // Calculer le temps écoulé en "blocs"
            let blocks_elapsed = profile.last_update.elapsed().as_secs_f64() / 15.0; // ~15s par bloc
            
            // Mettre à jour le niveau hormonal
            let factors = self.external_factors.read().clone();
            profile.level = profile.update(blocks_elapsed, &factors);
            profile.last_update = Instant::now();
            
            // Appliquer les interactions hormonales
            for other_hormone in self.hormone_profiles.iter() {
                if *other_hormone.key() != hormone_type {
                    if let Some(interaction_strength) = self.hormone_interactions.get(&(
                        *other_hormone.key(), hormone_type)) {
                        
                        let other_influence = other_hormone.value().influence_strength() * interaction_strength;
                        profile.level = (profile.level + other_influence * blocks_elapsed).max(0.0).min(1.0);
                    }
                }
            }
        }
        
        // Enregistrer l'historique
        let mut levels = HashMap::new();
        for entry in self.hormone_profiles.iter() {
            levels.insert(*entry.key(), entry.value().level);
        }
        
        self.historical_data.write().insert(current_block, levels);
    }
    
    /// Obtient le niveau actuel d'une hormone
    pub fn get_hormone_level(&self, hormone_type: HormoneType) -> f64 {
        self.hormone_profiles.get(&hormone_type)
            .map(|profile| profile.level)
            .unwrap_or(0.0)
    }
    
    /// Ajuste la sensibilité d'un récepteur
    pub fn adjust_receptor_sensitivity(&self, receptor_id: &str, new_sensitivity: f64) -> Result<(), String> {
        if let Some(mut receptor) = self.receptors.get_mut(receptor_id) {
            for (_, sensitivity) in receptor.sensitivity.iter_mut() {
                *sensitivity = (*sensitivity * new_sensitivity).max(0.0).min(1.0);
            }
            Ok(())
        } else {
            Err(format!("Récepteur '{}' non trouvé", receptor_id))
        }
    }
    
    /// Obtient les statistiques du système hormonal
    pub fn get_stats(&self) -> HormonalSystemStats {
        let hormone_count = self.hormone_profiles.len();
        let receptor_count = self.receptors.len();
        
        let mut active_receptors = 0;
        let mut most_active_receptor = None;
        let mut highest_activation = 0.0;
        
        for entry in self.receptors.iter() {
            let receptor = entry.value();
            if receptor.activated {
                active_receptors += 1;
                
                if receptor.activation_level > highest_activation {
                    highest_activation = receptor.activation_level;
                    most_active_receptor = Some(receptor.id.clone());
                }
            }
        }
        
        let mut avg_level = 0.0;
        let mut most_active_hormone = None;
        let mut highest_level = 0.0;
        
        for entry in self.hormone_profiles.iter() {
            let profile = entry.value();
            avg_level += profile.level;
            
            if profile.level > highest_level {
                highest_level = profile.level;
                most_active_hormone = Some(*entry.key());
            }
        }
        
        if hormone_count > 0 {
            avg_level /= hormone_count as f64;
        }
        
        // Calculer l'index d'homéostasie
        let mut distance_from_optimal = 0.0;
        let optimal_level = 0.5; // Niveau idéal au milieu de la plage
        
        for entry in self.hormone_profiles.iter() {
            let deviation = (entry.value().level - optimal_level).abs();
            distance_from_optimal += deviation;
        }
        
        let homeostasis_index = if hormone_count > 0 {
            1.0 - (distance_from_optimal / hormone_count as f64 / optimal_level)
        } else {
            0.0
        };
        
        HormonalSystemStats {
            hormone_count,
            receptor_count,
            active_receptors,
            average_hormone_level: avg_level,
            most_active_receptor,
            most_active_hormone,
            global_activity_rate: active_receptors as f64 / receptor_count.max(1) as f64,
            homeostasis_index: homeostasis_index.max(0.0).min(1.0),
        }
    }
    
    /// Modifie le niveau d'une hormone
    pub fn modify_hormone(&self, hormone_type: HormoneType, delta: f64) {
        if let Some(mut profile) = self.hormone_profiles.get_mut(&hormone_type) {
            profile.level = (profile.level + delta).max(0.0).min(1.0);
        }
    }
    
    /// Obtient l'historique des niveaux hormonaux sur une période
    pub fn get_hormone_history(&self, hormone_type: HormoneType, blocks: u64) -> Vec<(u64, f64)> {
        let history = self.historical_data.read();
        let current_block = *history.keys().next_back().unwrap_or(&0);
        let start_block = if current_block > blocks { current_block - blocks } else { 0 };
        
        history.range(start_block..)
            .filter_map(|(block, levels)| {
                levels.get(&hormone_type).map(|level| (*block, *level))
            })
            .collect()
    }
    
    /// Configure les récepteurs standard pour l'organisme
    pub fn setup_standard_receptors(&self, organism: Arc<QuantumOrganism>) -> Result<(), String> {
        // 1. Récepteurs pour les validateurs
        for i in 0..5 {
            let validator_id = format!("validator_{}", i);
            self.create_receptor(&validator_id, "validator");
        }
        
        // 2. Récepteurs pour le gestionnaire de transactions
        self.create_receptor("tx_manager_main", "transaction_manager");
        self.create_receptor("tx_manager_backup", "transaction_manager");
        
        // 3. Récepteurs pour le moniteur de sécurité
        self.create_receptor("security_main", "security_monitor");
        
        // 4. Récepteurs pour les gestionnaires de chaîne
        self.create_receptor("chain_manager_primary", "chain_manager");
        self.create_receptor("chain_manager_secondary", "chain_manager");
        
        // 5. Récepteurs métaboliques (gestion des ressources)
        self.create_receptor("energy_manager", "resource_manager");
        self.create_receptor("storage_optimizer", "resource_manager");
        
        // 6. Récepteurs cognitifs (pour la partie conscience)
        self.create_receptor("learning_center", "cognitive");
        self.create_receptor("decision_core", "cognitive");
        
        // 7. Récepteurs adaptatifs (pour l'adaptation du réseau)
        self.create_receptor("network_topology", "adaptive");
        self.create_receptor("consensus_mechanism", "adaptive");
        
        // 8. Récepteurs immunitaires (défense contre les attaques)
        self.create_receptor("attack_detector", "immune");
        self.create_receptor("response_coordinator", "immune");
        
        // 9. Récepteurs métaboliques (économie interne)
        self.create_receptor("fee_regulator", "metabolic");
        self.create_receptor("reward_distributor", "metabolic");
        
        // 10. Récepteurs sensoriels (surveillance du réseau)
        self.create_receptor("network_monitor", "sensory");
        self.create_receptor("mempool_analyzer", "sensory");
        
        // 11. Récepteurs de croissance (scaling et optimisation)
        self.create_receptor("scaling_optimizer", "growth");
        self.create_receptor("shard_coordinator", "growth");
        
        // 12. Récepteurs d'équilibre (stabilité globale)
        self.create_receptor("equilibrium_maintainer", "balance");
        
        Ok(())
    }
    
    /// Configure les chaînes de réaction hormonales
    pub fn setup_hormone_chains(&self) -> Result<(), String> {
        // Chaîne de réaction 1: Cascade de défense
        // Lorsque SecurityImmunity augmente, cela déclenche ConsensusHarmony et HashDifficulty
        self.hormone_interactions.insert(
            (HormoneType::SecurityImmunity, HormoneType::ConsensusHarmony),
            0.4
        );
        
        self.hormone_interactions.insert(
            (HormoneType::SecurityImmunity, HormoneType::HashDifficulty),
            0.5
        );
        
        self.hormone_interactions.insert(
            (HormoneType::ConsensusHarmony, HormoneType::ValidatorSynergy),
            0.3
        );
        
        // Chaîne de réaction 2: Cycle économique
        // MarketEquilibrium influence LiquidityFlow et TransactionFees
        self.hormone_interactions.insert(
            (HormoneType::MarketEquilibrium, HormoneType::LiquidityFlow),
            0.35
        );
        
        self.hormone_interactions.insert(
            (HormoneType::LiquidityFlow, HormoneType::TransactionFees),
            -0.25 // Effet négatif: plus de liquidité = frais plus bas
        );
        
        self.hormone_interactions.insert(
            (HormoneType::TransactionFees, HormoneType::RewardSignaling),
            0.4
        );
        
        self.hormone_interactions.insert(
            (HormoneType::RewardSignaling, HormoneType::BlockCatalyst),
            0.3
        );
        
        // Chaîne de réaction 3: Cycle d'expansion
        self.hormone_interactions.insert(
            (HormoneType::NetworkExpansion, HormoneType::MemoryOptimization),
            0.2
        );
        
        self.hormone_interactions.insert(
            (HormoneType::MemoryOptimization, HormoneType::EnergyEfficiency),
            0.3
        );
        
        self.hormone_interactions.insert(
            (HormoneType::NetworkExpansion, HormoneType::DecentralizationGuardian),
            0.4
        );
        
        // Chaîne de réaction 4: Cycle de réparation et maintenance
        self.hormone_interactions.insert(
            (HormoneType::ChainRepair, HormoneType::EnergyEfficiency),
            -0.2 // La réparation consomme de l'énergie
        );
        
        self.hormone_interactions.insert(
            (HormoneType::ChainRepair, HormoneType::ValidatorSynergy),
            0.3
        );
        
        // Chaîne de réaction 5: Cycle d'innovation
        self.hormone_interactions.insert(
            (HormoneType::EvolutionaryPressure, HormoneType::HyperdimensionalAdaption),
            0.5
        );
        
        self.hormone_interactions.insert(
            (HormoneType::HyperdimensionalAdaption, HormoneType::NetworkExpansion),
            0.3
        );
        
        // Effets de rétroaction négative pour stabilisation (homéostasie)
        self.hormone_interactions.insert(
            (HormoneType::BlockCatalyst, HormoneType::HashDifficulty),
            0.4 // Plus de blocs = difficulté plus élevée
        );
        
        self.hormone_interactions.insert(
            (HormoneType::TransactionFees, HormoneType::LiquidityFlow),
            -0.3 // Frais élevés réduisent la liquidité
        );
        
        self.hormone_interactions.insert(
            (HormoneType::HashDifficulty, HormoneType::EnergyEfficiency),
            -0.4 // Difficulté élevée réduit l'efficacité énergétique
        );
        
        Ok(())
    }
    
    /// Réinitialise l'équilibre hormonal à des niveaux normaux
    pub fn reset_hormone_balance(&self) -> Result<(), String> {
        // Niveaux d'équilibre pour chaque hormone
        let balanced_levels: HashMap<HormoneType, f64> = [
            (HormoneType::MarketEquilibrium, 0.5),
            (HormoneType::BlockCatalyst, 0.4),
            (HormoneType::HashDifficulty, 0.5),
            (HormoneType::TransactionFees, 0.4),
            (HormoneType::NetworkExpansion, 0.3),
            (HormoneType::SecurityImmunity, 0.6),
            (HormoneType::MemoryOptimization, 0.5),
            (HormoneType::ConsensusHarmony, 0.5),
            (HormoneType::EvolutionaryPressure, 0.2),
            (HormoneType::LiquidityFlow, 0.5),
            (HormoneType::EnergyEfficiency, 0.5),
            (HormoneType::ChainRepair, 0.3),
            (HormoneType::ValidatorSynergy, 0.4),
            (HormoneType::HyperdimensionalAdaption, 0.1),
            (HormoneType::RewardSignaling, 0.5),
            (HormoneType::DecentralizationGuardian, 0.6),
        ].iter().cloned().collect();
        
        // Ajuster progressivement les niveaux vers l'équilibre
        for (hormone_type, target_level) in &balanced_levels {
            if let Some(mut profile) = self.hormone_profiles.get_mut(hormone_type) {
                let current = profile.level;
                let delta = target_level - current;
                
                // Transition progressive (50% du chemin vers l'équilibre)
                profile.level = current + delta * 0.5;
            }
        }
        
        // Enregistrer l'événement de réinitialisation
        let current_block = self.bios_time.age_in_blocks();
        
        // Créer un snapshot des niveaux après réinitialisation
        let mut levels = HashMap::new();
        for entry in self.hormone_profiles.iter() {
            levels.insert(*entry.key(), entry.value().level);
        }
        
        self.historical_data.write().insert(current_block, levels);
        
        Ok(())
    }
    
    /// Simule une réponse au stress du système
    pub fn simulate_stress_response(&self, intensity: f64, duration_secs: u64) -> Result<(), String> {
        if intensity <= 0.0 || intensity > 1.0 {
            return Err("L'intensité doit être comprise entre 0.0 et 1.0".to_string());
        }
        
        // Augmenter les hormones de stress
        if let Some(mut security) = self.hormone_profiles.get_mut(&HormoneType::SecurityImmunity) {
            security.level = (security.level + intensity * 0.6).min(1.0);
        }
        
        if let Some(mut hash_diff) = self.hormone_profiles.get_mut(&HormoneType::HashDifficulty) {
            hash_diff.level = (hash_diff.level + intensity * 0.4).min(1.0);
        }
        
        if let Some(mut energy) = self.hormone_profiles.get_mut(&HormoneType::EnergyEfficiency) {
            // Diminuer l'efficacité énergétique (consommation accrue)
            energy.level = (energy.level - intensity * 0.3).max(0.1);
        }
        
        if let Some(mut tx_fees) = self.hormone_profiles.get_mut(&HormoneType::TransactionFees) {
            // Augmenter les frais de transaction
            tx_fees.level = (tx_fees.level + intensity * 0.5).min(1.0);
        }
        
        if let Some(mut liquidity) = self.hormone_profiles.get_mut(&HormoneType::LiquidityFlow) {
            // Réduire la liquidité
            liquidity.level = (liquidity.level - intensity * 0.4).max(0.1);
        }
        
        // Enregistrer l'événement de stress dans l'historique
        let current_block = self.bios_time.age_in_blocks();
        
        // Créer un snapshot des niveaux après le stress
        let mut levels = HashMap::new();
        for entry in self.hormone_profiles.iter() {
            levels.insert(*entry.key(), entry.value().level);
        }
        
        self.historical_data.write().insert(current_block, levels);
        
        // Configurer le facteur externe de stress
        {
            let mut factors = self.external_factors.write();
            factors.insert("system_stress".to_string(), intensity);
        }
        
        Ok(())
    }
    
    /// Simule une réponse de bien-être du système (opposée au stress)
    pub fn simulate_wellbeing_response(&self, intensity: f64) -> Result<(), String> {
        if intensity <= 0.0 || intensity > 1.0 {
            return Err("L'intensité doit être comprise entre 0.0 et 1.0".to_string());
        }
        
        // Augmenter les hormones positives
        if let Some(mut consensus) = self.hormone_profiles.get_mut(&HormoneType::ConsensusHarmony) {
            consensus.level = (consensus.level + intensity * 0.5).min(1.0);
        }
        
        if let Some(mut validators) = self.hormone_profiles.get_mut(&HormoneType::ValidatorSynergy) {
            validators.level = (validators.level + intensity * 0.4).min(1.0);
        }
        
        if let Some(mut energy) = self.hormone_profiles.get_mut(&HormoneType::EnergyEfficiency) {
            // Améliorer l'efficacité énergétique
            energy.level = (energy.level + intensity * 0.4).min(1.0);
        }
        
        if let Some(mut market) = self.hormone_profiles.get_mut(&HormoneType::MarketEquilibrium) {
            // Améliorer l'équilibre du marché
            market.level = (market.level + intensity * 0.3).min(1.0);
        }
        
        if let Some(mut liquidity) = self.hormone_profiles.get_mut(&HormoneType::LiquidityFlow) {
            // Augmenter la liquidité
            liquidity.level = (liquidity.level + intensity * 0.5).min(1.0);
        }
        
        if let Some(mut tx_fees) = self.hormone_profiles.get_mut(&HormoneType::TransactionFees) {
            // Réduire les frais de transaction
            tx_fees.level = (tx_fees.level - intensity * 0.2).max(0.1);
        }
        
        // Enregistrer l'événement de bien-être dans l'historique
        let current_block = self.bios_time.age_in_blocks();
        
        // Créer un snapshot des niveaux après le bien-être
        let mut levels = HashMap::new();
        for entry in self.hormone_profiles.iter() {
            levels.insert(*entry.key(), entry.value().level);
        }
        
        self.historical_data.write().insert(current_block, levels);
        
        // Configurer le facteur externe de bien-être
        {
            let mut factors = self.external_factors.write();
            factors.insert("system_wellbeing".to_string(), intensity);
            // Réduire le stress s'il existe
            if factors.contains_key("system_stress") {
                factors.insert("system_stress".to_string(), 
                              (factors.get("system_stress").unwrap_or(&0.0) * 0.5).max(0.0));
            }
        }
        
        Ok(())
    }
}

impl Default for HormonalField {
    fn default() -> Self {
        Self::new(Arc::new(BiosTime::new()))
    }
}
