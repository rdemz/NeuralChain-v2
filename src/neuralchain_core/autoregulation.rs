//! Module d'Autorégulation Avancée pour NeuralChain-v2
//! 
//! Ce module implémente des mécanismes d'autorégulation homéostatique
//! permettant à l'organisme blockchain d'ajuster automatiquement ses paramètres
//! pour maintenir un état optimal de fonctionnement face à des conditions
//! environnementales changeantes.
//!
//! Optimisé spécifiquement pour Windows avec analyse de performance native
//! et zéro dépendance Linux.

use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use dashmap::DashMap;

use crate::neuralchain_core::quantum_organism::QuantumOrganism;
use crate::neuralchain_core::cortical_hub::CorticalHub;
use crate::neuralchain_core::hormonal_field::{HormonalField, HormoneType};
use crate::neuralchain_core::emergent_consciousness::ConsciousnessEngine;
use crate::bios_time::neuralchain_core::BiosTime;

/// Paramètre autorégulé
#[derive(Debug, Clone)]
pub struct RegulatedParameter {
    /// Nom du paramètre
    pub name: String,
    /// Valeur actuelle
    pub current_value: f64,
    /// Plage de valeurs autorisées [min, max]
    pub range: (f64, f64),
    /// Valeur cible optimale
    pub target_value: f64,
    /// Dernière mise à jour
    pub last_update: Instant,
    /// Historique des valeurs récentes
    pub history: VecDeque<(Instant, f64)>,
    /// Taux de changement maximal autorisé par cycle
    pub max_change_rate: f64,
    /// Priorité du paramètre (1-10)
    pub priority: u8,
    /// Composant associé
    pub component: String,
    /// Métadonnées
    pub metadata: HashMap<String, String>,
}

impl RegulatedParameter {
    /// Crée un nouveau paramètre régulé
    pub fn new(name: &str, initial_value: f64, min: f64, max: f64, component: &str) -> Self {
        let mut history = VecDeque::with_capacity(100);
        history.push_back((Instant::now(), initial_value));
        
        Self {
            name: name.to_string(),
            current_value: initial_value,
            range: (min, max),
            target_value: initial_value,
            last_update: Instant::now(),
            history,
            max_change_rate: 0.1,
            priority: 5,
            component: component.to_string(),
            metadata: HashMap::new(),
        }
    }
    
    /// Définit la valeur cible
    pub fn set_target(&mut self, target: f64) {
        self.target_value = target.max(self.range.0).min(self.range.1);
    }
    
    /// Ajuste la valeur actuelle en direction de la cible
    pub fn adjust_toward_target(&mut self, adjustment_factor: f64) -> f64 {
        if (self.current_value - self.target_value).abs() < 1e-6 {
            return 0.0; // Déjà à la valeur cible
        }
        
        let direction = if self.target_value > self.current_value { 1.0 } else { -1.0 };
        
        // Calculer l'ajustement limité par le taux max de changement
        let distance = (self.target_value - self.current_value).abs();
        let max_step = self.max_change_rate * adjustment_factor;
        let actual_step = max_step.min(distance) * direction;
        
        // Appliquer l'ajustement
        let new_value = self.current_value + actual_step;
        let clamped_value = new_value.max(self.range.0).min(self.range.1);
        
        let delta = clamped_value - self.current_value;
        self.current_value = clamped_value;
        
        // Enregistrer dans l'historique
        self.history.push_back((Instant::now(), clamped_value));
        if self.history.len() > 100 {
            self.history.pop_front();
        }
        
        self.last_update = Instant::now();
        
        delta
    }
    
    /// Calcule la tendance récente
    pub fn calculate_trend(&self, timeframe: Duration) -> f64 {
        let now = Instant::now();
        
        // Filtrer l'historique selon le timeframe
        let recent_values: Vec<_> = self.history.iter()
            .filter(|(time, _)| now.duration_since(*time) <= timeframe)
            .collect();
            
        if recent_values.len() < 2 {
            return 0.0;
        }
        
        // Calculer la pente de la tendance linéaire
        let first = recent_values.first().unwrap();
        let last = recent_values.last().unwrap();
        
        let time_diff = last.0.duration_since(first.0).as_secs_f64();
        if time_diff < 0.001 {
            return 0.0;
        }
        
        let value_diff = last.1 - first.1;
        value_diff / time_diff
    }
}

/// Définit la relation entre deux paramètres
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParameterRelation {
    /// Augmenter un paramètre augmente l'autre
    DirectlyProportional,
    /// Augmenter un paramètre diminue l'autre
    InverselyProportional,
    /// Relation complexe ou non-linéaire
    Complex,
    /// Changements coordonnés (les deux changent ensemble)
    Coordinated,
    /// Pas de relation directe
    Independent,
}

/// Règle d'autorégulation
#[derive(Debug, Clone)]
pub struct RegulationRule {
    /// Identifiant unique de la règle
    pub id: String,
    /// Description de la règle
    pub description: String,
    /// Condition d'activation
    pub condition: RegulationCondition,
    /// Action à effectuer
    pub action: RegulationAction,
    /// Priorité de la règle (1-10)
    pub priority: u8,
    /// Dernière activation de la règle
    pub last_trigger: Option<Instant>,
    /// Nombre total d'activations
    pub trigger_count: u32,
    /// Règle active
    pub active: bool,
}

/// Condition d'activation d'une règle
#[derive(Debug, Clone)]
pub enum RegulationCondition {
    /// Paramètre au-dessus d'un seuil
    ParameterAbove(String, f64),
    /// Paramètre en dessous d'un seuil
    ParameterBelow(String, f64),
    /// Paramètre dans une plage
    ParameterInRange(String, f64, f64),
    /// Paramètre change plus vite qu'un taux donné
    ParameterChangeRate(String, f64),
    /// Deux paramètres différant par une marge
    ParameterDifference(String, String, f64),
    /// Tendance d'un paramètre
    ParameterTrend(String, f64, Duration),
    /// Combinaison ET de conditions
    And(Vec<RegulationCondition>),
    /// Combinaison OU de conditions
    Or(Vec<RegulationCondition>),
    /// Condition temporelle (intervalle régulier)
    TimeInterval(Duration),
    /// Condition basée sur un état système
    SystemState(String, String),
}

/// Action d'une règle d'autorégulation
#[derive(Debug, Clone)]
pub enum RegulationAction {
    /// Ajuster un paramètre à une valeur absolue
    SetParameter(String, f64),
    /// Ajuster un paramètre par un delta relatif
    AdjustParameter(String, f64),
    /// Ajuster un paramètre vers une cible avec un facteur
    AdjustTowardTarget(String, f64),
    /// Émettre une hormone
    EmitHormone(HormoneType, String, f64, f64, f64),
    /// Définir une cible pour un paramètre
    SetTarget(String, f64),
    /// Combinaison d'actions
    Sequence(Vec<RegulationAction>),
    /// Activer une autre règle
    ActivateRule(String),
    /// Désactiver une règle
    DeactivateRule(String),
    /// Action personnalisée
    Custom(String, String),
}

/// Mesure de performance système
#[derive(Debug, Clone)]
pub struct SystemMetric {
    /// Nom de la métrique
    pub name: String,
    /// Valeur actuelle
    pub current_value: f64,
    /// Unité de mesure
    pub unit: String,
    /// Historique des valeurs
    pub history: VecDeque<(Instant, f64)>,
    /// Composant source
    pub source: String,
    /// Importance (0.0-1.0)
    pub importance: f64,
}

impl SystemMetric {
    /// Crée une nouvelle métrique système
    pub fn new(name: &str, initial_value: f64, unit: &str, source: &str) -> Self {
        let mut history = VecDeque::with_capacity(100);
        history.push_back((Instant::now(), initial_value));
        
        Self {
            name: name.to_string(),
            current_value: initial_value,
            unit: unit.to_string(),
            history,
            source: source.to_string(),
            importance: 0.5,
        }
    }
    
    /// Ajoute une nouvelle valeur
    pub fn add_measurement(&mut self, value: f64) {
        self.current_value = value;
        self.history.push_back((Instant::now(), value));
        
        if self.history.len() > 100 {
            self.history.pop_front();
        }
    }
}

/// État de stress du système
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StressState {
    /// Fonctionnement optimal
    Optimal,
    /// Stress léger
    LightStress,
    /// Stress modéré
    ModerateStress,
    /// Stress élevé
    HighStress,
    /// État critique
    Critical,
}

/// Système d'autorégulation principal
pub struct Autoregulation {
    /// Référence à l'organisme parent
    organism: Arc<QuantumOrganism>,
    /// Référence au cortex
    cortical_hub: Arc<CorticalHub>,
    /// Référence au système hormonal
    hormonal_system: Arc<HormonalField>,
    /// Référence à la conscience
    consciousness: Arc<ConsciousnessEngine>,
    /// Référence à l'horloge interne
    bios_clock: Arc<BiosTime>,
    /// Paramètres régulés
    parameters: DashMap<String, RegulatedParameter>,
    /// Règles de régulation
    rules: DashMap<String, RegulationRule>,
    /// Métriques système
    metrics: DashMap<String, SystemMetric>,
    /// Relations entre paramètres
    parameter_relations: DashMap<(String, String), ParameterRelation>,
    /// État de stress actuel
    stress_state: RwLock<StressState>,
    /// Capacité d'adaptation (0.0-1.0)
    adaptation_capacity: RwLock<f64>,
    /// Dernier cycle de régulation
    last_regulation_cycle: Mutex<Instant>,
    /// Cycle de régulation actif
    regulation_active: std::sync::atomic::AtomicBool,
}

impl Autoregulation {
    /// Crée un nouveau système d'autorégulation
    pub fn new(
        organism: Arc<QuantumOrganism>,
        cortical_hub: Arc<CorticalHub>,
        hormonal_system: Arc<HormonalField>,
        consciousness: Arc<ConsciousnessEngine>,
        bios_clock: Arc<BiosTime>,
    ) -> Self {
        Self {
            organism,
            cortical_hub,
            hormonal_system,
            consciousness,
            bios_clock,
            parameters: DashMap::new(),
            rules: DashMap::new(),
            metrics: DashMap::new(),
            parameter_relations: DashMap::new(),
            stress_state: RwLock::new(StressState::Optimal),
            adaptation_capacity: RwLock::new(0.8),
            last_regulation_cycle: Mutex::new(Instant::now()),
            regulation_active: std::sync::atomic::AtomicBool::new(false),
        }
    }
    
    /// Enregistre un nouveau paramètre
    pub fn register_parameter(&self, parameter: RegulatedParameter) -> Result<(), String> {
        // Vérifier si le paramètre existe déjà
        if self.parameters.contains_key(&parameter.name) {
            return Err(format!("Paramètre '{}' déjà enregistré", parameter.name));
        }
        
        // Enregistrer le paramètre
        self.parameters.insert(parameter.name.clone(), parameter);
        
        Ok(())
    }
    
    /// Enregistre une nouvelle règle
    pub fn register_rule(&self, rule: RegulationRule) -> Result<(), String> {
        // Vérifier si la règle existe déjà
        if self.rules.contains_key(&rule.id) {
            return Err(format!("Règle '{}' déjà enregistrée", rule.id));
        }
        
        // Enregistrer la règle
        self.rules.insert(rule.id.clone(), rule);
        
        Ok(())
    }
    
    /// Démarre le cycle d'autorégulation
    pub fn start_regulation(&self) -> Result<(), String> {
        // Vérifier si le cycle est déjà actif
        if self.regulation_active.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("Cycle d'autorégulation déjà actif".to_string());
        }
        
        // Marquer comme actif
        self.regulation_active.store(true, std::sync::atomic::Ordering::SeqCst);
        
        // Créer les règles et paramètres de base si nécessaire
        if self.parameters.is_empty() {
            self.initialize_default_parameters();
        }
        
        if self.rules.is_empty() {
            self.initialize_default_rules();
        }
        
        // Mettre à jour les métriques système
        self.update_system_metrics();
        
        // Démarrer le thread de régulation
        let self_arc = Arc::new(self.clone());
        
        std::thread::spawn(move || {
            // Activer les optimisations Windows
            #[cfg(target_os = "windows")]
            self_arc.optimize_windows_thread();
            
            println!("Cycle d'autorégulation démarré");
            
            while self_arc.regulation_active.load(std::sync::atomic::Ordering::SeqCst) {
                // Exécuter un cycle de régulation
                if let Err(e) = self_arc.regulation_cycle() {
                    println!("Erreur dans le cycle d'autorégulation: {}", e);
                }
                
                // Pause entre les cycles
                std::thread::sleep(Duration::from_millis(1000));
            }
        });
        
        Ok(())
    }
    
    /// Arrête le cycle d'autorégulation
    pub fn stop_regulation(&self) -> Result<(), String> {
        // Vérifier si le cycle est actif
        if !self.regulation_active.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("Cycle d'autorégulation déjà arrêté".to_string());
        }
        
        // Marquer comme inactif
        self.regulation_active.store(false, std::sync::atomic::Ordering::SeqCst);
        
        Ok(())
    }
    
    /// Exécute un cycle d'autorégulation
    fn regulation_cycle(&self) -> Result<(), String> {
        // Mettre à jour l'horodatage du dernier cycle
        let mut last_cycle = self.last_regulation_cycle.lock();
        *last_cycle = Instant::now();
        
        // Mettre à jour les métriques système
        self.update_system_metrics();
        
        // Évaluer l'état de stress système
        self.evaluate_stress_state();
        
        // Évaluer et déclencher les règles actives
        let triggered_rules = self.evaluate_rules();
        
        // Appliquer l'autorégulation
        self.apply_regulation(triggered_rules);
        
        Ok(())
    }
    
    /// Optimisations spécifiques à Windows pour le thread de régulation
    #[cfg(target_os = "windows")]
    fn optimize_windows_thread(&self) {
        use crate::neuralchain_core::system_utils::{SystemMonitor, PerformanceOptimizer};
        
        unsafe {
            // Augmenter la priorité du thread
            let current_thread = GetCurrentThread();
            SetThreadPriority(current_thread, THREAD_PRIORITY_ABOVE_NORMAL);
        }
    }
    
    /// Initialise les paramètres par défaut
    fn initialize_default_parameters(&self) {
        // Paramètres du système hormonal
        let hormone_base = RegulatedParameter::new("hormone_base_production", 0.5, 0.1, 1.0, "hormonal_system");
        let hormone_decay = RegulatedParameter::new("hormone_decay_rate", 0.1, 0.01, 0.5, "hormonal_system");
        
        // Paramètres du cortex
        let neural_plasticity = RegulatedParameter::new("neural_plasticity", 0.6, 0.2, 1.0, "cortical_hub");
        let activation_threshold = RegulatedParameter::new("activation_threshold", 0.3, 0.1, 0.9, "cortical_hub");
        
        // Paramètres de conscience
        let consciousness_depth = RegulatedParameter::new("consciousness_depth", 0.5, 0.1, 1.0, "consciousness");
        let thought_duration = RegulatedParameter::new("thought_duration", 10.0, 1.0, 30.0, "consciousness");
        
        // Paramètres métaboliques
        let energy_allocation = RegulatedParameter::new("energy_allocation", 0.7, 0.2, 1.0, "organism");
        let idle_threshold = RegulatedParameter::new("idle_threshold", 0.3, 0.1, 0.6, "organism");
        
        // Enregistrer les paramètres
        let _ = self.register_parameter(hormone_base);
        let _ = self.register_parameter(hormone_decay);
        let _ = self.register_parameter(neural_plasticity);
        let _ = self.register_parameter(activation_threshold);
        let _ = self.register_parameter(consciousness_depth);
        let _ = self.register_parameter(thought_duration);
        let _ = self.register_parameter(energy_allocation);
        let _ = self.register_parameter(idle_threshold);
        
        // Enregistrer les relations entre paramètres
        self.parameter_relations.insert(
            ("neural_plasticity".to_string(), "activation_threshold".to_string()),
            ParameterRelation::InverselyProportional
        );
        
        self.parameter_relations.insert(
            ("consciousness_depth".to_string(), "energy_allocation".to_string()),
            ParameterRelation::DirectlyProportional
        );
        
        self.parameter_relations.insert(
            ("hormone_base_production".to_string(), "hormone_decay_rate".to_string()),
            ParameterRelation::Coordinated
        );
    }
    
    /// Initialise les règles par défaut
    fn initialize_default_rules(&self) {
        // Règle 1: Si le stress est élevé, réduire la plasticité neurale
        let high_stress_rule = RegulationRule {
            id: "high_stress_adaptation".to_string(),
            description: "Réduire la plasticité neurale en cas de stress élevé".to_string(),
            condition: RegulationCondition::Or(vec![
                RegulationCondition::SystemState("stress".to_string(), "HighStress".to_string()),
                RegulationCondition::SystemState("stress".to_string(), "Critical".to_string()),
            ]),
            action: RegulationAction::Sequence(vec![
                RegulationAction::AdjustParameter("neural_plasticity".to_string(), -0.1),
                RegulationAction::EmitHormone(
                    HormoneType::Cortisol, 
                    "stress_response".to_string(), 
                    0.7, 0.9, 0.6
                ),
            ]),
            priority: 8,
            last_trigger: None,
            trigger_count: 0,
            active: true,
        };
        
        // Règle 2: Optimiser l'énergie pendant les périodes de faible activité
        let low_activity_rule = RegulationRule {
            id: "energy_optimization".to_string(),
            description: "Optimiser l'allocation d'énergie pendant les périodes calmes".to_string(),
            condition: RegulationCondition::And(vec![
                RegulationCondition::ParameterBelow("energy_allocation".to_string(), 0.5),
                RegulationCondition::SystemState("stress".to_string(), "Optimal".to_string()),
            ]),
            action: RegulationAction::Sequence(vec![
                RegulationAction::AdjustParameter("energy_allocation".to_string(), 0.05),
                RegulationAction::AdjustParameter("idle_threshold".to_string(), 0.02),
            ]),
            priority: 5,
            last_trigger: None,
            trigger_count: 0,
            active: true,
        };
        
        // Règle 3: Renforcer la conscience pendant les périodes d'activité intense
        let high_activity_rule = RegulationRule {
            id: "consciousness_boost".to_string(),
            description: "Renforcer la profondeur de conscience lors des pics d'activité".to_string(),
            condition: RegulationCondition::ParameterAbove("activation_threshold".to_string(), 0.6),
            action: RegulationAction::Sequence(vec![
                RegulationAction::AdjustTowardTarget("consciousness_depth".to_string(), 0.8),
                RegulationAction::EmitHormone(
                    HormoneType::Dopamine, 
                    "activity_boost".to_string(), 
                    0.6, 0.8, 0.7
                ),
            ]),
            priority: 6,
            last_trigger: None,
            trigger_count: 0,
            active: true,
        };
        
        // Règle 4: Régulation périodique du système hormonal
        let hormone_regulation_rule = RegulationRule {
            id: "hormone_regulation".to_string(),
            description: "Équilibrer périodiquement le système hormonal".to_string(),
            condition: RegulationCondition::TimeInterval(Duration::from_secs(300)), // 5 minutes
            action: RegulationAction::Sequence(vec![
                RegulationAction::SetTarget("hormone_base_production".to_string(), 0.5),
                RegulationAction::AdjustTowardTarget("hormone_decay_rate".to_string(), 0.2),
            ]),
            priority: 4,
            last_trigger: None,
            trigger_count: 0,
            active: true,
        };
        
        // Enregistrer les règles
        let _ = self.register_rule(high_stress_rule);
        let _ = self.register_rule(low_activity_rule);
        let _ = self.register_rule(high_activity_rule);
        let _ = self.register_rule(hormone_regulation_rule);
    }
    
    /// Met à jour les métriques système
    fn update_system_metrics(&self) {
        // CPU Load
        let cpu_load = self.get_system_cpu_load();
        if let Some(mut metric) = self.metrics.get_mut("cpu_load") {
            metric.add_measurement(cpu_load);
        } else {
            let metric = SystemMetric::new("cpu_load", cpu_load, "%", "system");
            self.metrics.insert("cpu_load".to_string(), metric);
        }
        
        // Memory Usage
        let memory_usage = self.get_system_memory_usage();
        if let Some(mut metric) = self.metrics.get_mut("memory_usage") {
            metric.add_measurement(memory_usage);
        } else {
            let metric = SystemMetric::new("memory_usage", memory_usage, "%", "system");
            self.metrics.insert("memory_usage".to_string(), metric);
        }
        
        // Neural Activity
        let neural_activity = self.get_neural_activity();
        if let Some(mut metric) = self.metrics.get_mut("neural_activity") {
            metric.add_measurement(neural_activity);
        } else {
            let metric = SystemMetric::new("neural_activity", neural_activity, "level", "cortical_hub");
            self.metrics.insert("neural_activity".to_string(), metric);
        }
        
        // Hormone Balance
        let hormone_balance = self.get_hormone_balance();
        if let Some(mut metric) = self.metrics.get_mut("hormone_balance") {
            metric.add_measurement(hormone_balance);
        } else {
            let metric = SystemMetric::new("hormone_balance", hormone_balance, "index", "hormonal_system");
            self.metrics.insert("hormone_balance".to_string(), metric);
        }
        
        // Consciousness Level
        let consciousness_level = self.get_consciousness_level();
        if let Some(mut metric) = self.metrics.get_mut("consciousness_level") {
            metric.add_measurement(consciousness_level);
        } else {
            let metric = SystemMetric::new("consciousness_level", consciousness_level, "level", "consciousness");
            self.metrics.insert("consciousness_level".to_string(), metric);
        }
    }
    
    /// Évalue l'état de stress du système
    fn evaluate_stress_state(&self) {
        // Récupérer les métriques clés
        let cpu_load = self.metrics.get("cpu_load").map_or(0.5, |m| m.current_value) / 100.0;
        let memory_usage = self.metrics.get("memory_usage").map_or(0.5, |m| m.current_value) / 100.0;
        let neural_activity = self.metrics.get("neural_activity").map_or(0.5, |m| m.current_value);
        let hormone_balance = self.metrics.get("hormone_balance").map_or(0.5, |m| m.current_value);
        
        // Calculer un score de stress composite
        let stress_score = cpu_load * 0.3 + memory_usage * 0.2 + neural_activity * 0.3 + 
                         (1.0 - hormone_balance) * 0.2;
        
        // Déterminer l'état de stress
        let stress_state = if stress_score < 0.3 {
            StressState::Optimal
        } else if stress_score < 0.5 {
            StressState::LightStress
        } else if stress_score < 0.7 {
            StressState::ModerateStress
        } else if stress_score < 0.9 {
            StressState::HighStress
        } else {
            StressState::Critical
        };
        
        // Mettre à jour l'état
        let mut current_state = self.stress_state.write();
        
        // Si l'état a changé, émettre une hormone appropriée
        if *current_state != stress_state {
            match stress_state {
                StressState::Optimal => {
                    self.hormonal_system.emit_hormone(
                        HormoneType::Serotonin,
                        "stress_regulation",
                        0.6,
                        0.8,
                        0.5,
                        HashMap::new(),
                    ).unwrap_or_default();
                },
                StressState::LightStress => {
                    self.hormonal_system.emit_hormone(
                        HormoneType::Adrenaline,
                        "stress_regulation",
                        0.3,
                        0.5,
                        0.4,
                        HashMap::new(),
                    ).unwrap_or_default();
                },
                StressState::ModerateStress => {
                    self.hormonal_system.emit_hormone(
                        HormoneType::Adrenaline,
                        "stress_regulation",
                        0.5,
                        0.7,
                        0.6,
                        HashMap::new(),
                    ).unwrap_or_default();
                },
                StressState::HighStress => {
                    self.hormonal_system.emit_hormone(
                        HormoneType::Cortisol,
                        "stress_regulation",
                        0.7,
                        0.8,
                        0.7,
                        HashMap::new(),
                    ).unwrap_or_default();
                },
                StressState::Critical => {
                    self.hormonal_system.emit_hormone(
                        HormoneType::Cortisol,
                        "stress_regulation",
                        0.9,
                        1.0,
                        0.9,
                        HashMap::new(),
                    ).unwrap_or_default();
                },
            }
        }
        
        *current_state = stress_state;
    }
    
    /// Évalue les règles et retourne celles qui sont déclenchées
    fn evaluate_rules(&self) -> Vec<String> {
        let mut triggered_rules = Vec::new();
        
        // Évaluer toutes les règles actives
        for entry in self.rules.iter() {
            let rule = entry.value();
            
            // Ignorer les règles inactives
            if !rule.active {
                continue;
            }
            
            // Évaluer la condition
            if self.evaluate_condition(&rule.condition) {
                triggered_rules.push(rule.id.clone());
                
                // Mettre à jour les statistiques de la règle
                if let Some(mut rule_entry) = self.rules.get_mut(&rule.id) {
                    let rule = rule_entry.value_mut();
                    rule.last_trigger = Some(Instant::now());
                    rule.trigger_count += 1;
                }
            }
        }
        
        // Trier par priorité
        triggered_rules.sort_by(|a, b| {
            let priority_a = self.rules.get(a).map_or(0, |r| r.priority);
            let priority_b = self.rules.get(b).map_or(0, |r| r.priority);
            priority_b.cmp(&priority_a) // Du plus prioritaire au moins prioritaire
        });
        
        triggered_rules
    }
    
    /// Évalue une condition
    fn evaluate_condition(&self, condition: &RegulationCondition) -> bool {
        match condition {
            RegulationCondition::ParameterAbove(param_name, threshold) => {
                if let Some(param) = self.parameters.get(param_name) {
                    param.current_value > *threshold
                } else {
                    false
                }
            },
            
            RegulationCondition::ParameterBelow(param_name, threshold) => {
                if let Some(param) = self.parameters.get(param_name) {
                    param.current_value < *threshold
                } else {
                    false
                }
            },
            
            RegulationCondition::ParameterInRange(param_name, min, max) => {
                if let Some(param) = self.parameters.get(param_name) {
                    param.current_value >= *min && param.current_value <= *max
                } else {
                    false
                }
            },
            
            RegulationCondition::ParameterChangeRate(param_name, rate) => {
                if let Some(param) = self.parameters.get(param_name) {
                    if param.history.len() < 2 {
                        false
                    } else {
                        let oldest = param.history.front().unwrap();
                        let newest = param.history.back().unwrap();
                        
                        let time_diff = newest.0.duration_since(oldest.0).as_secs_f64();
                        if time_diff < 0.001 {
                            false
                        } else {
                            let value_diff = newest.1 - oldest.1;
                            (value_diff / time_diff).abs() > *rate
                        }
                    }
                } else {
                    false
                }
            },
            
            RegulationCondition::ParameterDifference(param1, param2, margin) => {
                if let (Some(p1), Some(p2)) = (self.parameters.get(param1), self.parameters.get(param2)) {
                    (p1.current_value - p2.current_value).abs() > *margin
                } else {
                    false
                }
            },
            
            RegulationCondition::ParameterTrend(param_name, trend, duration) => {
                if let Some(param) = self.parameters.get(param_name) {
                    let actual_trend = param.calculate_trend(*duration);
                    
                    if *trend >= 0.0 {
                        actual_trend >= *trend
                    } else {
                        actual_trend <= *trend
                    }
                } else {
                    false
                }
            },
            
            RegulationCondition::And(conditions) => {
                conditions.iter().all(|c| self.evaluate_condition(c))
            },
            
            RegulationCondition::Or(conditions) => {
                conditions.iter().any(|c| self.evaluate_condition(c))
            },
            
            RegulationCondition::TimeInterval(interval) => {
                let now = Instant::now();
                
                // Pour les intervalles, vérifier par rapport au dernier cycle
                let last_cycle = *self.last_regulation_cycle.lock();
                now.duration_since(last_cycle) >= *interval
            },
            
            RegulationCondition::SystemState(state_name, expected_value) => {
                match state_name.as_str() {
                    "stress" => {
                        let current_stress = *self.stress_state.read();
                        match expected_value.as_str() {
                            "Optimal" => current_stress == StressState::Optimal,
                            "LightStress" => current_stress == StressState::LightStress,
                            "ModerateStress" => current_stress == StressState::ModerateStress,
                            "HighStress" => current_stress == StressState::HighStress,
                            "Critical" => current_stress == StressState::Critical,
                            _ => false,
                        }
                    },
                    // Autres états système...
                    _ => false,
                }
            },
        }
    }
    
    /// Applique les actions de régulation pour les règles déclenchées
    fn apply_regulation(&self, rule_ids: Vec<String>) {
        let adaptation_capacity = *self.adaptation_capacity.read();
        
        for rule_id in rule_ids {
            if let Some(rule) = self.rules.get(&rule_id) {
                // Appliquer l'action de la règle
                self.apply_action(&rule.action, adaptation_capacity);
            }
        }
    }
    
    /// Applique une action de régulation
    fn apply_action(&self, action: &RegulationAction, adaptation_factor: f64) {
        match action {
            RegulationAction::SetParameter(param_name, value) => {
                if let Some(mut param) = self.parameters.get_mut(param_name) {
                    let clamped_value = value.max(param.range.0).min(param.range.1);
                    param.current_value = clamped_value;
                    param.last_update = Instant::now();
                    param.history.push_back((Instant::now(), clamped_value));
                    
                    if param.history.len() > 100 {
                        param.history.pop_front();
                    }
                }
            },
            
            RegulationAction::AdjustParameter(param_name, delta) => {
                if let Some(mut param) = self.parameters.get_mut(param_name) {
                    let new_value = param.current_value + delta * adaptation_factor;
                    let clamped_value = new_value.max(param.range.0).min(param.range.1);
                    
                    param.current_value = clamped_value;
                    param.last_update = Instant::now();
                    param.history.push_back((Instant::now(), clamped_value));
                    
                    if param.history.len() > 100 {
                        param.history.pop_front();
                    }
                }
            },
            
            RegulationAction::AdjustTowardTarget(param_name, factor) => {
                if let Some(mut param) = self.parameters.get_mut(param_name) {
                    param.adjust_toward_target(factor * adaptation_factor);
                }
            },
            
            RegulationAction::EmitHormone(hormone_type, reason, intensity, urgency, influence) => {
                let _ = self.hormonal_system.emit_hormone(
                    *hormone_type,
                    reason,
                    intensity * adaptation_factor,
                    urgency,
                    influence * adaptation_factor,
                    HashMap::new(),
                );
            },
            
            RegulationAction::SetTarget(param_name, target) => {
                if let Some(mut param) = self.parameters.get_mut(param_name) {
                    param.set_target(*target);
                }
            },
            
            RegulationAction::Sequence(actions) => {
                for sub_action in actions {
                    self.apply_action(sub_action, adaptation_factor);
                }
            },
            
            RegulationAction::ActivateRule(rule_id) => {
                if let Some(mut rule) = self.rules.get_mut(rule_id) {
                    rule.active = true;
                }
            },
            
            RegulationAction::DeactivateRule(rule_id) => {
                if let Some(mut rule) = self.rules.get_mut(rule_id) {
                    rule.active = false;
                }
            },
            
            RegulationAction::Custom(action_type, _param) => {
                // Actions personnalisées spécifiques
                match action_type.as_str() {
                    "neural_organization" => {
                        // Exemple: réorganisation neurale adaptative
                        self.cortical_hub.optimize_neural_pathways();
                    },
                    "consciousness_focus_shift" => {
                        // Exemple: déplacement du focus de conscience
                        self.consciousness.shift_attention_focus();
                    },
                    _ => {},
                }
            },
        }
    }
    
    /// Obtient la charge CPU actuelle du système
    #[cfg(target_os = "windows")]
    fn get_system_cpu_load(&self) -> f64 {
        use windows_sys::Win32::System::Performance::*;
        
        let mut cpu_load = 70.0; // Valeur par défaut en cas d'échec
        
        unsafe {
            let mut query: PDH_HQUERY = 0;
            let mut counter: PDH_HCOUNTER = 0;
            
            // Créer une requête PDH
            if PdhOpenQueryA(std::ptr::null(), 0, &mut query) == 0 {
                // Ajouter le compteur de charge CPU
                let counter_path = b"\\Processor(_Total)\\% Processor Time\0";
                
                if PdhAddEnglishCounterA(query, counter_path.as_ptr() as *const i8, 0, &mut counter) == 0 {
                    // Collecter les données
                    if PdhCollectQueryData(query) == 0 {
                        // Attendre un moment pour permettre au compteur de s'initialiser
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        
                        if PdhCollectQueryData(query) == 0 {
                            // Structure pour recevoir la valeur formatée
                            #[allow(non_snake_case)]
                            struct PDH_FMT_COUNTERVALUE_DOUBLE {
                                CStatus: u32,
                                DoubleValue: f64,
                            }
                            
                            let mut counter_value: PDH_FMT_COUNTERVALUE_DOUBLE = std::mem::zeroed();
                            let pdhStatus = PdhGetFormattedCounterValue(
                                counter,
                                PDH_FMT_DOUBLE,
                                std::ptr::null_mut(),
                                &mut counter_value as *mut _ as *mut _
                            );
                            
                            if pdhStatus == 0 {
                                cpu_load = counter_value.DoubleValue;
                            }
                        }
                    }
                }
                
                // Note: Dans une implémentation réelle, nous devrions fermer la requête
                // avec PdhCloseQuery, mais nous l'omettons ici pour simplifier
            }
        }
        
        cpu_load
    }

    /// Version portable pour obtenir la charge CPU
    #[cfg(not(target_os = "windows"))]
    fn get_system_cpu_load(&self) -> f64 {
        // Simulation d'une charge CPU aléatoire entre 30% et 90%
        30.0 + rand::thread_rng().gen::<f64>() * 60.0
    }
    
    /// Obtient l'utilisation mémoire actuelle du système
    #[cfg(target_os = "windows")]
    fn get_system_memory_usage(&self) -> f64 {
        use windows_sys::Win32::System::SystemInformation::{
    GlobalMemoryStatus, MEMORYSTATUS
};
        
        unsafe {
            // Initialiser la structure
            let mut mem_info: MEMORYSTATUSEX = std::mem::zeroed();
            mem_info.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            
            // Récupérer les informations mémoire
            if GlobalMemoryStatusEx(&mut mem_info) != 0 {
                // Retourner le pourcentage d'utilisation
                mem_info.dwMemoryLoad as f64
            } else {
                // En cas d'erreur, retourner une valeur par défaut
                65.0
            }
        }
    }
    
    /// Version portable pour obtenir l'utilisation mémoire
    #[cfg(not(target_os = "windows"))]
    fn get_system_memory_usage(&self) -> f64 {
        // Simulation d'une utilisation mémoire aléatoire entre 40% et 80%
        40.0 + rand::thread_rng().gen::<f64>() * 40.0
    }
    
    /// Obtient le niveau d'activité neurale
    fn get_neural_activity(&self) -> f64 {
        // Récupérer l'activité des régions cérébrales
        let brain_activity = self.cortical_hub.get_brain_activity();
        
        if brain_activity.is_empty() {
            return 0.5; // Valeur par défaut
        }
        
        // Calculer la moyenne d'activité
        let total_activity: f64 = brain_activity.values().sum();
        total_activity / brain_activity.len() as f64
    }
    
    /// Obtient l'équilibre hormonal
    fn get_hormone_balance(&self) -> f64 {
        // Récupérer les niveaux des hormones principales
        let dopamine = self.hormonal_system.get_hormone_level(&HormoneType::Dopamine);
        let serotonin = self.hormonal_system.get_hormone_level(&HormoneType::Serotonin);
        let cortisol = self.hormonal_system.get_hormone_level(&HormoneType::Cortisol);
        let oxytocin = self.hormonal_system.get_hormone_level(&HormoneType::Oxytocin);
        
        // Calculer un indice d'équilibre (1.0 = parfaitement équilibré)
        let optimal_dopamine = 0.6;
        let optimal_serotonin = 0.7;
        let optimal_cortisol = 0.4;
        let optimal_oxytocin = 0.5;
        
        // Calculer les écarts par rapport aux valeurs optimales
        let dopamine_diff = (dopamine - optimal_dopamine).abs();
        let serotonin_diff = (serotonin - optimal_serotonin).abs();
        let cortisol_diff = (cortisol - optimal_cortisol).abs();
        let oxytocin_diff = (oxytocin - optimal_oxytocin).abs();
        
        // Moyenne des écarts, transformée en indice d'équilibre (0 = déséquilibré, 1 = équilibré)
        1.0 - ((dopamine_diff + serotonin_diff + cortisol_diff + oxytocin_diff) / 4.0)
    }
    
    /// Obtient le niveau de conscience
    fn get_consciousness_level(&self) -> f64 {
        // Récupérer les statistiques de conscience
        let stats = self.consciousness.get_stats();
        stats.consciousness_level
    }
}

/// Intégration du système d'autorégulation
pub mod integration {
    use super::*;
    use crate::neuralchain_core::quantum_organism::QuantumOrganism;
    use crate::neuralchain_core::cortical_hub::CorticalHub;
    use crate::neuralchain_core::hormonal_field::HormonalField;
    use crate::neuralchain_core::emergent_consciousness::ConsciousnessEngine;
    use crate::neuralchain_core::bios_time::BiosTime;
    
    /// Intègre le système d'autorégulation à un organisme
    pub fn integrate_autoregulation(
        organism: Arc<QuantumOrganism>,
        cortical_hub: Arc<CorticalHub>,
        hormonal_system: Arc<HormonalField>,
        consciousness: Arc<ConsciousnessEngine>,
        bios_clock: Arc<BiosTime>,
    ) -> Arc<Autoregulation> {
        // Créer le système d'autorégulation
        let autoregulation = Arc::new(Autoregulation::new(
            organism.clone(),
            cortical_hub.clone(),
            hormonal_system.clone(),
            consciousness.clone(),
            bios_clock.clone(),
        ));
        
        // Démarrer le cycle d'autorégulation
        if let Err(e) = autoregulation.start_regulation() {
            println!("Erreur au démarrage de l'autorégulation: {}", e);
        }
        
        autoregulation
    }
}

/// Extensions pour le cortex
trait CorticalExtensions {
    fn optimize_neural_pathways(&self);
}

impl CorticalExtensions for CorticalHub {
    fn optimize_neural_pathways(&self) {
        // Dans une implémentation réelle, cette méthode réorganiserait
        // les chemins neuronaux pour optimiser les performances
    }
}
