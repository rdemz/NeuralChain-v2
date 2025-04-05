//! Module adaptive_reward.rs - Système de récompense adaptatif à modulation lente
//! Inspiré des synapses dopaminergiques du système limbique avec auto-évolution

use std::sync::{Arc, Mutex, RwLock};
use std::collections::{VecDeque, HashMap};
use std::time::{Duration, Instant};
use log::{debug, info, warn, error};
use rand::{thread_rng, Rng};

// Caractéristiques d'un système vivant
const BASELINE_REWARD: u64 = 100_000_000;            // Récompense de référence
const MEMORY_CAPACITY: usize = 1000;                 // Capacité de mémoire synaptique
const DOPAMINE_REPLENISHMENT_RATE: f64 = 0.005;      // Taux de régénération de dopamine
const PLASTICITY_THRESHOLD: f64 = 0.15;              // Seuil de plasticité synaptique
const EVOLUTION_RATE: f64 = 0.0001;                  // Taux d'évolution naturelle

/// Représentation d'un neurotransmetteur dans le système de récompense
#[derive(Clone, Debug)]
pub enum Neurotransmitter {
    /// Dopamine - récompense et motivation
    Dopamine(f64),
    /// Sérotonine - bien-être et stabilité
    Serotonin(f64),
    /// Noradrénaline - attention et vigilance
    Noradrenaline(f64),
    /// GABA - inhibition et régulation
    GABA(f64),
}

/// Type de récompense et son effet biologique
#[derive(Clone, Debug)]
pub enum RewardType {
    /// Récompense standard pour minage
    Mining,
    /// Récompense pour validation de transactions
    Validation,
    /// Récompense pour maintien de réseau
    Maintenance,
    /// Récompense pour comportement innovant
    Innovation,
    /// Récompense pour collaboration avec d'autres nœuds
    Collaboration,
}

/// Événement de récompense avec contexte biologique
#[derive(Clone, Debug)]
pub struct RewardEvent {
    /// Type de récompense
    pub reward_type: RewardType,
    /// Quantité de crypto accordée
    pub amount: u64,
    /// Identifiant du destinataire
    pub recipient: Vec<u8>,
    /// Score de réputation du destinataire
    pub reputation_score: f64,
    /// Difficulté/complexité de l'action récompensée
    pub task_difficulty: f64,
    /// Contexte de la récompense (métadonnées)
    pub context: HashMap<String, String>,
    /// Horodatage
    pub timestamp: Instant,
}

/// Système de récompense adaptatif vivant
pub struct AdaptiveReward {
    // État interne vivant
    dopamine_pool: Arc<RwLock<f64>>,               // Réservoir de dopamine disponible
    serotonin_level: Arc<RwLock<f64>>,             // Niveau de sérotonine (stabilité)
    noradrenaline_level: Arc<RwLock<f64>>,         // Niveau de noradrénaline (vigilance)
    gaba_level: Arc<RwLock<f64>>,                  // Niveau de GABA (inhibition)
    
    // Plasticité et mémoire synaptique
    reward_memory: Arc<Mutex<VecDeque<RewardEvent>>>, // Mémoire des récompenses
    reward_patterns: Arc<RwLock<HashMap<String, f64>>>, // Patterns de récompense reconnus
    plasticity_factor: f64,                         // Facteur de plasticité synaptique
    
    // Adaptabilité et évolution
    evolution_counter: u64,                         // Compteur d'évolution
    mutation_probability: f64,                      // Probabilité de mutation
    adaptation_history: Vec<(String, f64, Instant)>, // Historique des adaptations
    
    // Auto-régulation
    reward_ceiling: Arc<RwLock<u64>>,               // Plafond de récompense dynamique
    reward_floor: Arc<RwLock<u64>>,                 // Plancher de récompense dynamique
    fatigue_state: Arc<RwLock<f64>>,                // État de fatigue synaptique
    recovery_rate: f64,                             // Taux de récupération
    
    // Interconnexions synaptiques
    neural_feedback: Arc<Mutex<f64>>,              // Rétroaction vers neural_pow
    metabolism_link: Option<Arc<Mutex<f64>>>,      // Lien vers le métabolisme
    reputation_link: Option<Arc<RwLock<f64>>>,     // Lien vers le système de réputation
    
    // Compteurs vitaux
    birth_time: Instant,                           // Moment de création
    total_rewards_issued: u64,                     // Total des récompenses émises
    last_adaptation_time: Instant,                 // Dernier moment d'adaptation
}

impl AdaptiveReward {
    /// Crée un nouveau système de récompense vivant
    pub fn new() -> Self {
        let reward_memory = VecDeque::with_capacity(MEMORY_CAPACITY);
        let reward_patterns = HashMap::new();
        
        let instance = Self {
            dopamine_pool: Arc::new(RwLock::new(1.0)),
            serotonin_level: Arc::new(RwLock::new(0.5)),
            noradrenaline_level: Arc::new(RwLock::new(0.3)),
            gaba_level: Arc::new(RwLock::new(0.4)),
            
            reward_memory: Arc::new(Mutex::new(reward_memory)),
            reward_patterns: Arc::new(RwLock::new(reward_patterns)),
            plasticity_factor: 0.1,
            
            evolution_counter: 0,
            mutation_probability: 0.01,
            adaptation_history: Vec::with_capacity(100),
            
            reward_ceiling: Arc::new(RwLock::new(BASELINE_REWARD * 3)),
            reward_floor: Arc::new(RwLock::new(BASELINE_REWARD / 10)),
            fatigue_state: Arc::new(RwLock::new(0.0)),
            recovery_rate: 0.02,
            
            neural_feedback: Arc::new(Mutex::new(1.0)),
            metabolism_link: None,
            reputation_link: None,
            
            birth_time: Instant::now(),
            total_rewards_issued: 0,
            last_adaptation_time: Instant::now(),
        };
        
        // Démarrer le cycle de vie autonome du système de récompense
        instance.start_reward_lifecycle();
        
        info!("Système de récompense adaptatif créé à {:?}", instance.birth_time);
        instance
    }
    
    /// Démarre le cycle de vie autonome du système de récompense
    fn start_reward_lifecycle(&self) {
        // Cloner les références pour le thread
        let dopamine_pool = Arc::clone(&self.dopamine_pool);
        let serotonin_level = Arc::clone(&self.serotonin_level);
        let noradrenaline_level = Arc::clone(&self.noradrenaline_level);
        let gaba_level = Arc::clone(&self.gaba_level);
        let fatigue_state = Arc::clone(&self.fatigue_state);
        let reward_memory = Arc::clone(&self.reward_memory);
        let reward_patterns = Arc::clone(&self.reward_patterns);
        let recovery_rate = self.recovery_rate;
        
        // Démarrer un thread autonome qui maintient le système de récompense
        std::thread::spawn(move || {
            info!("Cycle de vie du système de récompense démarré");
            let lifecycle_cycle = Duration::from_millis(200); // 5 Hz
            
            loop {
                // Reconstituer progressivement la dopamine
                if let Ok(mut dopamine) = dopamine_pool.write() {
                    *dopamine += DOPAMINE_REPLENISHMENT_RATE;
                    *dopamine = dopamine.min(1.0);
                }
                
                // Récupération de la fatigue synaptique
                if let Ok(mut fatigue) = fatigue_state.write() {
                    if *fatigue > 0.0 {
                        *fatigue -= recovery_rate * lifecycle_cycle.as_secs_f64();
                        *fatigue = fatigue.max(0.0);
                    }
                }
                
                // Auto-régulation des neurotransmetteurs
                if let Ok(mut serotonin) = serotonin_level.write() {
                    // La sérotonine tend naturellement vers l'équilibre
                    if *serotonin < 0.5 {
                        *serotonin += 0.001;
                    } else if *serotonin > 0.5 {
                        *serotonin -= 0.001;
                    }
                }
                
                if let Ok(mut noradrenaline) = noradrenaline_level.write() {
                    // La noradrénaline diminue naturellement avec le temps
                    if *noradrenaline > 0.1 {
                        *noradrenaline -= 0.002;
                    }
                }
                
                if let Ok(mut gaba) = gaba_level.write() {
                    // Le GABA augmente progressivement avec le temps (effet inhibiteur croissant)
                    if *gaba < 0.6 {
                        *gaba += 0.0005;
                    }
                }
                
                // Analyse périodique des patterns de récompense
                if thread_rng().gen_bool(0.05) { // 5% de chance par cycle
                    Self::analyze_reward_patterns(&reward_memory, &reward_patterns);
                }
                
                // Pause pour réduire l'utilisation du CPU
                std::thread::sleep(lifecycle_cycle);
            }
        });
    }
    
    /// Analyse des patterns de récompense pour adaptation
    fn analyze_reward_patterns(
        reward_memory: &Arc<Mutex<VecDeque<RewardEvent>>>,
        reward_patterns: &Arc<RwLock<HashMap<String, f64>>>
    ) {
        // Accès à la mémoire des récompenses
        if let Ok(memory) = reward_memory.lock() {
            if memory.len() < 10 {
                return; // Pas assez de données
            }
            
            // Analyser les tendances
            let mut mining_rewards = 0;
            let mut validation_rewards = 0;
            let mut other_rewards = 0;
            
            for event in memory.iter() {
                match event.reward_type {
                    RewardType::Mining => mining_rewards += 1,
                    RewardType::Validation => validation_rewards += 1,
                    _ => other_rewards += 1,
                }
            }
            
            // Déterminer les patterns dominants
            let total = mining_rewards + validation_rewards + other_rewards;
            if total > 0 {
                if let Ok(mut patterns) = reward_patterns.write() {
                    patterns.insert("mining_ratio".to_string(), 
                                   mining_rewards as f64 / total as f64);
                    patterns.insert("validation_ratio".to_string(), 
                                   validation_rewards as f64 / total as f64);
                    patterns.insert("other_ratio".to_string(), 
                                   other_rewards as f64 / total as f64);
                }
            }
        }
    }
    
    /// Calcule une récompense adaptative pour une action
    pub fn calculate_reward(&mut self, 
                           reward_type: RewardType, 
                           recipient_reputation: f64,
                           task_difficulty: f64) -> u64 {
        // Vérifier si la dopamine est suffisante
        let dopamine_available = match self.dopamine_pool.read() {
            Ok(d) => *d,
            Err(_) => 0.5, // Valeur par défaut
        };
        
        if dopamine_available < 0.1 {
            warn!("Niveau de dopamine critique ({:.2}), récompense minimale", dopamine_available);
            return match self.reward_floor.read() {
                Ok(floor) => *floor,
                Err(_) => BASELINE_REWARD / 10,
            };
        }
        
        // Récupérer l'état de fatigue
        let fatigue = match self.fatigue_state.read() {
            Ok(f) => *f,
            Err(_) => 0.0,
        };
        
        // Facteur de base selon le type
        let base_factor = match reward_type {
            RewardType::Mining => 1.0,
            RewardType::Validation => 0.7,
            RewardType::Maintenance => 0.5,
            RewardType::Innovation => 1.5,
            RewardType::Collaboration => 1.2,
        };
        
        // Moduler avec la réputation
        let reputation_factor = 0.5 + (0.5 * recipient_reputation);
        
        // Moduler avec la difficulté de la tâche
        let difficulty_factor = 0.8 + (0.4 * task_difficulty);
        
        // Appliquer les niveaux de neurotransmetteurs
        let neurotransmitter_factor = match (self.noradrenaline_level.read(), self.serotonin_level.read(), self.gaba_level.read()) {
            (Ok(na), Ok(s), Ok(g)) => {
                // Formule biochimique complexe qui simule l'interaction des neurotransmetteurs
                0.7 + (*na * 0.3) + (*s * 0.2) - (*g * 0.1)
            },
            _ => 1.0, // Valeur par défaut
        };
        
        // Intégrer la fatigue synaptique
        let fatigue_factor = 1.0 - (fatigue * 0.5);
        
        // Calcul final
        let mut raw_reward = (BASELINE_REWARD as f64 * 
                             base_factor * 
                             reputation_factor * 
                             difficulty_factor * 
                             neurotransmitter_factor *
                             fatigue_factor) as u64;
        
        // Appliquer l'évolution naturelle (légère randomisation biologique)
        if thread_rng().gen_bool(self.mutation_probability as f64) {
            let mutation_factor = thread_rng().gen_range(0.9..1.1);
            raw_reward = (raw_reward as f64 * mutation_factor) as u64;
            
            // Enregistrer cette mutation dans l'historique
            self.adaptation_history.push((
                format!("reward_mutation_{}", self.evolution_counter),
                mutation_factor,
                Instant::now()
            ));
            self.evolution_counter += 1;
        }
        
        // Limiter aux bornes
        let ceiling = match self.reward_ceiling.read() {
            Ok(c) => *c,
            Err(_) => BASELINE_REWARD * 3,
        };
        
        let floor = match self.reward_floor.read() {
            Ok(f) => *f,
            Err(_) => BASELINE_REWARD / 10,
        };
        
        let final_reward = raw_reward.max(floor).min(ceiling);
        
        // Consommer de la dopamine pour cette récompense
        if let Ok(mut dopamine) = self.dopamine_pool.write() {
            let consumption = (final_reward as f64 / BASELINE_REWARD as f64) * 0.05;
            *dopamine -= consumption;
            *dopamine = dopamine.max(0.0);
        }
        
        // Augmenter la fatigue synaptique
        if let Ok(mut fatigue) = self.fatigue_state.write() {
            *fatigue += 0.01;
            *fatigue = fatigue.min(1.0);
        }
        
        // Créer l'événement de récompense
        let event = RewardEvent {
            reward_type: reward_type.clone(),
            amount: final_reward,
            recipient: Vec::new(), // À remplir par l'appelant
            reputation_score: recipient_reputation,
            task_difficulty,
            context: HashMap::new(),
            timestamp: Instant::now(),
        };
        
        // Stocker dans la mémoire
        if let Ok(mut memory) = self.reward_memory.lock() {
            memory.push_back(event);
            if memory.len() > MEMORY_CAPACITY {
                memory.pop_front();
            }
        }
        
        // Mettre à jour le compteur
        self.total_rewards_issued += 1;
        
        // Déclencher une adaptation si nécessaire
        if self.last_adaptation_time.elapsed() > Duration::from_secs(3600) { // 1 heure
            self.adapt_to_environment();
        }
        
        info!("Récompense calculée: {} (type: {:?}, réputation: {:.2}, difficulté: {:.2})",
             final_reward, reward_type, recipient_reputation, task_difficulty);
        
        final_reward
    }
    
    /// S'adapte à l'environnement en fonction des données historiques
    fn adapt_to_environment(&mut self) {
        self.last_adaptation_time = Instant::now();
        
        // Analyser toutes les récompenses récentes
        let mut avg_reward = 0.0;
        let mut count = 0;
        
        if let Ok(memory) = self.reward_memory.lock() {
            for event in memory.iter().rev().take(100) {
                avg_reward += event.amount as f64;
                count += 1;
            }
        }
        
        if count == 0 {
            return;
        }
        
        avg_reward /= count as f64;
        
        // Adapter le plafond et le plancher en fonction de la moyenne
        if let Ok(mut ceiling) = self.reward_ceiling.write() {
            *ceiling = (avg_reward * 3.0) as u64;
        }
        
        if let Ok(mut floor) = self.reward_floor.write() {
            *floor = (avg_reward * 0.1) as u64;
        }
        
        // Adapter la plasticité
        let old_plasticity = self.plasticity_factor;
        if avg_reward > BASELINE_REWARD as f64 * 1.5 {
            // Si les récompenses sont élevées, diminuer la plasticité
            self.plasticity_factor *= 0.9;
        } else if avg_reward < BASELINE_REWARD as f64 * 0.5 {
            // Si les récompenses sont faibles, augmenter la plasticité
            self.plasticity_factor *= 1.1;
        }
        
        // Limiter la plasticité
        self.plasticity_factor = self.plasticity_factor.max(0.01).min(0.5);
        
        // Enregistrer cette adaptation
        if (self.plasticity_factor - old_plasticity).abs() > PLASTICITY_THRESHOLD {
            self.adaptation_history.push((
                "plasticity_adaptation".to_string(),
                self.plasticity_factor / old_plasticity,
                Instant::now()
            ));
        }
        
        info!("Adaptation environnementale: nouveau plafond {}, nouveau plancher {}, plasticité {:.2}",
             self.reward_ceiling.read().unwrap_or(&0), 
             self.reward_floor.read().unwrap_or(&0), 
             self.plasticity_factor);
    }
    
    /// Libère un neurotransmetteur spécifique
    pub fn release_neurotransmitter(&self, neurotransmitter: Neurotransmitter) -> bool {
        match neurotransmitter {
            Neurotransmitter::Dopamine(amount) => {
                if let Ok(mut level) = self.dopamine_pool.write() {
                    *level += amount;
                    *level = level.min(1.0);
                    true
                } else {
                    false
                }
            },
            Neurotransmitter::Serotonin(amount) => {
                if let Ok(mut level) = self.serotonin_level.write() {
                    *level += amount;
                    *level = level.min(1.0);
                    true
                } else {
                    false
                }
            },
            Neurotransmitter::Noradrenaline(amount) => {
                if let Ok(mut level) = self.noradrenaline_level.write() {
                    *level += amount;
                    *level = level.min(1.0);
                    true
                } else {
                    false
                }
            },
            Neurotransmitter::GABA(amount) => {
                if let Ok(mut level) = self.gaba_level.write() {
                    *level += amount;
                    *level = level.min(1.0);
                    true
                } else {
                    false
                }
            },
        }
    }
    
    /// Connecte ce système de récompense au système neuronal
    pub fn connect_to_neural(&mut self, neural_feedback: Arc<Mutex<f64>>) {
        self.neural_feedback = neural_feedback;
        info!("Connexion établie avec le système neuronal");
    }
    
    /// Récupère l'état actuel du système de récompense
    pub fn get_reward_state(&self) -> RewardSystemState {
        RewardSystemState {
            dopamine_level: *self.dopamine_pool.read().unwrap_or(&0.0),
            serotonin_level: *self.serotonin_level.read().unwrap_or(&0.0),
            noradrenaline_level: *self.noradrenaline_level.read().unwrap_or(&0.0),
            gaba_level: *self.gaba_level.read().unwrap_or(&0.0),
            fatigue_state: *self.fatigue_state.read().unwrap_or(&0.0),
            reward_ceiling: *self.reward_ceiling.read().unwrap_or(&0),
            reward_floor: *self.reward_floor.read().unwrap_or(&0),
            plasticity_factor: self.plasticity_factor,
            total_rewards_issued: self.total_rewards_issued,
            age_seconds: self.birth_time.elapsed().as_secs(),
        }
    }
}

/// État actuel du système de récompense
#[derive(Debug)]
pub struct RewardSystemState {
    pub dopamine_level: f64,
    pub serotonin_level: f64,
    pub noradrenaline_level: f64,
    pub gaba_level: f64,
    pub fatigue_state: f64,
    pub reward_ceiling: u64,
    pub reward_floor: u64,
    pub plasticity_factor: f64,
    pub total_rewards_issued: u64,
    pub age_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reward_system_creation() {
        let reward_system = AdaptiveReward::new();
        let state = reward_system.get_reward_state();
        
        assert!(state.dopamine_level > 0.5); // Devrait commencer avec de la dopamine
        assert_eq!(state.total_rewards_issued, 0);
    }
    
    #[test]
    fn test_reward_calculation() {
        let mut reward_system = AdaptiveReward::new();
        
        let reward1 = reward_system.calculate_reward(
            RewardType::Mining,
            0.8, // Haute réputation
            0.5  // Difficulté moyenne
        );
        
        let reward2 = reward_system.calculate_reward(
            RewardType::Validation,
            0.8, // Même réputation
            0.5  // Même difficulté
        );
        
        // La récompense pour le minage devrait être plus élevée
        assert!(reward1 > reward2);
        
        // Le nombre total devrait être 2
        assert_eq!(reward_system.total_rewards_issued, 2);
    }
    
    #[test]
    fn test_neurotransmitter_release() {
        let reward_system = AdaptiveReward::new();
        
        // Niveau initial de dopamine
        let initial_dopamine = reward_system.get_reward_state().dopamine_level;
        
        // Libérer de la dopamine
        let result = reward_system.release_neurotransmitter(Neurotransmitter::Dopamine(0.2));
        assert!(result);
        
        // Vérifier que le niveau a augmenté
        let new_dopamine = reward_system.get_reward_state().dopamine_level;
        assert!(new_dopamine > initial_dopamine);
    }
}
