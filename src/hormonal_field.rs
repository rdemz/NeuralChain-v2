//! Module hormonal_field.rs - Système de signalisation biochimique distribué
//! Inspiré du système endocrinien humain, permettant la communication à longue
//! distance et la régulation globale via des "hormones" numériques qui influencent
//! le comportement de l'ensemble de la blockchain.

use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, BTreeMap, HashSet};
use std::time::{Duration, Instant};
use log::{debug, info, warn, error};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use dashmap::DashMap; // Collections concurrentes optimisées
use parking_lot::RwLock as PLRwLock; // Verrous optimisés pour Windows

// Constantes d'optimisation hormonale
const HALF_LIFE_BASE: f64 = 1.5;              // Base pour calcul de demi-vie exponentielle
const MAX_LOCAL_HORMONE_BUFFER: usize = 1000;  // Limite d'entrées locales par hormone
const HORMONE_TICK_RATE_MS: u64 = 100;         // Fréquence de mise à jour (ms)
const DEFAULT_DIFFUSION_RATE: f64 = 0.05;      // Taux de diffusion par défaut
const MAX_RECEPTOR_SENSITIVITY: f64 = 5.0;     // Sensibilité maximale des récepteurs

/// Types d'hormones supportés par le système
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HormoneType {
    /// Cortisol - hormone de stress
    Cortisol,
    /// Adrénaline - réponse d'urgence
    Adrenaline,
    /// Dopamine - récompense
    Dopamine,
    /// Sérotonine - stabilité et satisfaction
    Serotonin,
    /// Mélatonine - cycles de sommeil
    Melatonin,
    /// Ocytocine - coopération
    Oxytocin,
    /// Endorphine - bien-être et réduction de douleur
    Endorphin,
    /// Testostérone - agressivité et compétition
    Testosterone,
    /// Insuline - régulation énergétique
    Insulin,
    /// Hormone synthétique personnalisée
    Custom(String),
}

/// Mode d'action d'un récepteur hormonal
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceptorAction {
    /// Augmenter une valeur proportionnellement
    Increase,
    /// Diminuer une valeur proportionnellement
    Decrease,
    /// Action inversée de seuil (active si inférieur au seuil)
    ThresholdLow,
    /// Action de seuil (active si supérieur au seuil)
    ThresholdHigh,
    /// Moduler une fréquence
    ModulateFrequency,
    /// Moduler une amplitude
    ModulateAmplitude,
    /// Activer/désactiver un comportement
    Toggle,
    /// Action personnalisée
    Custom(String),
}

/// Récepteur hormonal attaché à un module
#[derive(Debug, Clone)]
pub struct HormoneReceptor {
    /// Identifiant unique
    pub id: String,
    /// Module hôte contenant ce récepteur
    pub host_module: String,
    /// Type d'hormone reconnue
    pub hormone_type: HormoneType,
    /// Sensibilité du récepteur (amplification du signal)
    pub sensitivity: f64,
    /// Seuil d'activation
    pub threshold: f64,
    /// Action déclenchée
    pub action: ReceptorAction,
    /// Paramètre cible affecté
    pub target_parameter: String,
    /// Horodatage de création
    pub creation_time: Instant,
    /// Dernière activation
    pub last_activation: Option<Instant>,
    /// Métadonnées
    pub metadata: HashMap<String, Vec<u8>>,
}

/// Signal hormonal distribué dans le système
#[derive(Debug, Clone)]
pub struct HormoneSignal {
    /// Type d'hormone
    pub hormone_type: HormoneType,
    /// Identifiant unique de l'émission
    pub signal_id: String,
    /// Module source émetteur
    pub source_module: String,
    /// Intensité initiale
    pub initial_intensity: f64,
    /// Intensité actuelle
    pub current_intensity: f64,
    /// Horodatage d'émission
    pub emission_time: Instant,
    /// Demi-vie (durée pour que l'intensité soit réduite de moitié)
    pub half_life: Duration,
    /// Rayon de diffusion (0.0-1.0, 1.0 = système entier)
    pub diffusion_radius: f64,
    /// Contexte d'émission
    pub context: HashMap<String, Vec<u8>>,
}

/// Événement de réaction hormonale
#[derive(Debug, Clone)]
pub struct HormoneReaction {
    /// Récepteur activé
    pub receptor_id: String,
    /// Module hôte
    pub host_module: String,
    /// Type d'hormone
    pub hormone_type: HormoneType,
    /// Intensité du signal
    pub signal_intensity: f64,
    /// Intensité de la réaction
    pub reaction_intensity: f64,
    /// Action déclenchée
    pub action: ReceptorAction,
    /// Paramètre affecté
    pub target_parameter: String,
    /// Horodatage de réaction
    pub reaction_time: Instant,
}

/// État du champ hormonal dans une région
#[derive(Debug, Clone)]
pub struct HormoneFieldState {
    /// Niveaux de chaque hormone
    pub hormone_levels: HashMap<HormoneType, f64>,
    /// Gradients/variations
    pub gradients: HashMap<HormoneType, f64>,
    /// Signaux actifs
    pub active_signals_count: HashMap<HormoneType, usize>,
    /// Horodatage de la mesure
    pub measurement_time: Instant,
}

/// Structure principale du champ hormonal
pub struct HormonalField {
    // État du système
    active: Arc<RwLock<bool>>,
    global_field_strength: Arc<RwLock<f64>>,
    
    // Collection de signaux actifs
    // Utilise DashMap pour une haute concurrence avec faible verrouillage
    active_signals: Arc<DashMap<String, HormoneSignal>>,
    
    // Niveaux hormonaux globaux et locaux, optimisés pour lectures fréquentes
    global_hormone_levels: Arc<PLRwLock<HashMap<HormoneType, f64>>>,
    local_hormone_buffers: Arc<DashMap<String, VecDeque<(HormoneType, f64, Instant)>>>,
    
    // Récepteurs et leurs réactions
    hormone_receptors: Arc<RwLock<HashMap<String, HormoneReceptor>>>,
    receptor_by_module: Arc<DashMap<String, Vec<String>>>,
    recent_reactions: Arc<Mutex<VecDeque<HormoneReaction>>>,
    
    // Callbacks pour effets hormonaux
    receptor_callbacks: Arc<DashMap<String, Box<dyn Fn(&HormoneReaction) -> bool + Send + Sync>>>,
    
    // Constantes du système et historique
    diffusion_rates: Arc<DashMap<HormoneType, f64>>,
    half_lives: Arc<RwLock<HashMap<HormoneType, Duration>>>,
    hormone_history: Arc<Mutex<HashMap<HormoneType, VecDeque<(f64, Instant)>>>>,
    
    // Paramètres de synchronisation
    last_global_sync: Arc<RwLock<Instant>>,
    last_cleanup: Arc<RwLock<Instant>>,
    
    // Statistiques
    total_signals_emitted: Arc<PLRwLock<u64>>,
    total_reactions: Arc<PLRwLock<u64>>,
    
    // Démarrage du système
    system_birth_time: Instant,
}

impl HormonalField {
    /// Crée une nouvelle instance du champ hormonal
    pub fn new() -> Self {
        // Initialisation des niveaux hormonaux par défaut
        let mut default_levels = HashMap::new();
        default_levels.insert(HormoneType::Cortisol, 0.2);
        default_levels.insert(HormoneType::Adrenaline, 0.1);
        default_levels.insert(HormoneType::Dopamine, 0.3);
        default_levels.insert(HormoneType::Serotonin, 0.4);
        default_levels.insert(HormoneType::Melatonin, 0.1);
        default_levels.insert(HormoneType::Oxytocin, 0.2);
        default_levels.insert(HormoneType::Endorphin, 0.3);
        default_levels.insert(HormoneType::Testosterone, 0.2);
        default_levels.insert(HormoneType::Insulin, 0.5);
        
        // Initialisation des taux de diffusion
        let diffusion_rates = Arc::new(DashMap::new());
        diffusion_rates.insert(HormoneType::Cortisol, 0.08);
        diffusion_rates.insert(HormoneType::Adrenaline, 0.12);
        diffusion_rates.insert(HormoneType::Dopamine, 0.05);
        diffusion_rates.insert(HormoneType::Serotonin, 0.04);
        diffusion_rates.insert(HormoneType::Melatonin, 0.03);
        diffusion_rates.insert(HormoneType::Oxytocin, 0.06);
        diffusion_rates.insert(HormoneType::Endorphin, 0.07);
        diffusion_rates.insert(HormoneType::Testosterone, 0.04);
        diffusion_rates.insert(HormoneType::Insulin, 0.09);
        
        // Initialisation des demi-vies (en secondes)
        let mut half_lives = HashMap::new();
        half_lives.insert(HormoneType::Cortisol, Duration::from_secs(600));     // 10 minutes
        half_lives.insert(HormoneType::Adrenaline, Duration::from_secs(60));    // 1 minute
        half_lives.insert(HormoneType::Dopamine, Duration::from_secs(120));     // 2 minutes
        half_lives.insert(HormoneType::Serotonin, Duration::from_secs(1800));   // 30 minutes
        half_lives.insert(HormoneType::Melatonin, Duration::from_secs(7200));   // 2 heures
        half_lives.insert(HormoneType::Oxytocin, Duration::from_secs(180));     // 3 minutes
        half_lives.insert(HormoneType::Endorphin, Duration::from_secs(300));    // 5 minutes
        half_lives.insert(HormoneType::Testosterone, Duration::from_secs(3600)); // 1 heure
        half_lives.insert(HormoneType::Insulin, Duration::from_secs(240));      // 4 minutes
        
        let instance = Self {
            active: Arc::new(RwLock::new(true)),
            global_field_strength: Arc::new(RwLock::new(1.0)),
            
            active_signals: Arc::new(DashMap::with_capacity(1000)),
            
            global_hormone_levels: Arc::new(PLRwLock::new(default_levels)),
            local_hormone_buffers: Arc::new(DashMap::with_capacity(100)),
            
            hormone_receptors: Arc::new(RwLock::new(HashMap::with_capacity(100))),
            receptor_by_module: Arc::new(DashMap::with_capacity(100)),
            recent_reactions: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            
            receptor_callbacks: Arc::new(DashMap::with_capacity(100)),
            
            diffusion_rates,
            half_lives: Arc::new(RwLock::new(half_lives)),
            hormone_history: Arc::new(Mutex::new(HashMap::new())),
            
            last_global_sync: Arc::new(RwLock::new(Instant::now())),
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
            
            total_signals_emitted: Arc::new(PLRwLock::new(0)),
            total_reactions: Arc::new(PLRwLock::new(0)),
            
            system_birth_time: Instant::now(),
        };
        
        // Démarrer le système hormonal
        instance.start_hormonal_system();
        
        info!("Champ hormonal créé à {:?}", instance.system_birth_time);
        instance
    }
    
    /// Démarre le système hormonal dans un thread autonome
    fn start_hormonal_system(&self) {
        // Cloner les références nécessaires
        let active = Arc::clone(&self.active);
        let active_signals = Arc::clone(&self.active_signals);
        let global_hormone_levels = Arc::clone(&self.global_hormone_levels);
        let local_hormone_buffers = Arc::clone(&self.local_hormone_buffers);
        let hormone_receptors = Arc::clone(&self.hormone_receptors);
        let receptor_by_module = Arc::clone(&self.receptor_by_module);
        let recent_reactions = Arc::clone(&self.recent_reactions);
        let receptor_callbacks = Arc::clone(&self.receptor_callbacks);
        let diffusion_rates = Arc::clone(&self.diffusion_rates);
        let half_lives = Arc::clone(&self.half_lives);
        let hormone_history = Arc::clone(&self.hormone_history);
        let last_global_sync = Arc::clone(&self.last_global_sync);
        let last_cleanup = Arc::clone(&self.last_cleanup);
        let total_reactions = Arc::clone(&self.total_reactions);
        
        // Démarrer le thread du système hormonal
        std::thread::spawn(move || {
            info!("Système hormonal démarré - diffusion et régulation hormonale active");
           // Suite du code précédent...
            let tick_interval = Duration::from_millis(HORMONE_TICK_RATE_MS);
            let mut last_tick = Instant::now();
            
            // Optimisation pour Windows: création d'un pool de threads dédié
            #[cfg(target_os = "windows")]
            let thread_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_cpus::get_physical().min(8))
                .stack_size(3 * 1024 * 1024) // 3MB stack - optimisé pour Windows
                .thread_name(|i| format!("hormone_worker_{}", i))
                .build()
                .unwrap_or_else(|e| {
                    warn!("Échec de création du pool de threads optimisé: {}", e);
                    rayon::ThreadPoolBuilder::new().build().unwrap()
                });
            
            #[cfg(not(target_os = "windows"))]
            let thread_count = 4; // Valeur par défaut pour autres OS
            
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
                
                let now = Instant::now();
                let elapsed = now.duration_since(last_tick);
                last_tick = now;
                
                // 1. Mettre à jour tous les signaux hormonaux actifs et les niveaux globaux
                Self::update_hormone_signals(
                    &active_signals,
                    &global_hormone_levels,
                    &half_lives,
                    elapsed,
                );
                
                // 2. Synchroniser les niveaux hormonaux globaux avec les tampons locaux
                if let Ok(mut last_sync) = last_global_sync.write() {
                    if now.duration_since(*last_sync) > Duration::from_secs(1) {
                        *last_sync = now;
                        
                        Self::synchronize_hormone_levels(
                            &global_hormone_levels,
                            &local_hormone_buffers,
                            &diffusion_rates,
                        );
                    }
                }
                
                // 3. Traiter les réactions aux hormones (optimisé pour Windows)
                #[cfg(target_os = "windows")]
                Self::process_hormone_reactions_windows(
                    &thread_pool,
                    &hormone_receptors,
                    &receptor_by_module,
                    &global_hormone_levels,
                    &local_hormone_buffers,
                    &receptor_callbacks,
                    &recent_reactions,
                    &total_reactions,
                );
                
                #[cfg(not(target_os = "windows"))]
                Self::process_hormone_reactions(
                    thread_count,
                    &hormone_receptors,
                    &receptor_by_module,
                    &global_hormone_levels,
                    &local_hormone_buffers,
                    &receptor_callbacks,
                    &recent_reactions,
                    &total_reactions,
                );
                
                // 4. Mettre à jour l'historique des niveaux hormonaux
                if now.duration_since(last_tick) > Duration::from_secs(60) {
                    Self::update_hormone_history(
                        &global_hormone_levels,
                        &hormone_history,
                    );
                }
                
                // 5. Nettoyage périodique
                if let Ok(mut last_clean) = last_cleanup.write() {
                    if now.duration_since(*last_clean) > Duration::from_secs(300) { // 5 minutes
                        *last_clean = now;
                        
                        // Nettoyer les signaux expirés
                        active_signals.retain(|_, signal| {
                            signal.current_intensity > 0.01 && 
                            now.duration_since(signal.emission_time) < Duration::from_secs(3600)
                        });
                        
                        // Nettoyer les tampons locaux anciens
                        local_hormone_buffers.iter_mut().for_each(|mut entry| {
                            entry.retain(|(_, _, timestamp)| 
                                now.duration_since(*timestamp) < Duration::from_secs(3600));
                        });
                        
                        // Nettoyer l'historique des réactions
                        if let Ok(mut reactions) = recent_reactions.lock() {
                            reactions.retain(|r| now.duration_since(r.reaction_time) < Duration::from_secs(3600));
                        }
                    }
                }
                
                // Pause optimisée pour réduire l'utilisation CPU
                // Ajustement spécial pour Windows pour éviter le "thread thrashing"
                #[cfg(target_os = "windows")]
                {
                    let sleep_time = tick_interval.saturating_sub(Instant::now().duration_since(now));
                    if sleep_time > Duration::from_millis(1) {
                        std::thread::sleep(sleep_time);
                    } else {
                        // Yield brièvement pour permettre aux autres threads de s'exécuter
                        std::thread::yield_now();
                    }
                }
                
                #[cfg(not(target_os = "windows"))]
                {
                    std::thread::sleep(tick_interval);
                }
            }
        });
    }
    
    /// Met à jour les signaux hormonaux et les niveaux globaux
    fn update_hormone_signals(
        active_signals: &Arc<DashMap<String, HormoneSignal>>,
        global_hormone_levels: &Arc<PLRwLock<HashMap<HormoneType, f64>>>,
        half_lives: &Arc<RwLock<HashMap<HormoneType, Duration>>>,
        elapsed: Duration,
    ) {
        // Créer un tableau temporaire pour accumuler les changements d'intensité par hormone
        let mut hormone_intensity_changes: HashMap<HormoneType, f64> = HashMap::new();
        
        // Map pour suivre les signaux à supprimer
        let mut signals_to_remove = Vec::new();
        
        // Pré-charger les demi-vies pour éviter les verrous répétés
        let half_lives_map = match half_lives.read() {
            Ok(hl) => hl.clone(),
            Err(_) => HashMap::new(),
        };
        
        // Calculer l'intensité décroissante de tous les signaux actifs
        for mut signal_entry in active_signals.iter_mut() {
            let signal_id = signal_entry.key().clone();
            let signal = signal_entry.value_mut();
            
            // Calculer la diminution basée sur la demi-vie
            let half_life = half_lives_map.get(&signal.hormone_type)
                .cloned()
                .unwrap_or_else(|| Duration::from_secs(300));
            
            let decay_factor = HALF_LIFE_BASE.powf(-elapsed.as_secs_f64() / half_life.as_secs_f64());
            let old_intensity = signal.current_intensity;
            signal.current_intensity *= decay_factor;
            
            // Si l'intensité est trop faible, marquer pour suppression
            if signal.current_intensity < 0.01 {
                signals_to_remove.push(signal_id);
                continue;
            }
            
            // Calculer le changement d'intensité pour ce signal
            let intensity_change = signal.current_intensity - old_intensity;
            
            // Accumuler le changement d'intensité par type d'hormone
            *hormone_intensity_changes.entry(signal.hormone_type.clone()).or_insert(0.0) += intensity_change;
        }
        
        // Supprimer les signaux marqués
        for id in signals_to_remove {
            active_signals.remove(&id);
        }
        
        // Appliquer les changements d'intensité aux niveaux hormonaux globaux
        if !hormone_intensity_changes.is_empty() {
            if let Ok(mut global_levels) = global_hormone_levels.write() {
                for (hormone_type, intensity_change) in hormone_intensity_changes {
                    let current_level = global_levels.entry(hormone_type).or_insert(0.0);
                    *current_level += intensity_change;
                    *current_level = current_level.max(0.0).min(1.0);
                }
            }
        }
    }
    
    /// Synchronise les niveaux hormonaux globaux avec les tampons locaux
    fn synchronize_hormone_levels(
        global_hormone_levels: &Arc<PLRwLock<HashMap<HormoneType, f64>>>,
        local_hormone_buffers: &Arc<DashMap<String, VecDeque<(HormoneType, f64, Instant)>>>,
        diffusion_rates: &Arc<DashMap<HormoneType, f64>>,
    ) {
        // Lire les niveaux globaux
        let global_levels = match global_hormone_levels.read() {
            Ok(levels) => levels.clone(),
            Err(_) => return,
        };
        
        // Pour chaque tampon local, synchroniser avec les niveaux globaux
        for mut buffer_entry in local_hormone_buffers.iter_mut() {
            let mut changes = HashMap::<HormoneType, f64>::new();
            
            // Appliquer la diffusion pour chaque type d'hormone
            for (hormone_type, global_level) in &global_levels {
                // Récupérer le taux de diffusion
                let diffusion_rate = diffusion_rates.get(hormone_type)
                    .map(|rate| *rate)
                    .unwrap_or(DEFAULT_DIFFUSION_RATE);
                
                // Calculer le niveau local actuel
                let local_level = buffer_entry.iter()
                    .filter(|(h_type, _, _)| h_type == hormone_type)
                    .map(|(_, intensity, _)| *intensity)
                    .sum::<f64>();
                
                // Calculer la différence et appliquer la diffusion
                let diff = global_level - local_level;
                let change = diff * diffusion_rate;
                
                if change.abs() > 0.001 {
                    changes.insert(hormone_type.clone(), change);
                }
            }
            
            // Appliquer les changements au tampon local
            for (hormone_type, change) in changes {
                if change > 0.0 {
                    // Ajouter une nouvelle entrée pour l'augmentation
                    buffer_entry.push_back((hormone_type, change, Instant::now()));
                    
                    // Limiter la taille du tampon
                    while buffer_entry.len() > MAX_LOCAL_HORMONE_BUFFER {
                        buffer_entry.pop_front();
                    }
                } else {
                    // Pour une diminution, réduire les entrées existantes
                    let mut remaining_reduction = -change;
                    let mut indices_to_remove = Vec::new();
                    
                    for (i, (h_type, intensity, _)) in buffer_entry.iter().enumerate() {
                        if h_type == &hormone_type && remaining_reduction > 0.0 {
                            let reduction = remaining_reduction.min(*intensity);
                            remaining_reduction -= reduction;
                            
                            // Si l'intensité est complètement réduite, marquer pour suppression
                            if (*intensity - reduction).abs() < 0.001 {
                                indices_to_remove.push(i);
                            }
                        }
                    }
                    
                    // Supprimer les entrées marquées (dans l'ordre inverse pour éviter les décalages d'indices)
                    for i in indices_to_remove.into_iter().rev() {
                        if i < buffer_entry.len() {
                            buffer_entry.remove(i);
                        }
                    }
                }
            }
        }
    }
    
    /// Traite les réactions hormonales - version optimisée pour Windows avec rayon
    #[cfg(target_os = "windows")]
    fn process_hormone_reactions_windows(
        thread_pool: &rayon::ThreadPool,
        hormone_receptors: &Arc<RwLock<HashMap<String, HormoneReceptor>>>,
        receptor_by_module: &Arc<DashMap<String, Vec<String>>>,
        global_hormone_levels: &Arc<PLRwLock<HashMap<HormoneType, f64>>>,
        local_hormone_buffers: &Arc<DashMap<String, VecDeque<(HormoneType, f64, Instant)>>>,
        receptor_callbacks: &Arc<DashMap<String, Box<dyn Fn(&HormoneReaction) -> bool + Send + Sync>>>,
        recent_reactions: &Arc<Mutex<VecDeque<HormoneReaction>>>,
        total_reactions: &Arc<PLRwLock<u64>>,
    ) {
        let now = Instant::now();
        
        // Récupérer tous les récepteurs
        let receptors = match hormone_receptors.read() {
            Ok(r) => r.clone(),
            Err(_) => return,
        };
        
        // Récupérer les niveaux hormonaux globaux
        let global_levels = match global_hormone_levels.read() {
            Ok(levels) => levels.clone(),
            Err(_) => return,
        };
        
        // Utiliser rayon pour le traitement parallèle
        thread_pool.install(|| {
            let reactions: Vec<HormoneReaction> = receptors.par_iter()
                .filter_map(|(receptor_id, receptor)| {
                    // Vérifier si un niveau local est disponible
                    let buffer_entry = local_hormone_buffers.get(&receptor.host_module);
                    
                    // Calculer l'intensité hormonale effective (locale ou globale)
                    let hormone_intensity = if let Some(buffer) = buffer_entry {
                        // Niveau local
                        buffer.iter()
                            .filter(|(h_type, _, _)| h_type == &receptor.hormone_type)
                            .map(|(_, intensity, _)| *intensity)
                            .sum::<f64>()
                    } else {
                        // Niveau global
                        *global_levels.get(&receptor.hormone_type).unwrap_or(&0.0)
                    };
                    
                    // Vérifier si le récepteur doit réagir
                    let should_react = match receptor.action {
                        ReceptorAction::ThresholdLow => hormone_intensity < receptor.threshold,
                        ReceptorAction::ThresholdHigh => hormone_intensity > receptor.threshold,
                        _ => hormone_intensity > 0.1, // Seuil par défaut pour les autres actions
                    };
                    
                    if should_react {
                        // Calculer l'intensité de la réaction en fonction de la sensibilité
                        let reaction_intensity = hormone_intensity * receptor.sensitivity;
                        
                        // Créer un objet de réaction
                        Some(HormoneReaction {
                            receptor_id: receptor_id.clone(),
                            host_module: receptor.host_module.clone(),
                            hormone_type: receptor.hormone_type.clone(),
                            signal_intensity: hormone_intensity,
                            reaction_intensity,
                            action: receptor.action.clone(),
                            target_parameter: receptor.target_parameter.clone(),
                            reaction_time: now,
                        })
                    } else {
                        None
                    }
                })
                .collect();
            
            // Traiter les réactions
            if !reactions.is_empty() {
                // Mettre à jour le compteur
                if let Ok(mut count) = total_reactions.write() {
                    *count += reactions.len() as u64;
                }
                
                // Stocker dans l'historique récent
                if let Ok(mut recent) = recent_reactions.lock() {
                    for reaction in &reactions {
                        recent.push_back(reaction.clone());
                        
                        // Limiter la taille
                        if recent.len() > 1000 {
                            recent.pop_front();
                        }
                    }
                }
                
                // Déclencher les callbacks en parallel
                reactions.into_par_iter().for_each(|reaction| {
                    if let Some(callback) = receptor_callbacks.get(&reaction.receptor_id) {
                        callback(&reaction);
                    }
                });
            }
        });
    }
    
    /// Traite les réactions hormonales - version générique
    #[cfg(not(target_os = "windows"))]
    fn process_hormone_reactions(
        thread_count: usize,
        hormone_receptors: &Arc<RwLock<HashMap<String, HormoneReceptor>>>,
        receptor_by_module: &Arc<DashMap<String, Vec<String>>>,
        global_hormone_levels: &Arc<PLRwLock<HashMap<HormoneType, f64>>>,
        local_hormone_buffers: &Arc<DashMap<String, VecDeque<(HormoneType, f64, Instant)>>>,
        receptor_callbacks: &Arc<DashMap<String, Box<dyn Fn(&HormoneReaction) -> bool + Send + Sync>>>,
        recent_reactions: &Arc<Mutex<VecDeque<HormoneReaction>>>,
        total_reactions: &Arc<PLRwLock<u64>>,
    ) {
        // Implémentation pour les autres systèmes d'exploitation
        // (Code équivalent mais sans l'optimisation spécifique à Windows)
        // ...
    }
    
    /// Met à jour l'historique des niveaux hormonaux
    fn update_hormone_history(
        global_hormone_levels: &Arc<PLRwLock<HashMap<HormoneType, f64>>>,
        hormone_history: &Arc<Mutex<HashMap<HormoneType, VecDeque<(f64, Instant)>>>>,
    ) {
        let global_levels = match global_hormone_levels.read() {
            Ok(levels) => levels.clone(),
            Err(_) => return,
        };
        
        let now = Instant::now();
        
        if let Ok(mut history) = hormone_history.lock() {
            for (hormone_type, level) in global_levels {
                // Récupérer ou créer l'historique pour ce type d'hormone
                let hormone_history = history.entry(hormone_type).or_insert_with(|| {
                    VecDeque::with_capacity(100)
                });
                
                // Ajouter le niveau actuel
                hormone_history.push_back((level, now));
                
                // Limiter la taille de l'historique
                while hormone_history.len() > 100 {
                    hormone_history.pop_front();
                }
            }
        }
    }
    
    /// Émet un signal hormonal dans le système
    pub fn emit_hormone(
        &self,
        hormone_type: HormoneType,
        source_module: &str,
        intensity: f64,
        half_life_factor: f64,
        diffusion_radius: f64,
        context: HashMap<String, Vec<u8>>,
    ) -> Result<String, String> {
        // Vérifier si le système est actif
        let is_active = match self.active.read() {
            Ok(a) => *a,
            Err(_) => return Err("Impossible d'accéder à l'état du système".to_string()),
        };
        
        if !is_active {
            return Err("Système hormonal inactif".to_string());
        }
        
        // Valider les paramètres
        if intensity <= 0.0 || intensity > 2.0 {
            return Err("L'intensité doit être entre 0.0 et 2.0".to_string());
        }
        
        if diffusion_radius <= 0.0 || diffusion_radius > 1.0 {
            return Err("Le rayon de diffusion doit être entre 0.0 et 1.0".to_string());
        }
        
        // Générer un ID unique pour le signal
        let signal_id = format!("{}_{}_{}",
                               hormone_type.to_string(),
                               chrono::Utc::now().timestamp_nanos(),
                               thread_rng().gen::<u32>());
        
        // Déterminer la demi-vie
        let half_life = {
            let base_half_life = match self.half_lives.read() {
                Ok(half_lives) => half_lives.get(&hormone_type)
                                          .cloned()
                                          .unwrap_or_else(|| Duration::from_secs(300)),
                Err(_) => Duration::from_secs(300), // Valeur par défaut en cas d'erreur
            };
            
            // Ajuster la demi-vie selon le facteur fourni
            let adjusted_secs = (base_half_life.as_secs_f64() * half_life_factor).max(1.0);
            Duration::from_secs_f64(adjusted_secs)
        };
        
        // Créer le signal
        let hormone_signal = HormoneSignal {
            hormone_type: hormone_type.clone(),
            signal_id: signal_id.clone(),
            source_module: source_module.to_string(),
            initial_intensity: intensity,
            current_intensity: intensity,
            emission_time: Instant::now(),
            half_life,
            diffusion_radius,
            context,
        };
        
        // Ajouter aux signaux actifs
        self.active_signals.insert(signal_id.clone(), hormone_signal);
        
        // Mettre à jour immédiatement le niveau global
        if let Ok(mut global_levels) = self.global_hormone_levels.write() {
            let current_level = global_levels.entry(hormone_type.clone()).or_insert(0.0);
            *current_level += intensity * diffusion_radius;
            *current_level = current_level.min(1.0);
        }
        
        // Mettre à jour immédiatement le tampon local du module source
        {
            let mut buffer = self.local_hormone_buffers
                .entry(source_module.to_string())
                .or_insert_with(VecDeque::new);
            
            buffer.push_back((hormone_type, intensity, Instant::now()));
            
            // Limiter la taille
            while buffer.len() > MAX_LOCAL_HORMONE_BUFFER {
                buffer.pop_front();
            }
        }
        
        // Mettre à jour les statistiques
        if let Ok(mut count) = self.total_signals_emitted.write() {
            *count += 1;
        }
        
        info!("Signal hormonal émis: {:?} depuis {} (intensité: {:.2}, diffusion: {:.2})",
              hormone_type, source_module, intensity, diffusion_radius);
        
        Ok(signal_id)
    }
    
    /// Enregistre un récepteur hormonal
    pub fn register_receptor(
        &self,
        host_module: &str,
        hormone_type: HormoneType,
        sensitivity: f64,
        threshold: f64,
        action: ReceptorAction,
        target_parameter: &str,
    ) -> Result<String, String> {
        // Valider les paramètres
        if sensitivity <= 0.0 || sensitivity > MAX_RECEPTOR_SENSITIVITY {
            return Err(format!("La sensibilité doit être entre 0.0 et {}", MAX_RECEPTOR_SENSITIVITY));
        }
        
        // Générer un ID unique
        let receptor_id = format!("{}_{}_{}",
                                 host_module,
                                 hormone_type.to_string(),
                                 chrono::Utc::now().timestamp_nanos());
        
        // Créer le récepteur
        let receptor = HormoneReceptor {
            id: receptor_id.clone(),
            host_module: host_module.to_string(),
            hormone_type,
            sensitivity,
            threshold,
            action,
            target_parameter: target_parameter.to_string(),
            creation_time: Instant::now(),
            last_activation: None,
            metadata: HashMap::new(),
        };
        
        // Enregistrer le récepteur
        if let Ok(mut receptors) = self.hormone_receptors.write() {
            receptors.insert(receptor_id.clone(), receptor);
        } else {
            return Err("Impossible d'accéder au registre des récepteurs".to_string());
        }
        
        // Ajouter à l'index par module
        self.receptor_by_module
            .entry(host_module.to_string())
            .or_insert_with(Vec::new)
            .push(receptor_id.clone());
        
        // Créer un tampon hormonal local pour ce module s'il n'existe pas
        self.local_hormone_buffers
            .entry(host_module.to_string())
            .or_insert_with(VecDeque::new);
        
        info!("Récepteur hormonal enregistré: {} pour {} dans {} (sensibilité: {:.2})",
              receptor_id, hormone_type.to_string(), host_module, sensitivity);
        
        Ok(receptor_id)
    }
    
    /// Enregistre une fonction de rappel pour un récepteur
    pub fn register_receptor_callback<F>(
        &self,
        receptor_id: &str,
        callback: F,
    ) -> Result<(), String>
    where
        F: Fn(&HormoneReaction) -> bool + Send + Sync + 'static,
    {
        // Vérifier que le récepteur existe
        let receptor_exists = if let Ok(receptors) = self.hormone_receptors.read() {
            receptors.contains_key(receptor_id)
        } else {
            false
        };
        
        if !receptor_exists {
            return Err(format!("Récepteur {} introuvable", receptor_id));
        }
        
        // Enregistrer le callback
        self.receptor_callbacks.insert(receptor_id.to_string(), Box::new(callback));
        
        info!("Callback enregistré pour le récepteur {}", receptor_id);
        
        Ok(())
    }
    
    /// Obtient le niveau actuel d'une hormone
    pub fn get_hormone_level(&self, hormone_type: &HormoneType) -> f64 {
        if let Ok(global_levels) = self.global_hormone_levels.read() {
            *global_levels.get(hormone_type).unwrap_or(&0.0)
        } else {
            0.0
        }
    }
    
    /// Obtient le niveau local d'une hormone pour un module spécifique
    pub fn get_local_hormone_level(&self, module_id: &str, hormone_type: &HormoneType) -> f64 {
        if let Some(buffer) = self.local_hormone_buffers.get(module_id) {
            buffer.iter()
                .filter(|(h_type, _, _)| h_type == hormone_type)
                .map(|(_, intensity, _)| *intensity)
                .sum()
        } else {
            // Si pas de tampon local, utiliser le niveau global
            self.get_hormone_level(hormone_type)
        }
    }
    
    /// Obtient l'état du champ hormonal
    pub fn get_hormone_field_state(&self) -> HormoneFieldState {
        let global_levels = match self.global_hormone_levels.read() {
            Ok(levels) => levels.clone(),
            Err(_) => HashMap::new(),
        };
        
        let mut gradients = HashMap::new();
        let mut active_signals_count = HashMap::new();
        
        // Calculer les gradients à partir de l'historique
        if let Ok(history) = self.hormone_history.lock() {
            for (hormone_type, hist) in history.iter() {
                if hist.len() >= 2 {
                    let newest = hist.back().map(|(level, _)| *level).unwrap_or(0.0);
                    let oldest = hist.front().map(|(level, _)| *level).unwrap_or(0.0);
                    
                    gradients.insert(hormone_type.clone(), newest - oldest);
                }
            }
        }
        
        // Compter les signaux actifs par type
        self.active_signals.iter().for_each(|entry| {
            let signal = entry.value();
            let count = active_signals_count
                .entry(signal.hormone_type.clone())
                .or_insert(0);
            *count += 1;
        });
        
        HormoneFieldState {
            hormone_levels: global_levels,
            gradients,
            active_signals_count,
            measurement_time: Instant::now(),
        }
    }
    
    /// Modifie la demi-vie d'une hormone
    pub fn set_hormone_half_life(
        &self,
        hormone_type: HormoneType,
        half_life: Duration,
    ) -> Result<(), String> {
        if half_life < Duration::from_secs(1) || half_life > Duration::from_secs(86400) {
            return Err("La demi-vie doit être entre 1 seconde et 24 heures".to_string());
        }
        
        if let Ok(mut half_lives) = self.half_lives.write() {
            half_lives.insert(hormone_type.clone(), half_life);
            
            info!("Demi-vie modifiée pour {:?}: {:?}", hormone_type, half_life);
            
            Ok(())
        } else {
            Err("Impossible d'accéder aux demi-vies des hormones".to_string())
        }
    }
    
    /// Modifie le taux de diffusion d'une hormone
    pub fn set_hormone_diffusion_rate(
        &self,
        hormone_type: HormoneType,
        rate: f64,
    ) -> Result<(), String> {
        if rate <= 0.0 || rate > 0.5 {
            return Err("Le taux de diffusion doit être entre 0.0 et 0.5".to_string());
        }
        
        self.diffusion_rates.insert(hormone_type.clone(), rate);
        
        info!("Taux de diffusion modifié pour {:?}: {:.3}", hormone_type, rate);
        
        Ok(())
    }
    
    /// Obtient les statistiques du système hormonal
    pub fn get_stats(&self) -> HormonalSystemStats {
        let active_signals_count = self.active_signals.len();
        
        let receptors_count = match self.hormone_receptors.read() {
            Ok(receptors) => receptors.len(),
            Err(_) => 0,
        };
        
        let total_signals = *self.total_signals_emitted.read();
        let total_reactions = *self.total_reactions.read();
        
        let global_levels = match self.global_hormone_levels.read() {
            Ok(levels) => levels.clone(),
            Err(_) => HashMap::new(),
        };
        
        HormonalSystemStats {
            active_signals_count,
            receptors_count,
            total_signals_emitted: total_signals,
            total_reactions,
            age_seconds: self.system_birth_time.elapsed().as_secs(),
            current_hormone_levels: global_levels,
        }
    }
}

impl std::fmt::Display for HormoneType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HormoneType::Cortisol => write!(f, "Cortisol"),
            HormoneType::Adrenaline => write!(f, "Adrenaline"),
            HormoneType::Dopamine => write!(f, "Dopamine"),
            HormoneType::Serotonin => write!(f, "Serotonin"),
            HormoneType::Melatonin => write!(f, "Melatonin"),
            HormoneType::Oxytocin => write!(f, "Oxytocin"),
            HormoneType::Endorphin => write!(f, "Endorphin"),
            HormoneType::Testosterone => write!(f, "Testosterone"),
            HormoneType::Insulin => write!(f, "Insulin"),
            HormoneType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Statistiques du système hormonal
#[derive(Debug, Clone)]
pub struct HormonalSystemStats {
    pub active_signals_count: usize,
    pub receptors_count: usize,
    pub total_signals_emitted: u64,
    pub total_reactions: u64,
    pub age_seconds: u64,
    pub current_hormone_levels: HashMap<HormoneType, f64>,
}

/// File d'attente fixe avec limite de capacité (optimisée pour Windows)
#[derive(Debug)]
struct VecDeque<T> {
    // Utilisation d'un tableau avec pointeur de début et de fin
    buffer: Vec<Option<T>>,
    head: usize,
    tail: usize,
    len: usize,
    capacity: usize,
}

impl<T: Clone> VecDeque<T> {
    fn with_capacity(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);
        
        Self {
            buffer,
            head: 0,
            tail: 0,
            len: 0,
            capacity,
        }
    }
    
    fn push_back(&mut self, value: T) {
        if self.len == self.capacity {
            // Queue pleine, écraser la valeur la plus ancienne
            self.head = (self.head + 1) % self.capacity;
        } else {
            self.len += 1;
        }
        
        self.buffer[self.tail] = Some(value);
        self.tail = (self.tail + 1) % self.capacity;
    }
    
    fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        
        let value = self.buffer[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.len -= 1;
        
        value
    }
    
    fn len(&self) -> usize {
        self.len
    }
    
    fn iter(&self) -> impl Iterator<Item = &T> {
        let mut index = self.head;
        let len = self.len;
        let capacity = self.capacity;
        let buffer = &self.buffer;
        
        std::iter::from_fn(move || {
            if len == 0 {
                return None;
            }
            
            let mut count = 0;
            while count < len {
                if let Some(ref value) = buffer[index] {
                    let current = index;
                    index = (index + 1) % capacity;
                    return Some(value);
                }
                
                index = (index + 1) % capacity;
                count += 1;
            }
            
            None
        })
    }
    
    fn retain<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&T) -> bool,
    {
        let mut new_deque = VecDeque::with_capacity(self.capacity);
        
        while let Some(item) = self.pop_front() {
            if predicate(&item) {
                new_deque.push_back(item);
            }
        }
        
        *self = new_deque;
    }
    
    fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        
        let actual_index = (self.head + index) % self.capacity;
        let value = self.buffer[actual_index].take();
        
        // Décaler tous les éléments après l'élément supprimé
        let mut i = actual_index;
        let mut next = (i + 1) % self.capacity;
        
        while next != self.tail {
            self.buffer[i] = self.buffer[next].take();
            i = next;
            next = (next + 1) % self.capacity;
        }
        
        self.tail = if self.tail == 0 {
            self.capacity - 1
        } else {
            self.tail - 1
        };
        
        self.len -= 1;
        
        value
    }
}

impl<T: Clone> Clone for VecDeque<T> {
    fn clone(&self) -> Self {
        let mut new_deque = VecDeque::with_capacity(self.capacity);
        
        for item in self.iter() {
            new_deque.push_back(item.clone());
        }
        
        new_deque
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hormonal_field_creation() {
        let hormonal = HormonalField::new();
        let stats = hormonal.get_stats();
        
        assert_eq!(stats.active_signals_count, 0);
        assert_eq!(stats.receptors_count, 0);
        assert_eq!(stats.total_signals_emitted, 0);
    }
    
    #[test]
    fn test_hormone_emission() {
        let hormonal = HormonalField::new();
        
        let result = hormonal.emit_hormone(
            HormoneType::Dopamine,
            "test_module",
            0.7,
            1.0,
            0.5,
            HashMap::new()
        );
        
        assert!(result.is_ok());
        
        // Vérifier que le niveau a été mis à jour
        let level = hormonal.get_hormone_level(&HormoneType::Dopamine);
        assert!(level > 0.0);
    }
    
    #[test]
    fn test_receptor_registration() {
        let hormonal = HormonalField::new();
        
        let result = hormonal.register_receptor(
            "test_module",
            HormoneType::Cortisol,
            1.5,
            0.3,
            ReceptorAction::Increase,
            "stress_level"
        );
        
        assert!(result.is_ok());
        
        // Vérifier que le récepteur a été enregistré
        let stats = hormonal.get_stats();
        assert_eq!(stats.receptors_count, 1);
    }
}
