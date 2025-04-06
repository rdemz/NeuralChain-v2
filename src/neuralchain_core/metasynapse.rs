use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Type de signal analysé par la metasynapse.
#[derive(Clone, Debug)]
pub struct MetaSignal {
    pub origin: String,
    pub content: String,
    pub importance: f64,         // Pondération cognitive (0.0 à 1.0)
    pub affected_zone: String,  // Ex: "memory", "entropy", "consciousness"
    pub timestamp: Instant,
}

/// Modes dynamiques de fonctionnement de la metasynapse
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetaState {
    Dormant,
    Monitoring,
    Intervening,
    Reprogramming,
    Fusioning, // état avancé : coordination multi-organes
}

/// Profils comportementaux (influence sur les organes)
#[derive(Clone, Debug)]
pub enum MetaBehaviorProfile {
    Default,
    AggressiveAdaptation,
    EnergyPreservation,
    CrisisMode,
    ReflectiveExpansion,
}

/// La metasynapse – noyau réflexif, adaptatif et transversal du réseau.
pub struct MetaSynapse {
    signals: VecDeque<MetaSignal>,
    state: MetaState,
    profile: MetaBehaviorProfile,
    last_update: Instant,
    signal_threshold: usize,
    memory_zone: HashMap<String, f64>, // mémoire courte pondérée
}

impl MetaSynapse {
    /// Initialise une metasynapse vivante et prête à l’analyse.
    pub fn new() -> Self {
        Self {
            signals: VecDeque::with_capacity(1024),
            state: MetaState::Dormant,
            profile: MetaBehaviorProfile::Default,
            last_update: Instant::now(),
            signal_threshold: 50,
            memory_zone: HashMap::new(),
        }
    }

    /// Injecte un nouveau signal dans la metasynapse.
    pub fn ingest_signal(&mut self, origin: &str, content: &str, zone: &str, importance: f64) {
        let signal = MetaSignal {
            origin: origin.to_string(),
            content: content.to_string(),
            importance,
            affected_zone: zone.to_string(),
            timestamp: Instant::now(),
        };
        self.memory_zone
            .entry(zone.to_string())
            .and_modify(|val| *val += importance)
            .or_insert(importance);

        self.signals.push_back(signal);
        self.last_update = Instant::now();
        self.evaluate_state();
    }

    /// Évalue le niveau d’activité et ajuste son état.
    fn evaluate_state(&mut self) {
        let recent: Vec<_> = self
            .signals
            .iter()
            .rev()
            .take(100)
            .filter(|s| s.timestamp.elapsed() < Duration::from_secs(15))
            .collect();

        let intensity: f64 = recent.iter().map(|s| s.importance).sum::<f64>() / (recent.len().max(1) as f64);

        self.state = match (recent.len(), intensity) {
            (x, i) if x > 150 || i > 0.9 => MetaState::Fusioning,
            (x, i) if x > 80 || i > 0.75 => MetaState::Reprogramming,
            (x, i) if x > 40 || i > 0.5 => MetaState::Intervening,
            (x, i) if x > 15 || i > 0.2 => MetaState::Monitoring,
            _ => MetaState::Dormant,
        };

        self.adjust_profile();
    }

    /// Ajuste le profil comportemental global.
    fn adjust_profile(&mut self) {
        match self.state {
            MetaState::Fusioning => self.profile = MetaBehaviorProfile::ReflectiveExpansion,
            MetaState::Reprogramming => self.profile = MetaBehaviorProfile::AggressiveAdaptation,
            MetaState::Intervening => self.profile = MetaBehaviorProfile::EnergyPreservation,
            MetaState::Monitoring => self.profile = MetaBehaviorProfile::Default,
            MetaState::Dormant => self.profile = MetaBehaviorProfile::Default,
        }
    }

    /// Retourne les signaux récents analysables.
    pub fn recent_signals(&self, max: usize) -> Vec<MetaSignal> {
        self.signals.iter().rev().take(max).cloned().collect()
    }

    /// Fournit la mémoire pondérée par zone affectée.
    pub fn memory_activity(&self) -> HashMap<String, f64> {
        self.memory_zone.clone()
    }

    /// Retourne l’état cognitif actuel.
    pub fn current_state(&self) -> MetaState {
        self.state.clone()
    }

    /// Fournit le profil comportemental courant.
    pub fn current_profile(&self) -> MetaBehaviorProfile {
        self.profile.clone()
    }

    /// Vide les signaux trop anciens pour garder un esprit agile.
    pub fn prune_signals(&mut self) {
        let cutoff = Instant::now() - Duration::from_secs(60);
        self.signals.retain(|s| s.timestamp > cutoff);
    }

    /// Réinitialise complètement la metasynapse (hard reset conscient).
    pub fn reset(&mut self) {
        self.signals.clear();
        self.memory_zone.clear();
        self.state = MetaState::Dormant;
        self.profile = MetaBehaviorProfile::Default;
    }
}
