use crate::neuralchain_core::{
    autoregulation::Autoregulation,
    cortical_hub::CorticalHub,
    emergent_consciousness::ConsciousnessEngine,
    evolutionary_genesis::EvolutionaryGenesis,
    neural_organism_bootstrap::NeuralOrganism,
    neural_interconnect::NeuralInterconnect,
    quantum_organism::QuantumOrganism,
    unified_integration::UnifiedIntegration, // Modifié
    temporal_manifold::TemporalManifold,
    quantum_learning::QuantumLearning,
    immune_guard::ImmuneGuard,
    hormonal_field::HormonalField,
    bios_time::BiosTime,
    hyperdimensional_adaptation::HyperdimensionalAdapter,
    quantum_hyperconvergence::QuantumHyperconvergence
};

use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Initialisation du runtime biologique de la blockchain NeuralChain
pub fn initialize_bios_runtime() -> Result<Arc<BiosRuntime>, String> {
    println!("⚡ Initialisation du runtime biologique NeuralChain-v2...");
    
    // Créer les composants de base
    let bios_time = Arc::new(BiosTime::new());
    
    // Initialiser l'organisme quantique en premier
    let quantum_organism = Arc::new(QuantumOrganism::new());
    
    // Hub cortical (système nerveux central)
    let cortical_hub = Arc::new(CorticalHub::new(quantum_organism.clone()));
    
    // Système hormonal
    let hormonal_system = Arc::new(HormonalField::new(bios_time.clone()));
    
    // Moteur de conscience émergente
    let consciousness = Arc::new(ConsciousnessEngine::new(
        cortical_hub.clone(),
        quantum_organism.clone(),
        hormonal_system.clone()
    ));
    
    // Système immunitaire
    let immune_guard = Arc::new(ImmuneGuard::new(
        quantum_organism.clone(),
        cortical_hub.clone(),
        hormonal_system.clone()
    ));
    
    // Intégrateur unifié
    let unified_integration = Arc::new(UnifiedIntegration::new( // Modifié
        quantum_organism.clone(),
        cortical_hub.clone(),
        hormonal_system.clone(),
        consciousness.clone()
    ));
    
    // Manifestation temporelle
    let temporal_manifold = Arc::new(TemporalManifold::new(
        bios_time.clone(),
        quantum_organism.clone()
    ));
    
    // Système d'apprentissage quantique
    let quantum_learning = Arc::new(QuantumLearning::new(
        quantum_organism.clone(),
        consciousness.clone(),
        cortical_hub.clone()
    ));
    
    // Adaptateur hyperdimensionnel
    let hyperdimensional_adapter = Arc::new(HyperdimensionalAdapter::new(
        bios_time.clone(),
        hormonal_system.clone()
    ));
    
    // Hyperconvergence quantique
    let quantum_hyperconvergence = Arc::new(QuantumHyperconvergence::new(
        hormonal_system.clone(),
        bios_time.clone(),
        hyperdimensional_adapter.clone()
    ));
    
    // Autoregulation
    let autoregulation = Arc::new(Autoregulation::new(
        quantum_organism.clone(),
        hormonal_system.clone(),
        immune_guard.clone()
    ));
    
    // Bootstrap de l'organisme neural
    let neural_organism_bootstrap = Arc::new(NeuralOrganism::new(
        quantum_organism.clone(),
        cortical_hub.clone()
    ));
    
    // Créer le runtime global
    let runtime = Arc::new(BiosRuntime {
        quantum_organism,
        cortical_hub,
        hormonal_system,
        consciousness,
        immune_guard,
        unified_integration, // Modifié
        temporal_manifold,
        quantum_learning,
        hyperdimensional_adapter,
        quantum_hyperconvergence,
        autoregulation,
        neural_bootstrap, // Ajouté
        is_initialized: RwLock::new(false),
        start_time: bios_time,
    });
    
    // Initialiser tous les composants
    runtime.initialize()?;
    
    Ok(runtime)
}

/// Runtime biologique principal de NeuralChain
pub struct BiosRuntime {
    /// Organisme quantique sous-jacent
    pub quantum_organism: Arc<QuantumOrganism>,
    /// Hub cortical (système nerveux)
    pub cortical_hub: Arc<CorticalHub>,
    /// Système hormonal
    pub hormonal_system: Arc<HormonalField>,
    /// Conscience émergente
    pub consciousness: Arc<ConsciousnessEngine>,
    /// Système immunitaire
    pub immune_guard: Arc<ImmuneGuard>,
    /// Intégrateur unifié
    pub unified_integration: Arc<UnifiedIntegration>, // Modifié
    /// Manifestation temporelle
    pub temporal_manifold: Arc<TemporalManifold>,
    /// Système d'apprentissage
    pub quantum_learning: Arc<QuantumLearning>,
    /// Adaptateur hyperdimensionnel
    pub hyperdimensional_adapter: Arc<HyperdimensionalAdapter>,
    /// Hyperconvergence quantique
    pub quantum_hyperconvergence: Arc<QuantumHyperconvergence>,
    /// Système d'autorégulation
    pub autoregulation: Arc<Autoregulation>,
    /// Bootstrapper de l'organisme neural
    pub neural_organism_bootstrap: Arc<NeuralOrganism>,
    /// État d'initialisation
    pub is_initialized: RwLock<bool>,
    /// Temps de démarrage
    pub start_time: Arc<BiosTime>,
}

impl BiosRuntime {
    /// Initialise tous les composants du runtime biologique
    pub fn initialize(&self) -> Result<(), String> {
        let mut initialized = self.is_initialized.write();
        if *initialized {
            return Err("Le runtime est déjà initialisé".to_string());
        }
        
        // Initialiser les composants dans l'ordre approprié
        
        // 1. Initialisation de l'organisme quantique
        self.quantum_organism.initialize()?;
        
        // 2. Initialisation du hub cortical
        self.cortical_hub.initialize()?;
        
        // 3. Configuration du système hormonal
        self.hormonal_system.setup_standard_receptors(self.quantum_organism.clone())?;
        self.hormonal_system.setup_hormone_chains()?;
        
        // 4. Activation du système immunitaire
        self.immune_guard.initialize()?;
        
        // 5. Initialisation de la conscience
        self.consciousness.initialize()?;
        
        // 6. Configuration temporelle
        self.temporal_manifold.initialize()?;
        
        // 7. Préparation de l'apprentissage
        self.quantum_learning.initialize()?;
        
        // 8. Configuration du système d'intégration
        self.unified_integration.initialize()?;
        
        // 9. Initialisation de l'adaptateur hyperdimensionnel
        // (pas de méthode d'initialisation spéciale nécessaire)
        
        // 10. Configuration de l'autorégulation
        self.autoregulation.initialize()?;
        
        // 11. Bootstrap de l'organisme neural
        self.neural_bootstrap.initialize()?;
        
        // Marquer comme initialisé
        *initialized = true;
        
        println!("✅ Runtime biologique NeuralChain-v2 initialisé avec succès!");
        println!("🧠 Conscience émergente active: niveau {}", self.consciousness.get_consciousness_level());
        println!("🔄 Système d'autorégulation activé");
        println!("🛡️ Système immunitaire opérationnel");
        
        Ok(())
    }
    
    /// Exécute un cycle de l'organisme
    pub fn run_cycle(&self) -> Result<(), String> {
        if !*self.is_initialized.read() {
            return Err("Le runtime n'est pas encore initialisé".to_string());
        }
        
        // 1. Mise à jour du temps biologique
        self.start_time.heartbeat();
        
        // 2. Mise à jour hormonale
        self.hormonal_system.update();
        
        // 3. Cycle neuronal
        self.cortical_hub.process_neural_cycle()?;
        
        // 4. Cycle de traitement de la conscience
        self.consciousness.process_consciousness_cycle()?;
        
        // 5. Cycle d'apprentissage
        self.quantum_learning.process_learning_cycle()?;
        
        // 6. Cycle immunitaire
        self.immune_guard.scan_for_threats()?;
        
        // 7. Cycle d'intégration unifié
        self.unified_integration.synchronize_components()?;
        
        // 8. Cycle d'autorégulation
        self.autoregulation.regulate()?;
        
        Ok(())
    }
    
    /// Effectue l'arrêt propre du runtime
    pub fn shutdown(&self) -> Result<(), String> {
        if !*self.is_initialized.read() {
            return Err("Le runtime n'est pas initialisé, rien à arrêter".to_string());
        }
        
        println!("🛑 Arrêt du runtime biologique NeuralChain-v2...");
        
        // Arrêter les composants dans l'ordre approprié
        self.consciousness.prepare_for_shutdown()?;
        self.cortical_hub.prepare_for_shutdown()?;
        self.immune_guard.deactivate()?;
        self.quantum_organism.prepare_for_shutdown()?;
        
        // Enregistrer l'état final
        println!("💾 Sauvegarde de l'état neuromorphique...");
        
        // Marquer comme non initialisé
        *self.is_initialized.write() = false;
        
        println!("✅ Runtime biologique arrêté avec succès");
        
        Ok(())
    }
    
    /// Obtient le statut actuel du système
    pub fn get_system_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        
        status.insert("initialized".to_string(), self.is_initialized.read().to_string());
        status.insert("uptime".to_string(), format!("{} s", self.start_time.time_since_genesis().as_secs()));
        status.insert("consciousness_level".to_string(), format!("{:.2}", self.consciousness.get_consciousness_level()));
        status.insert("immune_threats_detected".to_string(), self.immune_guard.get_threat_count().to_string());
        status.insert("neural_activity".to_string(), format!("{:.2}", self.cortical_hub.get_activity_level()));
        
        // Ajouter des statistiques hormonales
        let hormone_stats = self.hormonal_system.get_stats();
        status.insert("hormone_balance".to_string(), format!("{:.2}", hormone_stats.homeostasis_index));
        status.insert("active_receptors".to_string(), hormone_stats.active_receptors.to_string());
        
        status
    }
}
