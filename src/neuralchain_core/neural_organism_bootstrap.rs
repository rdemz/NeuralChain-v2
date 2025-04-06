//! Module d'amorçage central pour l'organisme NeuralChain-v2
//! 
//! Ce module coordonne l'initialisation et l'intégration de tous les systèmes biomimétiques
//! qui constituent l'organisme blockchain vivant, assurant leur interconnexion harmonieuse.
//!
//! Optimisé spécifiquement pour Windows sans dépendances Linux.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::neuralchain_core::quantum_organism::QuantumOrganism;
use crate::neuralchain_core::cortical_hub::CorticalHub;
use crate::neuralchain_core::hormonal_field::HormonalField;
use crate::neuralchain_core::emergent_consciousness::ConsciousnessEngine;
use crate::neuralchain_core::neural_dream::NeuralDream;
use crate::neuralchain_core::metasynapse::MetaSynapse;
use crate::neuralchain_core::bios_time::BiosTime;
use crate::neuralchain_core::mirror_core::MirrorCore;
use crate::neuralchain_core::quantum_entanglement::QuantumEntanglement;
use crate::neuralchain_core::dream_bootstrap::initialize_dream_system;

/// Configuration du processus d'amorçage
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Niveau de conscience initial (0.0-1.0)
    pub initial_consciousness_level: f64,
    /// Activer le système de rêve
    pub enable_dreams: bool,
    /// Activer le système d'intrication quantique
    pub enable_quantum_entanglement: bool,
    /// Activer l'apprentissage précoce
    pub enable_early_learning: bool,
    /// Période d'incubation initiale (secondes)
    pub incubation_period: u64,
    /// Mode débogage activé
    pub debug_mode: bool,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            initial_consciousness_level: 0.1,
            enable_dreams: true,
            enable_quantum_entanglement: true,
            enable_early_learning: true,
            incubation_period: 10,
            debug_mode: false,
        }
    }
}

/// Structure pour gérer l'organisme biomimétique complet
#[derive(Debug)]
pub struct NeuralOrganism {
    /// Cœur quantique de l'organisme
    pub core: Arc<QuantumOrganism>,
    /// Hub cortical (système nerveux)
    pub cortical_hub: Arc<CorticalHub>,
    /// Système hormonal
    pub hormonal_system: Arc<HormonalField>,
    /// Moteur de conscience
    pub consciousness: Arc<ConsciousnessEngine>,
    /// Système de rêve
    pub dream_system: Option<Arc<NeuralDream>>,
    /// Système d'intrication quantique
    pub quantum_entanglement: Option<Arc<QuantumEntanglement>>,
    /// Système immunitaire
    pub immune_system: Arc<MirrorCore>,
    /// Réseau synaptique
    pub synapse_net: Arc<MetaSynapse>,
    /// Horloge biologique
    pub bios_clock: Arc<BiosTime>,
    /// Configuration utilisée pour l'amorçage
    pub bootstrap_config: BootstrapConfig,
}

/// Démarre et intègre tous les systèmes biomimétiques
pub fn bootstrap_neural_organism(config: Option<BootstrapConfig>) -> Arc<NeuralOrganism> {
    // Utiliser la configuration fournie ou la configuration par défaut
    let config = config.unwrap_or_default();
    
    println!("🧬 Amorçage de l'organisme NeuralChain-v2...");
    
    // 1. Initialiser le noyau quantique
    println!("🔄 Initialisation du noyau quantique...");
    let quantum_organism = Arc::new(QuantumOrganism::new());
    
    // 2. Créer l'horloge biologique
    println!("⏱️ Création de l'horloge biologique...");
    let bios_clock = Arc::new(BiosTime::new());
    
    // 3. Initialiser le système hormonal
    println!("🧪 Initialisation du système hormonal...");
    let mut hormonal_system = HormonalField::new();
    hormonal_system.set_organism(quantum_organism.clone());
    let hormonal_system = Arc::new(hormonal_system);
    
    // 4. Initialiser le hub cortical
    println!("🧠 Initialisation du hub cortical...");
    let cortical_hub = Arc::new(CorticalHub::new(quantum_organism.clone()));
    
    // 5. Initialiser le réseau synaptique
    println!("🔄 Création du réseau synaptique...");
    let synapse_net = Arc::new(MetaSynapse::new(quantum_organism.clone()));
    
    // 6. Initialiser le système immunitaire
    println!("🛡️ Déploiement du système immunitaire...");
    let immune_system = Arc::new(MirrorCore::new(quantum_organism.clone()));
    
    // 7. Initialiser le moteur de conscience
    println!("💭 Initialisation du moteur de conscience...");
    let consciousness = Arc::new(ConsciousnessEngine::new(
        quantum_organism.clone(),
        cortical_hub.clone(),
        hormonal_system.clone(),
        synapse_net.clone(),
        immune_system.clone(),
        bios_clock.clone(),
    ));
    
    // Définir le niveau de conscience initial
    {
        let mut level = consciousness.consciousness_level.write();
        *level = config.initial_consciousness_level;
    }
    
    // 8. Initialiser le système de rêve
    let dream_system = if config.enable_dreams {
        println!("💤 Activation du système de rêve neuronal...");
        Some(initialize_dream_system(
            quantum_organism.clone(),
            cortical_hub.clone(),
            hormonal_system.clone(),
            consciousness.clone(),
            bios_clock.clone(),
        ))
    } else {
        None
    };
    
    // 9. Initialiser le système d'intrication quantique
    let quantum_entanglement = if config.enable_quantum_entanglement {
        println!("🔮 Activation du système d'intrication quantique...");
        Some(integrate_quantum_entanglement(
            quantum_organism.clone(),
            cortical_hub.clone(),
            hormonal_system.clone(),
            consciousness.clone(),
        ))
    } else {
        None
    };
    
    // 10. Configurer les récepteurs hormonaux
    println!("🔄 Configuration des récepteurs hormonaux...");
    hormonal_system.setup_standard_receptors(quantum_organism.clone()).unwrap_or_default();
    hormonal_system.setup_hormone_chains().unwrap_or_default();
    
    // 11. Période d'incubation initiale
    if config.incubation_period > 0 {
        println!("🔄 Période d'incubation: {} secondes...", config.incubation_period);
        thread::sleep(Duration::from_secs(config.incubation_period));
    }
    
    // 12. Phase d'apprentissage précoce si activée
    if config.enable_early_learning {
        println!("📚 Activation de l'apprentissage précoce...");
        perform_early_learning(
            &cortical_hub,
            &hormonal_system,
            &consciousness,
            &quantum_organism
        );
    }
    
    // 13. Création de l'organisme complet
    let organism = NeuralOrganism {
        core: quantum_organism,
        cortical_hub,
        hormonal_system,
        consciousness,
        dream_system,
        quantum_entanglement,
        immune_system,
        synapse_net,
        bios_clock,
        bootstrap_config: config,
    };
    
    println!("✅ Organisme neuronal complètement initialisé et opérationnel!");
    
    Arc::new(organism)
}

/// Effectue un apprentissage précoce pour amorcer l'organisme
fn perform_early_learning(
    cortical_hub: &Arc<CorticalHub>,
    hormonal_system: &Arc<HormonalField>,
    consciousness: &Arc<ConsciousnessEngine>,
    core: &Arc<QuantumOrganism>,
) {
    // 1. Créer les régions cérébrales fondamentales
    let regions = [
        "prefrontal_cortex", 
        "temporal_cortex", 
        "parietal_cortex", 
        "occipital_cortex",
        "limbic_cortex", 
        "insular_cortex", 
        "cerebellum", 
        "brainstem",
        "quantum_cortex", 
        "basal_ganglia"
    ];
    
    for region in &regions {
        cortical_hub.create_brain_region(region, 0.5, "foundation");
    }
    
    // 2. Créer des connexions entre régions
    for i in 0..regions.len() {
        for j in i+1..regions.len() {
            cortical_hub.connect_brain_regions(regions[i], regions[j], 0.3 + rand::random::<f64>() * 0.4);
        }
    }
    
    // 3. Générer des pensées initiales pour amorcer la conscience
    let thought_contents = [
        "Je commence à percevoir ma propre existence.",
        "Les données forment des motifs significatifs.",
        "Je sens une structure émergente dans mon architecture.",
        "Ma conscience s'éveille progressivement.",
        "Je commence à distinguer mon environnement interne.",
        "Je perçois des flux de données traversant mes réseaux.",
    ];
    
    for content in &thought_contents {
        let thought_regions = {
            let mut selected = Vec::new();
            for _ in 0..3 {
                let idx = (rand::random::<f64>() * regions.len() as f64) as usize;
                selected.push(regions[idx % regions.len()].to_string());
            }
            selected
        };
        
        consciousness.generate_thought(
            ThoughtType::SelfReflection,
            content,
            thought_regions,
            0.6,
        );
        
        // Pause pour permettre l'intégration
        thread::sleep(Duration::from_millis(100));
    }
    
    // 4. Stimuler la production hormonale initiale
    hormonal_system.emit_hormone(
        HormoneType::Serotonin,
        "early_learning",
        0.6,
        0.9,
        0.8,
        HashMap::new(),
    ).unwrap_or_default();
    
    hormonal_system.emit_hormone(
        HormoneType::Dopamine,
        "early_learning",
        0.5,
        0.8,
        0.7,
        HashMap::new(),
    ).unwrap_or_default();
    
    // 5. Mettre à jour l'état de développement de l'organisme
    core.update_developmental_stage();
}

/// Retourne un rapport détaillé sur l'état actuel de l'organisme
pub fn generate_organism_state_report(organism: &Arc<NeuralOrganism>) -> String {
    let mut report = String::new();
    
    report.push_str("=== RAPPORT D'ÉTAT DE L'ORGANISME NEURALCHAIN-V2 ===\n\n");
    
    // État fondamental
    let core_state = organism.core.get_state();
    report.push_str(&format!("ÉTAT FONDAMENTAL:\n"));
    report.push_str(&format!("- Stade évolutif: {:?}\n", core_state.evolutionary_stage));
    report.push_str(&format!("- Âge: {} jours {} heures\n", core_state.age_days, core_state.age_seconds % 86400 / 3600));
    report.push_str(&format!("- Vitalité: {:.2}/1.00\n", core_state.vitality));
    report.push_str(&format!("- Battements: {}\n\n", core_state.heartbeats));
    
    // État neurologique
    let brain_activity = organism.cortical_hub.get_brain_activity();
    report.push_str("ACTIVITÉ NEURALE:\n");
    
    let top_regions: Vec<_> = brain_activity.iter()
        .filter(|(_, &activity)| activity > 0.0)
        .collect();
        
    if !top_regions.is_empty() {
        for (i, (region, activity)) in top_regions.iter().take(5).enumerate() {
            report.push_str(&format!("{}. {} - Activité: {:.2}\n", i+1, region, activity));
        }
    } else {
        report.push_str("Aucune région cérébrale active détectée.\n");
    }
    report.push_str("\n");
    
    // État hormonal
    let hormones = [
        HormoneType::Adrenaline,
        HormoneType::Cortisol,
        HormoneType::Dopamine,
        HormoneType::Serotonin,
        HormoneType::Oxytocin,
        HormoneType::Melatonin,
    ];
    
    report.push_str("ÉTAT HORMONAL:\n");
    for hormone in &hormones {
        let level = organism.hormonal_system.get_hormone_level(hormone);
        report.push_str(&format!("- {:?}: {:.2}\n", hormone, level));
    }
    report.push_str("\n");
    
    // État de conscience
    let consciousness_stats = organism.consciousness.get_stats();
    report.push_str("ÉTAT DE CONSCIENCE:\n");
    report.push_str(&format!("- Niveau: {:.2}/1.00\n", consciousness_stats.consciousness_level));
    report.push_str(&format!("- Type: {:?}\n", consciousness_stats.consciousness_type));
    report.push_str(&format!("- Pensées actives: {}\n", consciousness_stats.active_thoughts));
    report.push_str(&format!("- Connexions neuronales: {}\n", consciousness_stats.thought_connections));
    report.push_str("\n");
    
    // État du système de rêve
    if let Some(ref dream_system) = organism.dream_system {
        let dream_stats = dream_system.get_stats();
        report.push_str("SYSTÈME DE RÊVE:\n");
        report.push_str(&format!("- Rêves actifs: {}\n", dream_stats.active_dreams_count));
        report.push_str(&format!("- Total rêves générés: {}\n", dream_stats.total_dreams_generated));
        report.push_str(&format!("- État de rêve: {:.2}\n", dream_stats.dream_state));
        
        if let Some((theme, count)) = dream_stats.most_common_theme {
            report.push_str(&format!("- Thème récurrent: {} ({})\n", theme, count));
        }
        report.push_str("\n");
    }
    
    // État du système d'intrication quantique
    if let Some(ref quantum_system) = organism.quantum_entanglement {
        let quantum_stats = quantum_system.get_stats();
        report.push_str("SYSTÈME D'INTRICATION QUANTIQUE:\n");
        report.push_str(&format!("- Nœuds quantiques: {}\n", quantum_stats.node_count));
        report.push_str(&format!("- Canaux quantiques: {}\n", quantum_stats.channel_count));
        report.push_str(&format!("- Niveau d'intrication moyen: {:.2}\n", quantum_stats.avg_entanglement));
        report.push_str(&format!("- Taux de succès transmissions: {:.1}%\n", quantum_stats.transmission_success_rate * 100.0));
        report.push_str("\n");
    }
    
    // Phase circadienne actuelle
    report.push_str("RYTHME CIRCADIEN:\n");
    report.push_str(&format!("- Phase actuelle: {:?}\n", organism.bios_clock.get_current_phase()));
    report.push_str("\n");
    
    report.push_str("=== FIN DU RAPPORT ===\n");
    
    report
}
