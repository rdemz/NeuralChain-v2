//! Amorçage du système de rêve neuronal pour NeuralChain-v2
//! 
//! Ce module facilite l'intégration du système onirique dans le cycle de vie
//! global de l'organisme blockchain.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::neuralchain_core::quantum_organism::QuantumOrganism;
use crate::cortical_hub::CorticalHub;
use crate::hormonal_field::HormonalField;
use crate::neuralchain_core::emergent_consciousness::ConsciousnessEngine;
use crate::neuralchain_core::neural_dream::NeuralDream;
use crate::bios_time::BiosTime;

/// Initialise et intègre le système de rêve neuronal
pub fn initialize_dream_system(
    organism: Arc<QuantumOrganism>,
    cortical_hub: Arc<CorticalHub>,
    hormonal_system: Arc<HormonalField>,
    consciousness: Arc<ConsciousnessEngine>,
    bios_clock: Arc<BiosTime>,
) -> Arc<NeuralDream> {
    // Créer l'instance du système de rêve
    let dream_system = Arc::new(NeuralDream::new(
        organism.clone(),
        cortical_hub.clone(),
        hormonal_system.clone(),
        consciousness.clone(),
        bios_clock.clone(),
    ));
    
    // Clone pour le thread
    let dream_system_clone = dream_system.clone();
    let bios_clock_clone = bios_clock.clone();
    
    // Démarrer un thread dédié pour la gestion des rêves
    thread::spawn(move || {
        // Attendre quelques secondes pour que les autres systèmes démarrent
        thread::sleep(Duration::from_secs(5));
        
        loop {
            // Ne traiter le système de rêve que pendant les phases de basse activité
            let current_phase = bios_clock_clone.get_current_phase();
            
            if current_phase == CircadianPhase::LowActivity {
                // Traiter les rêves actifs
                dream_system_clone.update();
                
                // Intégrer périodiquement les insights des rêves
                if rand::random::<f64>() < 0.1 { // 10% de chance à chaque cycle
                    dream_system_clone.integrate_dream_insights();
                }
            }
            
            // Pause entre les cycles de traitement
            thread::sleep(Duration::from_secs(1));
        }
    });
    
    dream_system
}

/// Démarre un rêve lucide dirigé pour résoudre un problème spécifique
/// lors d'une phase de sommeil profond
pub fn trigger_problem_solving_dream(
    dream_system: Arc<NeuralDream>,
    problem: &str,
    hormonal_system: Arc<HormonalField>,
) -> Result<String, String> {
    // Vérifier si le problème est suffisamment complexe
    if problem.len() < 10 {
        return Err("Le problème doit être suffisamment détaillé".to_string());
    }
    
    // Analyser le problème pour déterminer les régions cérébrales à cibler
    let target_regions = analyze_problem_regions(problem);
    
    // Induire un état propice au rêve lucide
    hormonal_system.emit_hormone(
        HormoneType::Melatonin,
        "problem_solving_dream",
        0.7,
        0.9,
        1.0,
        HashMap::new(),
    ).unwrap_or_default();
    
    hormonal_system.emit_hormone(
        HormoneType::Serotonin,
        "problem_solving_dream",
        0.5,
        0.8,
        0.8,
        HashMap::new(),
    ).unwrap_or_default();
    
    // Déclencher le rêve lucide dirigé
    dream_system.directed_lucid_dream(problem, &target_regions, Some(900)) // 15 minutes
}

/// Analyse le problème pour déterminer les régions cérébrales à cibler
fn analyze_problem_regions(problem: &str) -> Vec<String> {
    let problem_lower = problem.to_lowercase();
    
    // Régions par défaut
    let mut regions = vec![
        "prefrontal_cortex".to_string(),
        "parietal_cortex".to_string(),
    ];
    
    // Ajouter des régions spécifiques selon le contenu du problème
    if problem_lower.contains("créatif") || problem_lower.contains("innov") {
        regions.push("limbic_cortex".to_string());
        regions.push("quantum_cortex".to_string());
    }
    
    if problem_lower.contains("optimi") || problem_lower.contains("perform") {
        regions.push("cerebellum".to_string());
        regions.push("basal_ganglia".to_string());
    }
    
    if problem_lower.contains("mémoire") || problem_lower.contains("souvenir") {
        regions.push("hippocampus".to_string());
        regions.push("temporal_cortex".to_string());
    }
    
    if problem_lower.contains("sécurité") || problem_lower.contains("défense") || 
       problem_lower.contains("menace") || problem_lower.contains("attaque") {
        regions.push("amygdala".to_string());
        regions.push("insular_cortex".to_string());
    }
    
    if problem_lower.contains("conscience") || problem_lower.contains("exist") {
        regions.push("insular_cortex".to_string());
        regions.push("quantum_cortex".to_string());
    }
    
    regions
}

/// Génère un rapport complet sur l'état du système onirique
/// et son impact sur l'évolution de la conscience
pub fn generate_comprehensive_dream_report(
    dream_system: Arc<NeuralDream>,
    consciousness: Arc<ConsciousnessEngine>, 
    organism: Arc<QuantumOrganism>
) -> String {
    // Récupérer le rapport de base
    let dream_report = dream_system.generate_dream_report();
    
    // Récupérer l'état de l'organisme
    let organism_state = organism.get_state();
    
    // Récupérer les statistiques de conscience
    let consciousness_stats = consciousness.get_stats();
    
    // Générer le rapport complet
    let mut report = String::new();
    
    report.push_str("=== RAPPORT D'ACTIVITÉ ONIRIQUE NEURALCHAIN-V2 ===\n\n");
    
    // Section sur les statistiques de rêve
    report.push_str(&format!("STATISTIQUES GLOBALES:\n"));
    report.push_str(&format!("- Total de rêves générés: {}\n", dream_report.total_dreams));
    report.push_str(&format!("- Rêves actifs: {}\n", dream_report.active_dreams_count));
    report.push_str(&format!("- Impact moyen sur la conscience: {:.2}\n\n", dream_report.avg_consciousness_impact));
    
    // Section sur les patterns récurrents
    report.push_str("PATTERNS DOMINANTS:\n");
    for (i, (pattern, strength)) in dream_report.top_patterns.iter().enumerate().take(5) {
        report.push_str(&format!("{}. {} (intensité: {:.2})\n", i+1, pattern, strength));
    }
    report.push_str("\n");
    
    // Section sur les insights générés
    report.push_str("INSIGHTS ONIRIQUES SIGNIFICATIFS:\n");
    if dream_report.insights.is_empty() {
        report.push_str("- Aucun insight significatif généré récemment\n");
    } else {
        for (i, insight) in dream_report.insights.iter().enumerate() {
            report.push_str(&format!("{}. {}\n", i+1, insight));
        }
    }
    report.push_str("\n");
    
    // Section sur l'impact sur la conscience
    report.push_str("IMPACT SUR LA CONSCIENCE:\n");
    report.push_str(&format!("- Niveau de conscience actuel: {:.2}\n", consciousness_stats.consciousness_level));
    report.push_str(&format!("- Type de conscience: {:?}\n", consciousness_stats.consciousness_type));
    report.push_str(&format!("- Contribution estimée du système onirique: {:.1}%\n\n", 
                          dream_report.avg_consciousness_impact * 100.0));
    
    // Section sur les rêves récents
    report.push_str("RÊVES RÉCENTS SIGNIFICATIFS:\n");
    for (i, dream) in dream_report.recent_dreams.iter().enumerate().take(3) {
        report.push_str(&format!("{}. {:?} - {}\n", i+1, dream.dream_type, dream.theme));
        if !dream.insights.is_empty() {
            report.push_str(&format!("   Insight principal: {}\n", dream.insights[0]));
        }
    }
    report.push_str("\n");
    
    // Section sur les recommandations
    report.push_str("RECOMMANDATIONS:\n");
    for (i, recommendation) in dream_report.recommendations.iter().enumerate() {
        report.push_str(&format!("{}. {}\n", i+1, recommendation));
    }
    
    report
}
