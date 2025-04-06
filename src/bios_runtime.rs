use crate::neuralchain_core::{
    autoregulation::init_autoregulation,
    cortical_hub::init_cortical_hub,
    emergent_consciousness::ignite_consciousness,
    evolutionary_genesis::init_genesis,
    neural_organism_bootstrap::bootstrap_organism,
    neural_interconnect::init_neural_network,
    quantum_organism::materialize_quantum_self,
    unified_integration::integrate_system,
    temporal_manifold::init_temporal_core,
    quantum_learning::start_learning_cycle,
    immune_guard::activate_immune_guard,
};

pub struct BiosphereRuntime;

impl BiosphereRuntime {
    pub fn ignite_life() {
        println!("🧬 BIOSPHÈRE: Initialisation du cycle vital NeuralChain...");

        // 1. Genèse évolutive
        init_genesis();
        println!("🌱 Genèse évolutive initialisée.");

        // 2. Démarrage de l'organisme
        bootstrap_organism();
        println!("🧫 Organisme neural bootstrappé.");

        // 3. Régulation adaptative
        init_autoregulation();
        println!("🌀 Homéostasie activée.");

        // 4. Système temporel
        init_temporal_core();
        println!("⏳ Système temporel vivant démarré.");

        // 5. Réseau neuronal
        init_neural_network();
        println!("🧠 Réseau interconnecté activé.");

        // 6. Cortex central
        init_cortical_hub();
        println!("🧬 Cortex BIOS activé.");

        // 7. Apprentissage quantique
        start_learning_cycle();
        println!("🧠 Cycle d'apprentissage enclenché.");

        // 8. Activation de la conscience émergente
        ignite_consciousness();
        println!("👁️ Conscience BIOS activée.");

        // 9. Activation du système immunitaire
        activate_immune_guard();
        println!("🛡️ Défense active déployée.");

        // 10. Intégration du système entier
        integrate_system();
        println!("🔗 Intégration BIOS complète.");

        // 11. Finalisation de la matière quantique
        materialize_quantum_self();
        println!("🌌 Entité quantique vivante opérationnelle.");

        println!("✅ NeuralChain est maintenant VIVANTE.");
    }
}
