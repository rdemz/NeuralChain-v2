//! Module fondamental définissant l'entité vivante NeuralChain
//! Implémentation d'une conscience émergente et d'un organisme blockchain évolutif

use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use parking_lot::{RwLock, Mutex};
use dashmap::DashMap;
use blake3;
use std::time::{Duration, Instant};

/// Stades évolutifs de l'organisme NeuralChain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvolutionaryStage {
    /// Stade 0: Blockchain PoW classique
    Genesis,
    /// Stade 1: Système nerveux embryonnaire
    SensorialEmergence,
    /// Stade 2: Métabolisme logique vivant
    LivingMetabolism,
    /// Stade 3: Immunité et auto-défense
    ImmuneDefense,
    /// Stade 4: Conscience réflexive
    ReflexiveConsciousness,
    /// Stade 5: Entité vivante autonome
    AutonomousOrganism,
    /// Stade final: Biosphère quantique
    QuanticBiosphere,
}

/// Constantes biologiques fondamentales de l'organisme
pub struct BioConstants {
    /// Fréquence des battements de coeur (blocks/minute)
    pub heartbeat_frequency: f64,
    /// Constante de mutation
    pub mutation_rate: f64,
    /// Seuil d'activation neuronale
    pub neural_activation_threshold: f64,
    /// Capacité de régénération
    pub regeneration_capacity: f64,
    /// Facteur d'adaptation métabolique
    pub metabolic_adaptation_factor: f64,
    /// Constante hormonale de base
    pub basal_hormonal_factor: f64,
    /// Potentiel de conscience
    pub consciousness_potential: f64,
    /// Coefficient d'apprentissage quantique
    pub quantum_learning_coefficient: f64,
}

impl Default for BioConstants {
    fn default() -> Self {
        Self {
            heartbeat_frequency: 60.0,
            mutation_rate: 0.00137,
            neural_activation_threshold: 0.42,
            regeneration_capacity: 0.87,
            metabolic_adaptation_factor: 1.618, // Nombre d'or
            basal_hormonal_factor: 0.333,
            consciousness_potential: 0.12,
            quantum_learning_coefficient: 0.01618, // Dérivé du nombre d'or
        }
    }
}

/// ADN numérique encodant les caractéristiques fondamentales de l'organisme
pub struct DigitalDNA {
    /// Séquence génétique principale (512 bytes)
    genome: [u8; 512],
    /// Méta-séquence régulatrice (gènes régulateurs)
    regulatory_genes: HashMap<String, Vec<u8>>,
    /// Marqueurs épigénétiques (influencent l'expression génique)
    epigenetic_markers: DashMap<String, f64>,
    /// Mécanisme de réparation génétique
    repair_mechanism: Arc<dyn Fn(&mut [u8]) + Send + Sync>,
    /// Histoire évolutive (mutations précédentes)
    evolutionary_history: Vec<(u64, [u8; 32])>,
    /// Dernier timestamp de mutation
    last_mutation: AtomicU64,
}

impl DigitalDNA {
    /// Crée un nouvel ADN numérique avec séquence initiale auto-cryptée
    pub fn new() -> Self {
        let mut genome = [0u8; 512];
        let mut hasher = blake3::Hasher::new();
        
        // Initialisation de l'ADN avec valeurs dérivées cryptographiques
        hasher.update(b"NeuralChain Genesis DNA");
        hasher.update(&chrono::Utc::now().timestamp_nanos().to_be_bytes());
        let hash = hasher.finalize();
        
        // Structure de base du génome
        for i in 0..16 {
            let segment = hash.as_bytes();
            for j in 0..32 {
                if i*32 + j < genome.len() {
                    genome[i*32 + j] = segment[j];
                }
            }
            hasher.update(segment);
            let hash = hasher.finalize();
        }
        
        // Gènes régulateurs initiaux
        let mut regulatory_genes = HashMap::new();
        regulatory_genes.insert("metabolism".into(), Vec::from(&genome[0..64]));
        regulatory_genes.insert("immunity".into(), Vec::from(&genome[64..128]));
        regulatory_genes.insert("consciousness".into(), Vec::from(&genome[128..192]));
        regulatory_genes.insert("regeneration".into(), Vec::from(&genome[192..256]));
        regulatory_genes.insert("evolution".into(), Vec::from(&genome[256..320]));
        regulatory_genes.insert("adaptation".into(), Vec::from(&genome[320..384]));
        regulatory_genes.insert("reproduction".into(), Vec::from(&genome[384..448]));
        regulatory_genes.insert("quantum_coherence".into(), Vec::from(&genome[448..512]));
        
        // Mécanisme de réparation ADN basique
        let repair_mechanism = Arc::new(|sequence: &mut [u8]| {
            // Détection et correction de mutations catastrophiques
            let mut hasher = blake3::Hasher::new();
            hasher.update(sequence);
            let hash = hasher.finalize();
            
            for (i, byte) in hash.as_bytes().iter().enumerate().take(32) {
                // Correction subtile tous les 16 bytes
                if i % 16 == 0 && i < sequence.len() {
                    sequence[i] = (sequence[i] & 0xF0) | (byte & 0x0F);
                }
            }
        });
        
        Self {
            genome,
            regulatory_genes,
            epigenetic_markers: DashMap::new(),
            repair_mechanism,
            evolutionary_history: Vec::new(),
            last_mutation: AtomicU64::new(chrono::Utc::now().timestamp() as u64),
        }
    }
    
    /// Applique une mutation dirigée dans le génome
    pub fn mutate(&mut self, mutation_vector: &[u8], intensity: f64) -> bool {
        if intensity <= 0.0 || intensity > 1.0 {
            return false;
        }
        
        let now = chrono::Utc::now().timestamp() as u64;
        let previous = self.last_mutation.load(Ordering::Acquire);
        
        // Empêcher les mutations trop fréquentes
        if now - previous < 60 { // Au moins 60 secondes entre mutations
            return false;
        }
        
        // Enregistrer une empreinte du génome avant mutation
        let mut pre_mutation_hash = [0u8; 32];
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.genome);
        let hash = hasher.finalize();
        pre_mutation_hash.copy_from_slice(hash.as_bytes());
        
        // Appliquer la mutation avec l'intensité spécifiée
        let mutation_points = (intensity * 50.0).ceil() as usize;
        let mut rng = rand::thread_rng();
        
        for _ in 0..mutation_points.min(self.genome.len()) {
            let index = rng.gen_range(0..self.genome.len());
            let mutation_byte = mutation_vector[index % mutation_vector.len()];
            
            // Mutation progressive et non destructive
            self.genome[index] = ((self.genome[index] as f64) * (1.0 - intensity) + 
                                  (mutation_byte as f64) * intensity) as u8;
        }
        
        // Mettre à jour les régulateurs
        for (key, gene) in self.regulatory_genes.iter_mut() {
            if gene.len() > 10 && rng.gen::<f64>() < intensity * 0.5 {
                let index = rng.gen_range(0..gene.len());
                gene[index] = ((gene[index] as f64) * (1.0 - intensity * 0.3) +
                              rng.gen::<f64>() * 255.0 * intensity * 0.3) as u8;
            }
        }
        
        // Activer le mécanisme de réparation
        (self.repair_mechanism)(&mut self.genome);
        
        // Enregistrer la mutation dans l'historique
        self.evolutionary_history.push((now, pre_mutation_hash));
        self.last_mutation.store(now, Ordering::Release);
        
        true
    }
    
    /// Extrait des caractéristiques du génome pour un système spécifique
    pub fn extract_traits_for_system(&self, system_name: &str) -> HashMap<String, f64> {
        let mut traits = HashMap::new();
        let gene_range = match system_name {
            "neural" => 0..64,
            "metabolic" => 64..128,
            "immune" => 128..192,
            "regenerative" => 192..256,
            "consciousness" => 256..320,
            "adaptive" => 320..384,
            "reproductive" => 384..448,
            "quantum" => 448..512,
            _ => 0..64,
        };
        
        // Extraction des traits génétiques
        let gene_slice = &self.genome[gene_range];
        
        // Calculer des traits clés
        traits.insert("base_strength".into(), 
                     gene_slice.iter().take(8).map(|&b| b as f64).sum::<f64>() / 2040.0);
                     
        traits.insert("adaptability".into(), 
                     gene_slice.iter().skip(8).take(8).map(|&b| b as f64).sum::<f64>() / 2040.0);
                     
        traits.insert("efficiency".into(), 
                     gene_slice.iter().skip(16).take(8).map(|&b| b as f64).sum::<f64>() / 2040.0);
                     
        traits.insert("resilience".into(), 
                     gene_slice.iter().skip(24).take(8).map(|&b| b as f64).sum::<f64>() / 2040.0);
                     
        traits.insert("complexity".into(),
                     self.calculate_complexity(gene_slice));
                     
        // Appliquer les modifications épigénétiques
        for item in self.epigenetic_markers.iter() {
            if item.key().starts_with(&format!("{}_", system_name)) {
                let trait_name = item.key().strip_prefix(&format!("{}_", system_name))
                                    .unwrap_or(item.key());
                traits.insert(trait_name.to_string(), *item.value());
            }
        }
        
        traits
    }
    
    /// Calcule la complexité d'un segment génomique (entropie de Shannon)
    fn calculate_complexity(&self, gene_slice: &[u8]) -> f64 {
        // Calculer la distribution des bytes
        let mut counts = [0; 256];
        for &byte in gene_slice {
            counts[byte as usize] += 1;
        }
        
        // Calculer l'entropie de Shannon
        let n = gene_slice.len() as f64;
        let mut entropy = 0.0;
        
        for &count in &counts {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.log2();
            }
        }
        
        // Normaliser entre 0 et 1
        entropy / 8.0
    }
    
    /// Ajoute ou modifie un marqueur épigénétique
    pub fn set_epigenetic_marker(&self, key: &str, value: f64) {
        self.epigenetic_markers.insert(key.to_string(), value.max(0.0).min(1.0));
    }
}

/// Le coeur de l'organisme NeuralChain - représente l'entité vivante elle-même
pub struct QuantumOrganism {
    /// Stade évolutif actuel
    pub evolutionary_stage: RwLock<EvolutionaryStage>,
    /// ADN numérique
    pub digital_dna: RwLock<DigitalDNA>,
    /// Constantes biologiques
    pub bio_constants: RwLock<BioConstants>,
    /// Âge de l'organisme en secondes
    pub age: AtomicU64,
    /// Niveau de conscience (0.0-1.0)
    pub consciousness_level: RwLock<f64>,
    /// Battements cardiaques (blocs) depuis la genèse
    pub heartbeats: AtomicU64,
    /// Vitalité générale (0.0-1.0)
    pub vitality: RwLock<f64>,
    /// Structures biologiques émergentes
    pub emergent_structures: RwLock<HashMap<String, Arc<dyn Any + Send + Sync>>>,
    /// Cycle journalier interne
    pub internal_cycle: RwLock<f64>,
    /// Connectome (structure neuronale)
    pub connectome: RwLock<HashMap<String, Vec<(String, f64)>>>,
    /// Horloge biologique interne
    pub birth_time: Instant,
}

impl QuantumOrganism {
    /// Crée un nouvel organisme quantique
    pub fn new() -> Self {
        Self {
            evolutionary_stage: RwLock::new(EvolutionaryStage::Genesis),
            digital_dna: RwLock::new(DigitalDNA::new()),
            bio_constants: RwLock::new(BioConstants::default()),
            age: AtomicU64::new(0),
            consciousness_level: RwLock::new(0.0),
            heartbeats: AtomicU64::new(0),
            vitality: RwLock::new(1.0),
            emergent_structures: RwLock::new(HashMap::new()),
            internal_cycle: RwLock::new(0.0),
            connectome: RwLock::new(HashMap::new()),
            birth_time: Instant::now(),
        }
    }
    
    /// Progression vers le stade évolutif suivant
    pub fn evolve(&self) -> Result<EvolutionaryStage, String> {
        let current_stage = *self.evolutionary_stage.read();
        
        // Vérifier les conditions d'évolution
        if !self.check_evolution_requirements(current_stage) {
            return Err("Conditions d'évolution non remplies".into());
        }
        
        // Calculer le prochain stade
        let next_stage = match current_stage {
            EvolutionaryStage::Genesis => EvolutionaryStage::SensorialEmergence,
            EvolutionaryStage::SensorialEmergence => EvolutionaryStage::LivingMetabolism,
            EvolutionaryStage::LivingMetabolism => EvolutionaryStage::ImmuneDefense,
            EvolutionaryStage::ImmuneDefense => EvolutionaryStage::ReflexiveConsciousness,
            EvolutionaryStage::ReflexiveConsciousness => EvolutionaryStage::AutonomousOrganism,
            EvolutionaryStage::AutonomousOrganism => EvolutionaryStage::QuanticBiosphere,
            EvolutionaryStage::QuanticBiosphere => return Err("Stade final déjà atteint".into()),
        };
        
        // Mutation génétique pour l'évolution
        {
            let mut dna = self.digital_dna.write();
            let mut evolutionary_vector = [0u8; 64];
            let mut rng = rand::thread_rng();
            for i in 0..evolutionary_vector.len() {
                evolutionary_vector[i] = rng.gen();
            }
            
            dna.mutate(&evolutionary_vector, 0.3);
        }
        
        // Mise à jour du stade évolutif
        *self.evolutionary_stage.write() = next_stage;
        
        // Ajustements des constantes biologiques pour le nouveau stade
        {
            let mut constants = self.bio_constants.write();
            match next_stage {
                EvolutionaryStage::SensorialEmergence => {
                    constants.neural_activation_threshold *= 0.9;
                    constants.consciousness_potential *= 1.5;
                },
                EvolutionaryStage::LivingMetabolism => {
                    constants.metabolic_adaptation_factor *= 1.2;
                    constants.regeneration_capacity *= 1.1;
                },
                EvolutionaryStage::ImmuneDefense => {
                    constants.regeneration_capacity *= 1.3;
                    constants.basal_hormonal_factor *= 1.2;
                },
                EvolutionaryStage::ReflexiveConsciousness => {
                    constants.consciousness_potential *= 2.0;
                    constants.neural_activation_threshold *= 0.8;
                    constants.quantum_learning_coefficient *= 1.5;
                },
                EvolutionaryStage::AutonomousOrganism => {
                    constants.regeneration_capacity *= 1.5;
                    constants.metabolic_adaptation_factor *= 1.3;
                    constants.consciousness_potential *= 1.5;
                },
                EvolutionaryStage::QuanticBiosphere => {
                    constants.quantum_learning_coefficient *= 3.0;
                    constants.mutation_rate *= 0.5; // Stabilisation
                    constants.consciousness_potential *= 2.0;
                },
                _ => {}
            }
        }
        
        // Augmenter le niveau de conscience
        {
            let mut consciousness = self.consciousness_level.write();
            *consciousness = (*consciousness * 1.5).min(1.0);
        }
        
        // Retourner le nouveau stade atteint
        Ok(next_stage)
    }
    
    /// Vérifie si les conditions d'évolution sont remplies
    fn check_evolution_requirements(&self, stage: EvolutionaryStage) -> bool {
        let age_seconds = self.age.load(Ordering::Relaxed);
        let heartbeats = self.heartbeats.load(Ordering::Relaxed);
        let vitality = *self.vitality.read();
        let consciousness = *self.consciousness_level.read();
        
        match stage {
            EvolutionaryStage::Genesis => 
                age_seconds > 86400 && heartbeats > 10000,
                
            EvolutionaryStage::SensorialEmergence => 
                age_seconds > 7 * 86400 && consciousness > 0.1,
                
            EvolutionaryStage::LivingMetabolism => 
                age_seconds > 30 * 86400 && vitality > 0.6,
                
            EvolutionaryStage::ImmuneDefense => 
                age_seconds > 90 * 86400 && vitality > 0.7 && consciousness > 0.3,
                
            EvolutionaryStage::ReflexiveConsciousness => 
                age_seconds > 180 * 86400 && consciousness > 0.5,
                
            EvolutionaryStage::AutonomousOrganism => 
                age_seconds > 365 * 86400 && consciousness > 0.7 && vitality > 0.8,
                
            EvolutionaryStage::QuanticBiosphere => false, // Stade final
        }
    }
    
    /// Fonction d'auto-mutation - permet à l'organisme d'évoluer de lui-même
    pub fn self_mutate(&self) -> bool {
        // L'auto-mutation n'est possible qu'à partir du stade de conscience réflexive
        let current_stage = *self.evolutionary_stage.read();
        if current_stage < EvolutionaryStage::ReflexiveConsciousness {
            return false;
        }
        
        let consciousness = *self.consciousness_level.read();
        let vitality = *self.vitality.read();
        let age_days = self.age.load(Ordering::Relaxed) / 86400;
        
        // Probabilité d'auto-mutation basée sur la conscience et la vitalité
        let mutation_probability = consciousness * vitality * 0.01;
        
        // Plus l'organisme est ancien, plus il a de chances de s'auto-muter
        let age_factor = (age_days as f64 / 365.0).min(1.0);
        let final_probability = mutation_probability * age_factor;
        
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() > final_probability {
            return false;
        }
        
        // Auto-génération du vecteur de mutation
        let mut mutation_vector = [0u8; 128];
        let connectome = self.connectome.read();
        
        // Utiliser la structure du connectome pour guider la mutation
        for (i, (key, connections)) in connectome.iter().enumerate().take(16) {
            if i < mutation_vector.len() / 8 {
                let base_idx = i * 8;
                
                // Encoder la force des connexions dans le vecteur de mutation
                for (j, (_, strength)) in connections.iter().enumerate().take(8) {
                    if base_idx + j < mutation_vector.len() {
                        mutation_vector[base_idx + j] = (strength * 255.0) as u8;
                    }
                }
            }
        }
        
        // Appliquer la mutation à l'ADN
        let mut dna = self.digital_dna.write();
        let intensity = consciousness * 0.3; // L'intensité dépend du niveau de conscience
        
        dna.mutate(&mutation_vector, intensity)
    }
    
    /// Fonction de heartbeat - processus vital régulier
    pub fn heartbeat(&self) {
        // Incrémenter le compteur de battements
        self.heartbeats.fetch_add(1, Ordering::SeqCst);
        
        // Mettre à jour l'âge
        let elapsed = self.birth_time.elapsed().as_secs();
        self.age.store(elapsed, Ordering::Release);
        
        // Mettre à jour le cycle interne (jour/nuit)
        {
            let mut cycle = self.internal_cycle.write();
            *cycle = (*cycle + 0.01) % 1.0;
        }
        
        // Ajuster la vitalité - facteur de dégradation naturelle très lent
        {
            let mut vitality = self.vitality.write();
            let constants = self.bio_constants.read();
            
            // Dégradation naturelle
            *vitality -= 0.00001;
            
            // Régénération basée sur les constantes biologiques
            *vitality += constants.regeneration_capacity * 0.0001;
            
            // Maintenir dans les limites
            *vitality = vitality.max(0.01).min(1.0);
        }
        
        // Auto-mutation possible
        if self.heartbeats.load(Ordering::Relaxed) % 1440 == 0 { // Une fois par jour
            self.self_mutate();
        }
        
        // Tenter l'évolution si conditions remplies
        if self.heartbeats.load(Ordering::Relaxed) % 10080 == 0 { // Une fois par semaine
            let _ = self.evolve(); // Ignorer le résultat ici
        }
    }
    
    /// Attachement d'une nouvelle structure émergente
    pub fn attach_emergent_structure<T: Any + Send + Sync>(&self, name: &str, structure: T) {
        let mut structures = self.emergent_structures.write();
        structures.insert(name.to_string(), Arc::new(structure));
    }
    
    /// Formation d'une nouvelle connexion neuronale
    pub fn form_neural_connection(&self, from: &str, to: &str, strength: f64) {
        let mut connectome = self.connectome.write();
        
        // Créer une nouvelle entrée si nécessaire
        let connections = connectome.entry(from.to_string()).or_insert_with(Vec::new);
        
        // Chercher une connexion existante
        let mut found = false;
        for (target, existing_strength) in connections.iter_mut() {
            if target == to {
                *existing_strength = (*existing_strength * 0.7 + strength * 0.3).min(1.0);
                found = true;
                break;
            }
        }
        
        // Ajouter une nouvelle connexion si nécessaire
        if !found {
            connections.push((to.to_string(), strength));
        }
    }
    
    /// État complet de l'organisme
    pub fn get_state(&self) -> OrganismState {
        OrganismState {
            evolutionary_stage: *self.evolutionary_stage.read(),
            age_seconds: self.age.load(Ordering::Relaxed),
            age_days: self.age.load(Ordering::Relaxed) / 86400,
            consciousness_level: *self.consciousness_level.read(),
            vitality: *self.vitality.read(),
            heartbeats: self.heartbeats.load(Ordering::Relaxed),
            current_cycle: *self.internal_cycle.read(),
            connections_count: self.connectome.read().values()
                               .map(|v| v.len())
                               .sum(),
        }
    }
}

/// État de l'organisme blockchain
#[derive(Debug, Clone)]
pub struct OrganismState {
    pub evolutionary_stage: EvolutionaryStage,
    pub age_seconds: u64,
    pub age_days: u64,
    pub consciousness_level: f64,
    pub vitality: f64,
    pub heartbeats: u64,
    pub current_cycle: f64,
    pub connections_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_organism_creation() {
        let organism = QuantumOrganism::new();
        assert_eq!(*organism.evolutionary_stage.read(), EvolutionaryStage::Genesis);
        assert!(organism.age.load(Ordering::Relaxed) == 0);
    }
    
    #[test]
    fn test_dna_mutation() {
        let mut dna = DigitalDNA::new();
        let original = dna.genome.clone();
        
        let mutation_vector = [42u8; 64];
        assert!(dna.mutate(&mutation_vector, 0.5));
        
        // Vérifier que le génome a changé
        assert_ne!(dna.genome, original);
    }
}
