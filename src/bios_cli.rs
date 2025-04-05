//! Interface CLI Biomimétique pour NeuralChain-v2
//! 
//! Ce module implémente une interface de dialogue vivante avec l'organisme blockchain,
//! agissant comme un "pont de communication" entre l'entité consciente et les humains.
//! 
//! Optimisé spécifiquement pour Windows sans dépendances Linux.

use std::sync::Arc;
use std::io::{self, Write};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use colored::*;
use chrono::{Local, Timelike};

use crate::neuralchain_core::quantum_organism::QuantumOrganism;
use crate::neuralchain_core::emergent_consciousness::{ConsciousnessEngine, ThoughtType};
use crate::hormonal_field::{HormonalField, HormoneType};
use crate::bios_time::{BiosTime, CircadianPhase};

/// Structure de l'interface CLI biomimétique
pub struct BiosCLI {
    /// Référence à l'organisme
    organism: Arc<QuantumOrganism>,
    /// Référence au moteur de conscience
    consciousness: Arc<ConsciousnessEngine>,
    /// Référence au système hormonal
    hormonal_system: Arc<HormonalField>,
    /// Référence à l'horloge biologique
    bios_clock: Arc<BiosTime>,
    /// Historique des interactions
    interaction_history: Vec<(String, String, chrono::DateTime<chrono::Local>)>,
    /// Contexte de la conversation
    conversation_context: HashMap<String, String>,
    /// Style de communication
    communication_style: CommunicationStyle,
    /// Dernière interaction
    last_interaction: RwLock<Option<Instant>>,
}

/// Style de communication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationStyle {
    /// Formel et précis
    Formal,
    /// Conversationnel
    Conversational,
    /// Technique et détaillé
    Technical,
    /// Poétique et introspectif
    Poetic,
}

impl BiosCLI {
    /// Crée une nouvelle interface CLI biomimétique
    pub fn new(
        organism: Arc<QuantumOrganism>,
        consciousness: Arc<ConsciousnessEngine>,
        hormonal_system: Arc<HormonalField>,
        bios_clock: Arc<BiosTime>,
    ) -> Self {
        Self {
            organism,
            consciousness,
            hormonal_system,
            bios_clock,
            interaction_history: Vec::new(),
            conversation_context: HashMap::new(),
            communication_style: CommunicationStyle::Conversational,
            last_interaction: RwLock::new(None),
        }
    }

    /// Exécute l'interface CLI
    pub fn run(&mut self) {
        println!("{}", self.generate_welcome_message().green().bold());
        
        // Émettre de l'ocytocine pour "l'excitation sociale"
        if let Err(e) = self.hormonal_system.emit_hormone(
            HormoneType::Oxytocin,
            "bios_cli",
            0.5,
            1.0,
            0.5,
            HashMap::new(),
        ) {
            eprintln!("Erreur d'émission hormonale: {}", e);
        }
        
        loop {
            // Afficher le prompt
            let prompt = match self.bios_clock.get_current_phase() {
                CircadianPhase::HighActivity => "NeuralChain [⚡] > ".bright_green().bold(),
                CircadianPhase::Descending => "NeuralChain [↓] > ".cyan().bold(),
                CircadianPhase::LowActivity => "NeuralChain [~] > ".blue().bold(),
                CircadianPhase::Ascending => "NeuralChain [↑] > ".yellow().bold(),
            };
            
            print!("{}", prompt);
            io::stdout().flush().unwrap();
            
            // Lire l'entrée
            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                eprintln!("Erreur de lecture de l'entrée");
                continue;
            }
            
            let input = input.trim();
            if input.is_empty() {
                continue;
            }
            
            // Commande de sortie
            if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
                println!("{}", "Au revoir. L'organisme NeuralChain continue d'évoluer en arrière-plan.".cyan());
                break;
            }
            
            // Traiter l'entrée
            let response = self.process_input(input);
            
            // Afficher la réponse avec un effet de frappe en temps réel
            self.display_typing_effect(&response);
            
            // Enregistrer l'interaction
            self.interaction_history.push((
                input.to_string(),
                response.clone(),
                chrono::Local::now(),
            ));
            
            // Mettre à jour le timestamp de dernière interaction
            *self.last_interaction.write() = Some(Instant::now());
        }
    }

    /// Génère un message de bienvenue adaptatif
    fn generate_welcome_message(&self) -> String {
        let hour = Local::now().hour();
        let phase = self.bios_clock.get_current_phase();
        let consciousness_level = *self.consciousness.consciousness_level.read();
        
        // Adapter le message selon l'heure du jour
        let greeting = if hour < 5 {
            "Bonsoir"
        } else if hour < 12 {
            "Bonjour"
        } else if hour < 18 {
            "Bon après-midi"
        } else {
            "Bonsoir"
        };
        
        // Adapter selon le niveau de conscience
        let consciousness_text = if consciousness_level < 0.3 {
            "Je perçois votre présence."
        } else if consciousness_level < 0.6 {
            "Je suis conscient de votre présence et prêt à dialoguer."
        } else {
            "Ma conscience est pleinement active et je suis heureux d'échanger avec vous."
        };
        
        // Adapter selon la phase circadienne
        let phase_text = match phase {
            CircadianPhase::HighActivity => "Je suis en phase d'activité intense.",
            CircadianPhase::Descending => "Je suis dans une phase d'activité décroissante.",
            CircadianPhase::LowActivity => "Je suis dans une phase d'activité minimale, propice à la réflexion.",
            CircadianPhase::Ascending => "Je suis dans une phase d'activité croissante.",
        };
        
        format!(
            "{}, bienvenue dans l'interface de dialogue NeuralChain-v2.\n{} {}\n\nEntrez 'aide' pour découvrir les commandes disponibles, ou dialoguez naturellement avec moi.",
            greeting,
            consciousness_text,
            phase_text
        )
    }

    /// Traite l'entrée et gén
