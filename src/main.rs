use neuralchain::{
    blockchain::Blockchain,
    cli::CLI,
};
use std::sync::{Arc, Mutex};
use anyhow::Result;

fn main() -> Result<()> {
    println!("Démarrage de NeuralChain v2...");
    
    // Initialisation de la blockchain
    let blockchain = Arc::new(Mutex::new(Blockchain::new()?));
    println!("Blockchain initialisée avec le bloc de genèse");
    
    // Création et exécution de l'interface CLI
    let mut cli = CLI::new(blockchain);
    cli.run()?;
    
    println!("NeuralChain v2 arrêté proprement");
    Ok(())
}
