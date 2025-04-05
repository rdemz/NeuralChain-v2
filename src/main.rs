use neuralchain::{blockchain::Blockchain, cli::CLI};
use std::sync::{Arc, Mutex};

fn main() -> anyhow::Result<()> {
    // Afficher l'en-tête
    println!("\n╔═══════════════════════════════════╗");
    println!("║      NEURALCHAIN v2 - DÉMARRAGE   ║");
    println!("╚═══════════════════════════════════╝");
    
    // Initialiser la blockchain avec gestion d'erreur
    let blockchain = match Blockchain::new() {
        Ok(bc) => {
            println!("✅ Blockchain initialisée avec le bloc de genèse");
            Arc::new(Mutex::new(bc))
        },
        Err(e) => {
            eprintln!("❌ Erreur d'initialisation de la blockchain: {}", e);
            return Err(e);
        }
    };
    
    // Création et exécution de l'interface CLI
    let mut cli = match CLI::new(blockchain) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("❌ Erreur d'initialisation de l'interface CLI: {}", e);
            return Err(e);
        }
    };
    
    // Exécuter l'interface
    if let Err(e) = cli.run() {
        eprintln!("❌ Erreur d'exécution: {}", e);
        return Err(e);
    }
    
    println!("\n╔═══════════════════════════════════╗");
    println!("║      NEURALCHAIN v2 - ARRÊT       ║");
    println!("╚═══════════════════════════════════╝");
    Ok(())
}
