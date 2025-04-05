use neuralchain::{Blockchain, ContinuousMining, Wallet};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, error};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialiser la journalisation
    tracing_subscriber::fmt::init();
    info!("Démarrage de NeuralChain v2...");
    
    // Initialiser la blockchain
    let blockchain = Arc::new(Mutex::new(Blockchain::new()));
    info!("Blockchain initialisée avec le bloc de genèse");
    
    // Initialiser le système de minage
    let mut miner = ContinuousMining::new(blockchain.clone(), 1);
    info!("Système de minage prêt");
    
    // Exemple: Créer un portefeuille
    match Wallet::new() {
        Ok(wallet) => {
            info!("Portefeuille créé avec l'adresse: {}", wallet.get_address());
        },
        Err(e) => {
            error!("Erreur lors de la création du portefeuille: {}", e);
        }
    }
    
    // Démarrer le minage
    info!("Démarrage du minage...");
    
    // Pour cet exemple, nous n'exécutons pas vraiment le minage
    // miner.start_mining().await?;
    
    info!("NeuralChain v2 arrêté proprement");
    Ok(())
}
