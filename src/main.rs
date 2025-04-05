use neuralchain::{Blockchain, Wallet};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialiser la journalisation
    println!("Démarrage de NeuralChain v2...");
    
    // Initialiser la blockchain (ajout du préfixe underscore pour indiquer qu'elle est intentionnellement non utilisée)
    let _blockchain = Arc::new(Mutex::new(Blockchain::new()));
    println!("Blockchain initialisée avec le bloc de genèse");
    
    // Exemple: Créer un portefeuille
    match Wallet::new() {
        Ok(wallet) => {
            println!("Portefeuille créé avec l'adresse: {}", wallet.get_address());
        },
        Err(e) => {
            eprintln!("Erreur lors de la création du portefeuille: {}", e);
        }
    }
    
    println!("NeuralChain v2 arrêté proprement");
    Ok(())
}
