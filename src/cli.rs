use crate::{
    block::Block,
    blockchain::Blockchain,
    transaction::{Transaction, TxInput, TxOutput},
    wallet::Wallet,
};
use anyhow::{Error, Result};
use std::{
    collections::HashMap,
    io::{self, Write},
    sync::{Arc, Mutex},
};

pub struct CLI {
    blockchain: Arc<Mutex<Blockchain>>,
    wallets: HashMap<String, Wallet>,
}

impl CLI {
    /// Crée une nouvelle instance de l'interface CLI
    pub fn new(blockchain: Arc<Mutex<Blockchain>>) -> Self {
        let mut wallets = HashMap::new();
        
        // Créer un portefeuille par défaut
        match Wallet::new() {
            Ok(wallet) => {
                if let Ok(address) = wallet.get_address() {
                    println!("Portefeuille créé avec l'adresse: {}", address);
                    wallets.insert(address, wallet);
                }
            },
            Err(e) => eprintln!("Erreur lors de la création du portefeuille: {}", e),
        }
        
        Self { blockchain, wallets }
    }
    
    /// Démarre la boucle principale d'interface
    pub fn run(&mut self) -> Result<()> {
        println!("Interface NeuralChain v2 démarrée");
        
        loop {
            self.print_menu();
            
            let mut choice = String::new();
            print!("> ");
            io::stdout().flush().unwrap();
            io::stdin().read_line(&mut choice)?;
            
            match choice.trim() {
                "1" => self.create_wallet()?,
                "2" => self.list_wallets(),
                "3" => self.send_transaction()?,
                "4" => self.mine_block()?,
                "5" => self.show_blockchain(),
                "6" => self.check_balance()?,
                "0" => break,
                _ => println!("Option non valide, veuillez réessayer"),
            }
        }
        
        println!("Interface NeuralChain v2 arrêtée");
        Ok(())
    }
    
    /// Affiche le menu principal
    fn print_menu(&self) {
        println!("\n=== MENU NEURALCHAIN ===");
        println!("1. Créer un nouveau portefeuille");
        println!("2. Lister les portefeuilles");
        println!("3. Envoyer une transaction");
        println!("4. Miner un bloc");
        println!("5. Afficher la blockchain");
        println!("6. Vérifier un solde");
        println!("0. Quitter");
        println!("=======================");
    }
    
    /// Crée un nouveau portefeuille et l'ajoute à la liste
    fn create_wallet(&mut self) -> Result<()> {
        let wallet = Wallet::new()?;
        let address = wallet.get_address()?;
        println!("Nouveau portefeuille créé avec l'adresse: {}", address);
        self.wallets.insert(address, wallet);
        Ok(())
    }
    
    /// Affiche la liste des portefeuilles disponibles
    fn list_wallets(&self) {
        println!("\n=== PORTEFEUILLES ===");
        if self.wallets.is_empty() {
            println!("Aucun portefeuille disponible");
        } else {
            for (i, (address, _)) in self.wallets.iter().enumerate() {
                println!("{}. {}", i+1, address);
            }
        }
    }
    
    /// Envoie une transaction entre deux adresses
    fn send_transaction(&mut self) -> Result<()> {
        // Afficher les portefeuilles disponibles
        self.list_wallets();
        
        if self.wallets.is_empty() {
            println!("Aucun portefeuille disponible pour effectuer des transactions");
            return Ok(());
        }
        
        // Sélectionner le portefeuille source
        println!("Choisissez le portefeuille source (par numéro):");
        let mut choice = String::new();
        print!("> ");
        io::stdout().flush()?;
        io::stdin().read_line(&mut choice)?;
        
        let addresses: Vec<String> = self.wallets.keys().cloned().collect();
        let index: usize = match choice.trim().parse::<usize>() {
            Ok(num) if num > 0 && num <= addresses.len() => num - 1,
            _ => return Err(Error::msg("Choix invalide")),
        };
        
        let from_address = addresses[index].clone();
        
        // Demander l'adresse destinataire
        println!("Entrez l'adresse du destinataire:");
        let mut to_address = String::new();
        print!("> ");
        io::stdout().flush()?;
        io::stdin().read_line(&mut to_address)?;
        to_address = to_address.trim().to_string();
        
        // Demander le montant
        println!("Entrez le montant à envoyer:");
        let mut amount_str = String::new();
        print!("> ");
        io::stdout().flush()?;
        io::stdin().read_line(&mut amount_str)?;
        let amount: u64 = amount_str.trim().parse()?;
        
        // Création de la transaction (exemple simplifié)
        println!("Création d'une transaction de {} vers {}, montant: {}", from_address, to_address, amount);
        
        // Ceci est une implémentation simplifiée
        let tx_input = TxInput {
            tx_id: vec![0; 32], // Dans une vraie implémentation, on utiliserait l'ID d'une UTXO
            vout: 0,
            signature: vec![],  // Sera signé plus tard
        };
        
        let tx_output = TxOutput {
            value: amount,
            pub_key_hash: to_address.into_bytes(), // Simplifié
        };
        
        let mut transaction = Transaction {
            id: vec![],
            vin: vec![tx_input],
            vout: vec![tx_output],
        };
        
        // Calcul de l'ID
        transaction.set_id()?;
        
        // Signer la transaction avec le portefeuille source
        if let Some(wallet) = self.wallets.get(&from_address) {
            // Dans une implémentation réelle, vous signeriez la transaction ici
            println!("Transaction signée et prête à être ajoutée au prochain bloc");
        } else {
            return Err(Error::msg("Portefeuille source non trouvé"));
        }
        
        // Dans une implémentation réelle, vous ajouteriez la transaction à un pool
        
        Ok(())
    }
    
    /// Mine un bloc avec les transactions disponibles
    fn mine_block(&mut self) -> Result<()> {
        println!("Minage d'un nouveau bloc...");
        
        // Dans un cas réel, vous récupéreriez les transactions de la mempool
        let transactions = vec![]; // Transactions vides pour cet exemple
        
        let mut bc = self.blockchain.lock().map_err(|_| Error::msg("Erreur de lock sur la blockchain"))?;
        bc.add_block(transactions)?;
        
        println!("Nouveau bloc miné avec succès!");
        Ok(())
    }
    
    /// Affiche l'état actuel de la blockchain
    fn show_blockchain(&self) {
        let bc = match self.blockchain.lock() {
            Ok(bc) => bc,
            Err(_) => {
                println!("Erreur lors de l'accès à la blockchain");
                return;
            }
        };
        
        println!("\n=== BLOCKCHAIN ===");
        for (i, block) in bc.blocks.iter().enumerate() {
            println!("Bloc #{}", i);
            println!("  Hash: {}", hex::encode(&block.hash));
            println!("  Hash précédent: {}", hex::encode(&block.prev_hash));
            println!("  Nonce: {}", block.nonce);
            println!("  Difficulté: {}", block.difficulty);
            println!("  Transactions: {}", block.transactions.len());
            println!("  Horodatage: {}", block.timestamp);
        }
    }
    
    /// Vérifie le solde d'une adresse
    fn check_balance(&self) -> Result<()> {
        self.list_wallets();
        
        if self.wallets.is_empty() {
            println!("Aucun portefeuille disponible");
            return Ok(());
        }
        
        println!("Choisissez le portefeuille pour vérifier le solde (par numéro):");
        let mut choice = String::new();
        print!("> ");
        io::stdout().flush()?;
        io::stdin().read_line(&mut choice)?;
        
        let addresses: Vec<String> = self.wallets.keys().cloned().collect();
        let index: usize = match choice.trim().parse::<usize>() {
            Ok(num) if num > 0 && num <= addresses.len() => num - 1,
            _ => return Err(Error::msg("Choix invalide")),
        };
        
        let address = &addresses[index];
        
        // Dans une implémentation réelle, calculer le solde avec les UTXOs
        // Pour cet exemple, nous affichons un solde fictif
        println!("Le solde de l'adresse {} est: 100", address);
        
        Ok(())
    }
}
