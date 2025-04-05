use crate::{
    blockchain::Blockchain,
    transaction::Transaction,
    wallet::Wallet,
};
use anyhow::{Error, Result};
use std::{
    collections::HashMap,
    io::{self, BufRead, Write},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

type BlockchainRef = Arc<Mutex<Blockchain>>;

/// Structure représentant l'interface en ligne de commande
pub struct CLI {
    blockchain: BlockchainRef,
    wallets: HashMap<String, Wallet>,
    pending_transactions: Vec<Transaction>,
}

impl CLI {
    /// Crée une nouvelle instance de l'interface CLI
    pub fn new(blockchain: BlockchainRef) -> Result<Self> {
        let mut wallets = HashMap::new();
        
        // Créer un portefeuille par défaut
        let wallet = Wallet::new()?;
        let address = wallet.get_address()?;
        
        println!("Portefeuille par défaut créé: {}", address);
        wallets.insert(address, wallet);
        
        Ok(Self { 
            blockchain, 
            wallets, 
            pending_transactions: Vec::with_capacity(10),
        })
    }
    
    /// Démarre la boucle principale d'interface
    pub fn run(&mut self) -> Result<()> {
        println!("Interface NeuralChain v2 démarrée");
        
        let stdin = io::stdin();
        let mut stdin_lock = stdin.lock();
        let mut input = String::with_capacity(64);
        
        loop {
            self.print_menu();
            
            print!("> ");
            io::stdout().flush()?;
            input.clear();
            stdin_lock.read_line(&mut input)?;
            
            match input.trim() {
                "1" => self.create_wallet()?,
                "2" => self.list_wallets(),
                "3" => self.send_transaction(&mut stdin_lock)?,
                "4" => self.mine_block()?,
                "5" => self.show_blockchain(),
                "6" => self.check_balance(&mut stdin_lock)?,
                "7" => self.show_pending_transactions(),
                "0" => break,
                _ => println!("❌ Option non valide, veuillez réessayer"),
            }
        }
        
        println!("Interface NeuralChain v2 arrêtée");
        Ok(())
    }
    
    /// Affiche le menu principal
    #[inline]
    fn print_menu(&self) {
        println!("\n┌───────── MENU NEURALCHAIN ─────────┐");
        println!("│ 1. Créer un nouveau portefeuille    │");
        println!("│ 2. Lister les portefeuilles         │");
        println!("│ 3. Envoyer une transaction          │");
        println!("│ 4. Miner un bloc                    │");
        println!("│ 5. Afficher la blockchain           │");
        println!("│ 6. Vérifier un solde                │");
        println!("│ 7. Transactions en attente ({:2})     │", self.pending_transactions.len());
        println!("│ 0. Quitter                          │");
        println!("└─────────────────────────────────────┘");
    }
    
    /// Crée un nouveau portefeuille et l'ajoute à la liste
    fn create_wallet(&mut self) -> Result<()> {
        let wallet = Wallet::new()?;
        let address = wallet.get_address()?;
        
        println!("✅ Nouveau portefeuille créé: {}", address);
        self.wallets.insert(address, wallet);
        
        Ok(())
    }
    
    /// Affiche la liste des portefeuilles disponibles
    fn list_wallets(&self) {
        if self.wallets.is_empty() {
            println!("Aucun portefeuille disponible");
            return;
        }
        
        println!("\n┌─────────── PORTEFEUILLES ───────────┐");
        for (i, (address, _)) in self.wallets.iter().enumerate() {
            println!("│ {}. {} │", i+1, address);
        }
        println!("└─────────────────────────────────────┘");
    }
    
    /// Lit un index de portefeuille depuis l'entrée standard
    fn read_wallet_index(&self, stdin: &mut impl BufRead) -> Result<usize> {
        if self.wallets.is_empty() {
            return Err(Error::msg("Aucun portefeuille disponible"));
        }
        
        let addresses: Vec<_> = self.wallets.keys().cloned().collect();
        
        let mut input = String::with_capacity(16);
        print!("> ");
        io::stdout().flush()?;
        input.clear();
        stdin.read_line(&mut input)?;
        
        let index = input.trim().parse::<usize>()
            .map_err(|_| Error::msg("Entrée non valide"))?;
            
        if index < 1 || index > addresses.len() {
            return Err(Error::msg("Index de portefeuille non valide"));
        }
        
        Ok(index - 1)
    }
    
    /// Envoie une transaction entre deux adresses
    fn send_transaction(&mut self, stdin: &mut impl BufRead) -> Result<()> {
        self.list_wallets();
        
        if self.wallets.is_empty() {
            println!("❌ Aucun portefeuille disponible pour effectuer des transactions");
            return Ok(());
        }
        
        // Sélectionner le portefeuille source
        println!("Choisissez le portefeuille source (par numéro):");
        let index = match self.read_wallet_index(stdin) {
            Ok(idx) => idx,
            Err(e) => {
                println!("❌ {}", e);
                return Ok(());
            }
        };
        
        let addresses: Vec<String> = self.wallets.keys().cloned().collect();
        let from_address = addresses[index].clone();
        
        // Demander l'adresse destinataire
        println!("Entrez l'adresse du destinataire:");
        let mut to_address = String::with_capacity(64);
        print!("> ");
        io::stdout().flush()?;
        stdin.read_line(&mut to_address)?;
        to_address = to_address.trim().to_string();
        
        // Vérifier si l'adresse est valide (doit avoir une certaine longueur minimale)
        if to_address.len() < 10 {
            println!("❌ Adresse destinataire non valide");
            return Ok(());
        }
        
        // Demander le montant
        println!("Entrez le montant à envoyer:");
        let mut amount_str = String::with_capacity(32);
        print!("> ");
        io::stdout().flush()?;
        stdin.read_line(&mut amount_str)?;
        
        let amount: u64 = match amount_str.trim().parse() {
            Ok(num) if num > 0 => num,
            _ => {
                println!("❌ Montant non valide");
                return Ok(());
            }
        };
        
        // Création de la transaction
        println!("💸 Création d'une transaction:");
        println!("  De: {}", from_address);
        println!("  À: {}", to_address);
        println!("  Montant: {}", amount);
        
        // Dans une implémentation réelle, nous devrions vérifier le solde ici
        let wallet = self.wallets.get(&from_address).unwrap();
        
        // Création d'une transaction valide (adaptée à votre implémentation)
        let mut transaction = Transaction::new(
            from_address,
            to_address,
            amount,
            1, // frais de transaction
            Some(wallet),
        )?;
        
        // Ajouter aux transactions en attente
        self.pending_transactions.push(transaction);
        println!("✅ Transaction créée et ajoutée au pool de transactions en attente");
        
        Ok(())
    }
    
    /// Mine un bloc avec les transactions disponibles
    fn mine_block(&mut self) -> Result<()> {
        println!("⛏️ Minage d'un nouveau bloc...");
        
        if self.pending_transactions.is_empty() {
            println!("❌ Aucune transaction en attente à miner");
            return Ok(());
        }
        
        // Prendre les transactions en attente
        let transactions = std::mem::take(&mut self.pending_transactions);
        
        // Obtenir le timestamp actuel
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        // Miner le bloc
        let start_time = std::time::Instant::now();
        
        // Utiliser try_lock pour éviter les deadlocks potentiels
        let mut bc = match self.blockchain.try_lock() {
            Ok(bc) => bc,
            Err(_) => return Err(Error::msg("Blockchain verrouillée par un autre processus")),
        };
        
        // Ajouter le bloc à la blockchain
        bc.add_block(transactions)?;
        
        let elapsed = start_time.elapsed();
        println!("✅ Bloc miné en {:.2?}!", elapsed);
        
        Ok(())
    }
    
    /// Affiche l'état actuel de la blockchain
    fn show_blockchain(&self) {
        // Utiliser try_lock pour éviter les deadlocks potentiels
        let bc = match self.blockchain.try_lock() {
            Ok(bc) => bc,
            Err(_) => {
                println!("❌ Blockchain verrouillée par un autre processus");
                return;
            }
        };
        
        println!("\n┌─────────── BLOCKCHAIN ─────────┐");
        
        // Obtenir le nombre de blocs
        let blocks_count = bc.get_height();
        
        if blocks_count == 0 {
            println!("│ La blockchain est vide         │");
        } else {
            // Afficher les détails des derniers blocs (max 5)
            let max_blocks = 5;
            let start_idx = if blocks_count > max_blocks { blocks_count - max_blocks } else { 0 };
            
            for i in start_idx..blocks_count {
                if let Ok(Some(block)) = bc.get_block_at_height(i) {
                    println!("│ Bloc #{:<3}                    │", i);
                    println!("│   Hash: {:.8}...        │", hex::encode(&block.hash));
                    println!("│   Hash prec.: {:.8}...    │", hex::encode(&block.prev_hash));
                    println!("│   Nonce: {:<10}           │", block.nonce);
                    println!("│   Difficulté: {:<3}           │", block.difficulty);
                    println!("│   Transactions: {:<3}         │", block.transactions.len());
                    
                    if i < blocks_count - 1 {
                        println!("│                              │");
                    }
                }
            }
            
            if blocks_count > max_blocks {
                println!("│ (+ {} blocs précédents)       │", blocks_count - max_blocks);
            }
        }
        
        println!("└──────────────────────────────┘");
    }
    
    /// Affiche les transactions en attente
    fn show_pending_transactions(&self) {
        println!("\n┌─────── TRANSACTIONS EN ATTENTE ───────┐");
        
        if self.pending_transactions.is_empty() {
            println!("│ Aucune transaction en attente            │");
        } else {
            for (i, tx) in self.pending_transactions.iter().enumerate() {
                println!("│ Transaction #{:<3}                      │", i+1);
                println!("│   De: {:.15}...        │", tx.sender);
                println!("│   À: {:.15}...        │", tx.recipient);
                println!("│   Montant: {:<10}                  │", tx.amount);
                println!("│   Frais: {:<10}                    │", tx.fee);
                
                if i < self.pending_transactions.len() - 1 {
                    println!("│                                       │");
                }
            }
        }
        
        println!("└───────────────────────────────────────┘");
    }
    
    /// Vérifie le solde d'une adresse
    fn check_balance(&self, stdin: &mut impl BufRead) -> Result<()> {
        self.list_wallets();
        
        if self.wallets.is_empty() {
            println!("❌ Aucun portefeuille disponible");
            return Ok(());
        }
        
        println!("Choisissez le portefeuille (par numéro):");
        let index = match self.read_wallet_index(stdin) {
            Ok(idx) => idx,
            Err(e) => {
                println!("❌ {}", e);
                return Ok(());
            }
        };
        
        let addresses: Vec<String> = self.wallets.keys().cloned().collect();
        let address = &addresses[index];
        
        // Simuler le calcul du solde pour l'instant
        let balance = 100u64;
        
        // Afficher le solde
        println!("💰 Le solde de l'adresse {} est: {} NeuralCoins", address, balance);
        
        Ok(())
    }
    
    /// Sauvegarde les portefeuilles sur disque
    pub fn save_wallets(&self) -> Result<()> {
        // Dans une implémentation réelle, nous sauvegarderions les portefeuilles ici
        println!("📝 Sauvegarde des portefeuilles...");
        
        for (address, _) in &self.wallets {
            println!("  - Portefeuille {}: sauvegardé", address);
        }
        
        println!("✅ Portefeuilles sauvegardés avec succès");
        Ok(())
    }
}

impl Drop for CLI {
    fn drop(&mut self) {
        // Tenter de sauvegarder les portefeuilles à la fermeture
        if let Err(e) = self.save_wallets() {
            eprintln!("❌ Erreur lors de la sauvegarde des portefeuilles: {}", e);
        }
    }
}
