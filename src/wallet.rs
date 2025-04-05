use crate::transaction::{Transaction, TransactionType, SignatureScheme};
use anyhow::{Result, Context};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use sha2::{Sha256, Digest};
use crate::utils::{bytes_to_hex, hex_to_bytes};
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::fs;
use std::io::Write;

/// Structure pour le stockage sécurisé d'une clé privée
#[derive(Serialize, Deserialize, Clone)]
pub struct WalletKeys {
    pub private_key: String,   // Clé privée en hexadécimal
    pub public_key: String,    // Clé publique en hexadécimal
    pub address: String,       // Adresse du portefeuille (hash de la clé publique)
}

/// Structure principale du portefeuille
pub struct Wallet {
    /// Clés du portefeuille
    keys: WalletKeys,
    /// Chemin du fichier de stockage des clés
    wallet_file: Option<PathBuf>,
    /// Nonce actuel pour les transactions
    current_nonce: u64,
}

impl Wallet {
    /// Crée un nouveau portefeuille avec de nouvelles clés
    pub fn new() -> Result<Self> {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        
        // Convertir les clés en format hexadécimal
        let private_key = bytes_to_hex(&signing_key.to_bytes());
        let public_key = bytes_to_hex(&verifying_key.to_bytes());
        
        // Calculer l'adresse (hash de la clé publique)
        let mut hasher = Sha256::new();
        hasher.update(verifying_key.to_bytes()); // Clippy fix: removed unnecessary borrow
        let address = bytes_to_hex(&hasher.finalize()[0..20]); // Prendre les 20 premiers octets
        
        let keys = WalletKeys {
            private_key,
            public_key,
            address,
        };
        
        Ok(Self {
            keys,
            wallet_file: None,
            current_nonce: 0,
        })
    }
    
    /// Charge un portefeuille depuis un fichier
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(&path)
            .context(format!("Échec de lecture du fichier de portefeuille {:?}", path.as_ref()))?;
            
        let keys: WalletKeys = serde_json::from_str(&content)
            .context("Échec de désérialisation des clés du portefeuille")?;
            
        Ok(Self {
            keys,
            wallet_file: Some(path.as_ref().to_path_buf()),
            current_nonce: 0,
        })
    }
    
    /// Sauvegarde le portefeuille dans un fichier
    pub fn save<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let serialized = serde_json::to_string_pretty(&self.keys)
            .context("Échec de sérialisation des clés du portefeuille")?;
            
        let mut file = fs::File::create(&path)
            .context(format!("Échec de création du fichier de portefeuille {:?}", path.as_ref()))?;
            
        file.write_all(serialized.as_bytes())
            .context("Échec d'écriture des données du portefeuille")?;
            
        self.wallet_file = Some(path.as_ref().to_path_buf());
        
        Ok(())
    }
    
    /// Obtient l'adresse du portefeuille
    pub fn get_address(&self) -> &str {
        &self.keys.address
    }
    
    /// Obtient la clé publique du portefeuille
    pub fn get_public_key(&self) -> &str {
        &self.keys.public_key
    }
    
    /// Crée une nouvelle transaction
    pub fn create_transaction(
        &mut self,
        recipient: Option<&str>,
        amount: u64,
        fee: u64,
        transaction_type: TransactionType,
        data: Vec<u8>,
    ) -> Result<Transaction> {
        // Convertir la clé publique de l'expéditeur en octets
        let sender_public_key = hex_to_bytes(&self.keys.public_key)?;
        
        // Convertir la clé publique du destinataire en octets (si fournie)
        let recipient_public_key = if let Some(recipient_hex) = recipient {
            Some(hex_to_bytes(recipient_hex)?)
        } else {
            None
        };
        
        // Créer la transaction non signée
        let mut tx = Transaction::new(
            transaction_type,
            sender_public_key,
            recipient_public_key,
            amount,
            fee,
            self.current_nonce,
            data,
        );
        
        // Signer la transaction
        self.sign_transaction(&mut tx)?;
        
        // Incrémenter le nonce
        self.current_nonce += 1;
        
        Ok(tx)
    }
    
    /// Signe une transaction
    fn sign_transaction(&self, tx: &mut Transaction) -> Result<()> {
        // Convertir la clé privée en octets
        let private_key_bytes = hex_to_bytes(&self.keys.private_key)?;
        
        // Reconstruire la clé de signature
        if private_key_bytes.len() != 32 {
            return Err(anyhow::anyhow!("Longueur de clé privée invalide"));
        }
        
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&private_key_bytes);
        
        // Simplifier le match inutile (Clippy fix)
        let signing_key = SigningKey::from_bytes(&key_bytes);
        
        // Définir le schéma de signature et signer
        tx.signature_scheme = SignatureScheme::Ed25519;
        tx.sign_ed25519(&signing_key)?;
        
        Ok(())
    }
    
    /// Met à jour le nonce du portefeuille
    pub fn update_nonce(&mut self, new_nonce: u64) {
        self.current_nonce = new_nonce;
    }
    
    /// Vérifie si une transaction a été signée par ce portefeuille
    pub fn verify_own_transaction(&self, tx: &Transaction) -> Result<bool> {
        // Convertir la clé publique du portefeuille en octets
        let wallet_public_key = hex_to_bytes(&self.keys.public_key)?;
        
        // Vérifier que la transaction a bien été envoyée par ce portefeuille
        if tx.sender != wallet_public_key {
            return Ok(false);
        }
        
        // Vérifier la signature de la transaction
        tx.verify_signature()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs::remove_file;
    
    #[test]
    fn test_wallet_creation() {
        let wallet = Wallet::new().unwrap();
        
        // Vérifier que les clés ont été générées
        assert!(!wallet.keys.private_key.is_empty());
        assert!(!wallet.keys.public_key.is_empty());
        assert!(!wallet.keys.address.is_empty());
    }
    
    #[test]
    fn test_wallet_save_and_load() {
        let test_path = env::temp_dir().join("test_wallet.json");
        
        // Créer et sauvegarder un portefeuille
        let mut wallet = Wallet::new().unwrap();
        wallet.save(&test_path).unwrap();
        
        // Charger le portefeuille
        let loaded_wallet = Wallet::load(&test_path).unwrap();
        
        // Vérifier que les clés sont identiques
        assert_eq!(wallet.keys.private_key, loaded_wallet.keys.private_key);
        assert_eq!(wallet.keys.public_key, loaded_wallet.keys.public_key);
        assert_eq!(wallet.keys.address, loaded_wallet.keys.address);
        
        // Nettoyer
        let _ = remove_file(&test_path);
    }
    
    #[test]
    fn test_transaction_creation() {
        let mut wallet = Wallet::new().unwrap();
        
        // Créer une transaction (utilisons un underscore pour éviter l'avertissement)
        let _tx = wallet.create_transaction(
            None,
            100,
            10,
            TransactionType::Transfer,
            vec![],
        ).unwrap();
        
        // Vérifier que le nonce a été incrémenté
        assert_eq!(wallet.current_nonce, 1);
        
        // La vérification de signature est testée séparément
    }
}
