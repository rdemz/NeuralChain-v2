use anyhow::{Result, Context, bail};
use ed25519_dalek::{Signature, SigningKey, VerifyingKey, Signer, Verifier};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Différents types de transactions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TransactionType {
    /// Transfert standard de fonds
    Transfer,
    /// Enregistrement de données sur la blockchain
    DataStorage,
    /// Contrats intelligents
    SmartContract,
    /// Création d'actif
    AssetCreation,
}

/// Types de schémas de signature supportés
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SignatureScheme {
    /// Ed25519
    Ed25519,
    /// ECDSA
    ECDSA,
    /// Aucune signature (utilisation interne uniquement)
    None,
}

/// Structure principale de transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Type de transaction
    pub transaction_type: TransactionType,
    /// Clé publique de l'expéditeur
    pub sender: Vec<u8>,
    /// Clé publique du destinataire (optionnel selon le type de transaction)
    pub recipient: Option<Vec<u8>>,
    /// Montant transféré
    pub amount: u64,
    /// Frais de transaction
    pub fee: u64,
    /// Nonce (numéro de séquence de la transaction)
    pub nonce: u64,
    /// Horodatage
    pub timestamp: u64,
    /// Données supplémentaires spécifiques au type de transaction
    pub data: Vec<u8>,
    /// Schéma de signature utilisé
    pub signature_scheme: SignatureScheme,
    /// Signature de la transaction
    pub signature: Option<Vec<u8>>,
}

impl Transaction {
    /// Crée une nouvelle transaction non signée
    pub fn new(
        transaction_type: TransactionType,
        sender: Vec<u8>,
        recipient: Option<Vec<u8>>,
        amount: u64,
        fee: u64,
        nonce: u64,
        data: Vec<u8>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            transaction_type,
            sender,
            recipient,
            amount,
            fee,
            nonce,
            timestamp,
            data,
            signature_scheme: SignatureScheme::None,
            signature: None,
        }
    }

    /// Signe la transaction avec une clé Ed25519
    pub fn sign_ed25519(&mut self, signing_key: &SigningKey) -> Result<()> {
        // Générer la représentation des données à signer
        let message = self.to_bytes_for_signing()?;
        
        // Signer les données
        let signature = signing_key.sign(&message);
        
        // Enregistrer la signature
        self.signature = Some(signature.to_bytes().to_vec());
        self.signature_scheme = SignatureScheme::Ed25519;
        
        Ok(())
    }

    /// Convertit la transaction en octets pour signature
    pub fn to_bytes_for_signing(&self) -> Result<Vec<u8>> {
        // Créer une copie temporaire sans signature
        let unsigned = Transaction {
            transaction_type: self.transaction_type,
            sender: self.sender.clone(),
            recipient: self.recipient.clone(),
            amount: self.amount,
            fee: self.fee,
            nonce: self.nonce,
            timestamp: self.timestamp,
            data: self.data.clone(),
            signature_scheme: self.signature_scheme,
            signature: None, // La signature est toujours None pour la vérification
        };
        
        // Sérialiser la transaction
        bincode::serialize(&unsigned)
            .context("Échec de sérialisation de la transaction pour signature")
    }

    /// Vérifie la signature de la transaction
    pub fn verify_signature(&self) -> Result<bool> {
        match self.signature_scheme {
            SignatureScheme::Ed25519 => self.verify_ed25519(),
            SignatureScheme::ECDSA => {
                // Non implémenté pour l'instant
                bail!("La vérification de signature ECDSA n'est pas encore implémentée")
            }
            SignatureScheme::None => Ok(false),
        }
    }

    /// Vérifie une signature Ed25519
    fn verify_ed25519(&self) -> Result<bool> {
        // Vérifier que la signature existe
        let signature_bytes = match &self.signature {
            Some(sig) => sig,
            None => return Ok(false),
        };
        
        // Vérifier la longueur de la signature
        if signature_bytes.len() != 64 {
            return Ok(false);
        }
        
        // Convertir la signature en tableau fixe
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(signature_bytes);
        let signature = Signature::from_bytes(&sig_bytes);
        
        // Vérifier la longueur de la clé publique
        if self.sender.len() != 32 {
            return Ok(false);
        }
        
        // Convertir la clé publique en tableau fixe
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&self.sender);
        
        // Reconstruire la clé publique
        let verifying_key = match VerifyingKey::from_bytes(&key_bytes) {
            Ok(key) => key,
            Err(_) => return Ok(false),
        };
        
        // Obtenir les données signées
        let message = self.to_bytes_for_signing()?;
        
        // Vérifier la signature
        match verifying_key.verify(&message, &signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Calculer l'ID de la transaction (hash)
    pub fn get_id(&self) -> Result<Vec<u8>> {
        use sha2::{Sha256, Digest};
        
        let bytes = bincode::serialize(self)
            .context("Échec de sérialisation de la transaction pour l'ID")?;
            
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        
        Ok(hasher.finalize().to_vec())
    }
}
