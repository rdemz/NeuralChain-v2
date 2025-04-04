use std::fmt;
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

/// Structure d'une transaction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transaction {
    pub id: [u8; 32],               // Hash de la transaction (identifiant unique)
    pub sender: [u8; 32],           // Adresse de l'expéditeur
    pub nonce: u64,                 // Nonce pour éviter la réutilisation de transaction
    pub fee: u64,                   // Frais de transaction
    pub timestamp: u64,             // Horodatage Unix en secondes
    pub signature: Vec<u8>,         // Signature cryptographique
    pub signature_scheme: SignatureScheme, // Schéma de signature utilisé
    pub tx_type: TransactionType,   // Type de transaction
    pub payload: Vec<u8>,           // Données supplémentaires (métadonnées, etc.)
}

/// Type de transaction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TransactionType {
    /// Transaction basée sur le modèle UTXO
    UTXO {
        inputs: Vec<[u8; 32]>,      // Références aux sorties non dépensées
        outputs: Vec<UTXOOutput>,   // Nouvelles sorties créées
    },
    
    /// Transaction basée sur le modèle de compte
    Account {
        to: [u8; 32],              // Adresse du destinataire
        value: u64,                // Montant à transférer
    },
    
    /// Transaction avec un contrat intelligent
    Contract {
        to: [u8; 32],              // Adresse du contrat (zéro pour création)
        data: Vec<u8>,             // Données du contrat (code ou appel)
        value: u64,                // Montant à transférer au contrat
    },
}

/// Sortie UTXO
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UTXOOutput {
    pub recipient: [u8; 32],      // Adresse du destinataire
    pub amount: u64,              // Montant
    pub script: Option<Vec<u8>>,  // Script de verrouillage (optionnel)
}

/// Schémas de signature supportés
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SignatureScheme {
    Ed25519,
    Secp256k1,
    Schnorr,
}

impl Transaction {
    /// Crée une nouvelle transaction
    pub fn new(
        sender: [u8; 32],
        nonce: u64,
        fee: u64,
        tx_type: TransactionType,
        signature_scheme: SignatureScheme,
        payload: Vec<u8>,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Le temps est avant l'époque UNIX")
            .as_secs();
        
        let mut tx = Self {
            id: [0; 32],
            sender,
            nonce,
            fee,
            timestamp,
            signature: Vec::new(), // Sera rempli plus tard
            signature_scheme,
            tx_type,
            payload,
        };
        
        // Calculer l'ID (hash) de la transaction
        tx.id = tx.compute_hash();
        
        tx
    }
    
    /// Calcule le hash de la transaction
    pub fn compute_hash(&self) -> [u8; 32] {
        // Créer un message à hacher excluant la signature (qui n'est pas encore définie)
        let message = self.compute_message();
        
        // Calculer le hash SHA-256
        let mut hasher = Sha256::new();
        hasher.update(&message);
        let result = hasher.finalize();
        
        // Convertir le résultat en tableau de 32 octets
        let mut hash = [0; 32];
        hash.copy_from_slice(&result);
        hash
    }
    
    /// Calcule le message qui sera signé
    pub fn compute_message(&self) -> Vec<u8> {
        let mut message = Vec::new();
        
        // Ajouter les champs de la transaction
        message.extend_from_slice(&self.sender);
        message.extend_from_slice(&self.nonce.to_le_bytes());
        message.extend_from_slice(&self.fee.to_le_bytes());
        message.extend_from_slice(&self.timestamp.to_le_bytes());
        
        // Ajouter les champs spécifiques au type de transaction
        match &self.tx_type {
            TransactionType::UTXO { inputs, outputs } => {
                // Identifiant de type
                message.push(1);
                
                // Inputs
                for input in inputs {
                    message.extend_from_slice(input);
                }
                
                // Outputs
                for output in outputs {
                    message.extend_from_slice(&output.recipient);
                    message.extend_from_slice(&output.amount.to_le_bytes());
                    if let Some(script) = &output.script {
                        message.extend_from_slice(script);
                    }
                }
            },
            TransactionType::Account { to, value } => {
                // Identifiant de type
                message.push(2);
                
                message.extend_from_slice(to);
                message.extend_from_slice(&value.to_le_bytes());
            },
            TransactionType::Contract { to, data, value } => {
                // Identifiant de type
                message.push(3);
                
                message.extend_from_slice(to);
                message.extend_from_slice(data);
                message.extend_from_slice(&value.to_le_bytes());
            }
        }
        
        // Ajouter la payload
        message.extend_from_slice(&self.payload);
        
        message
    }
    
    /// Signe la transaction avec une clé privée
    pub fn sign(&mut self, private_key: &[u8]) {
        // Le code réel dépendrait du schéma de signature
        // Dans une implémentation réelle, on utiliserait des bibliothèques cryptographiques
        
        // Pour l'exemple, simule une signature
        let message = self.compute_message();
        
        match self.signature_scheme {
            SignatureScheme::Ed25519 => {
                // Simuler une signature Ed25519 (64 octets)
                self.signature = vec![0; 64];
            },
            SignatureScheme::Secp256k1 => {
                // Simuler une signature Secp256k1 (65 octets)
                self.signature = vec![0; 65];
            },
            SignatureScheme::Schnorr => {
                // Simuler une signature Schnorr (64 octets)
                self.signature = vec![0; 64];
            },
        }
    }
    
    /// Vérifie si la signature est valide
    pub fn verify_signature(&self) -> bool {
        // Dans une implémentation réelle, on vérifierait la signature
        // Pour l'exemple, retourne simplement true
        true
    }
    
    /// Retourne le code de type de transaction
    pub fn tx_type_code(&self) -> u8 {
        match &self.tx_type {
            TransactionType::UTXO { .. } => 1,
            TransactionType::Account { .. } => 2,
            TransactionType::Contract { .. } => 3,
        }
    }
    
    /// Calcule la taille approximative de la transaction en octets
    pub fn size(&self) -> usize {
        let mut size = 0;
        
        // Champs communs
        size += 32; // id
        size += 32; // sender
        size += 8;  // nonce
        size += 8;  // fee
        size += 8;  // timestamp
        size += self.signature.len(); // signature
        size += 1;  // signature_scheme
        size += self.payload.len(); // payload
        
        // Champs spécifiques au type
        match &self.tx_type {
            TransactionType::UTXO { inputs, outputs } => {
                size += 1; // type code
                size += inputs.len() * 32; // inputs (32 octets par input)
                
                // outputs
                for output in outputs {
                    size += 32; // recipient
                    size += 8;  // amount
                    size += output.script.as_ref().map_or(0, |s| s.len()); // script
                }
            },
            TransactionType::Account { to: _, value: _ } => {
                size += 1; // type code
                size += 32; // to
                size += 8;  // value
            },
            TransactionType::Contract { to: _, data, value: _ } => {
                size += 1; // type code
                size += 32; // to
                size += data.len(); // data
                size += 8;  // value
            }
        }
        
        size
    }
    
    /// Sérialise la transaction en binaire
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Échec de la sérialisation de la transaction")
    }
    
    /// Désérialise une transaction à partir de données binaires
    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }
}

impl fmt::Display for Transaction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tx(id={}, type={})", 
            hex::encode(&self.id[0..6]), // Affiche seulement les 6 premiers octets pour la lisibilité
            match &self.tx_type {
                TransactionType::UTXO { .. } => "UTXO",
                TransactionType::Account { .. } => "ACCOUNT",
                TransactionType::Contract { .. } => "CONTRACT",
            }
        )
    }
}
