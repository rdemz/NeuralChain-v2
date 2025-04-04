use std::collections::HashMap;
use ring::signature::{self, Ed25519KeyPair, Signature, KeyPair};
use ring::rand::SystemRandom;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use dashmap::DashMap;
use thiserror::Error;
use chrono::{DateTime, Utc, Duration};
use std::time::Duration as StdDuration;
use std::sync::Arc;
use tokio::sync::Mutex;
use async_trait::async_trait;
use uuid::Uuid;

// Structure simple de cache
pub struct Cache<K, V> {
    data: DashMap<K, (V, DateTime<Utc>)>,
    capacity: usize,
    ttl: StdDuration,
}

impl<K, V> Cache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    pub fn new(capacity: usize, ttl: StdDuration) -> Self {
        Self {
            data: DashMap::new(),
            capacity,
            ttl,
        }
    }
    
    pub fn get(&self, key: &K) -> Option<V> {
        if let Some(entry) = self.data.get(key) {
            let (value, timestamp) = entry.value();
            
            // Vérifier si l'entrée est expirée
            if Utc::now() - *timestamp > Duration::from_std(self.ttl).unwrap() {
                self.data.remove(key);
                None
            } else {
                Some(value.clone())
            }
        } else {
            None
        }
    }
    
    pub fn insert(&self, key: K, value: V) {
        // Nettoyer le cache si nécessaire
        if self.data.len() >= self.capacity {
            self.clean();
        }
        
        self.data.insert(key, (value, Utc::now()));
    }
    
    fn clean(&self) {
        // Supprimer les entrées les plus anciennes ou expirées
        let expired_cutoff = Utc::now() - Duration::from_std(self.ttl).unwrap();
        let mut to_remove = Vec::new();
        
        // Identifier les clés à supprimer
        for entry in self.data.iter() {
            let (key, (_, timestamp)) = (entry.key().clone(), entry.value().clone());
            
            if timestamp < expired_cutoff {
                to_remove.push(key);
            }
        }
        
        // Supprimer les clés expirées
        for key in to_remove {
            self.data.remove(&key);
        }
        
        // Si le cache est encore trop grand, supprimer 10% des entrées
        if self.data.len() > self.capacity {
            let overflow = self.data.len() - self.capacity;
            let remove_count = std::cmp::max(overflow, self.capacity / 10);
            
            let mut entries: Vec<_> = self.data.iter().collect();
            entries.sort_by_key(|e| e.value().1);
            
            for i in 0..remove_count {
                if i < entries.len() {
                    self.data.remove(entries[i].key());
                }
            }
        }
    }
}

// Format de base d'un identifiant décentralisé (DID)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DecentralizedId {
    // Format standard: did:neural:base58encodedvalue
    did_string: String,
    // Clé publique associée
    public_key: Vec<u8>,
    // Version du format DID
    version: u8,
}

// Document d'identité associé au DID
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DidDocument {
    id: DecentralizedId,
    controller: Option<DecentralizedId>,
    verification_methods: Vec<VerificationMethod>,
    authentication: Vec<Authentication>,
    assertion_method: Vec<String>,
    service_endpoints: Vec<ServiceEndpoint>,
    created: DateTime<Utc>,
    updated: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationMethod {
    id: String,
    type_: String,
    controller: String,
    public_key_multibase: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Authentication {
    Referenced(String), // Référence à une méthode de vérification
    Embedded(VerificationMethod), // Méthode de vérification embarquée
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    id: String,
    type_: String,
    service_endpoint: String,
}

// Résultat d'une résolution de DID
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DidResolutionResult {
    did_document: Option<DidDocument>,
    metadata: DidMetadata,
    status: DidResolutionStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DidMetadata {
    created: DateTime<Utc>,
    updated: Option<DateTime<Utc>>,
    deactivated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DidResolutionStatus {
    Success,
    NotFound,
    InvalidDid,
    Deactivated,
    ServerError,
}

// Erreurs liées aux opérations DID
#[derive(Error, Debug)]
pub enum DidError {
    #[error("Format DID invalide")]
    InvalidDidFormat,
    
    #[error("Valeur DID invalide")]
    InvalidDidValue,
    
    #[error("DID non trouvé")]
    DidNotFound,
    
    #[error("Méthode DID non supportée")]
    UnsupportedDidMethod,
    
    #[error("DID révoqué")]
    DidRevoked,
    
    #[error("Non-correspondance de DID")]
    DidMismatch,
    
    #[error("Signature invalide")]
    InvalidSignature,
    
    #[error("DID déjà révoqué")]
    DidAlreadyRevoked,
    
    #[error("Erreur de sérialisation")]
    SerializationError,
    
    #[error("Aucune méthode de vérification")]
    NoVerificationMethod,
    
    #[error("Format de clé non supporté")]
    UnsupportedKeyFormat,
    
    #[error("Clé publique invalide")]
    InvalidPublicKey,
    
    #[error("Erreur de résolution externe")]
    ExternalResolutionError(String),
}

// Interface pour les résolveurs cross-chain
#[async_trait]
pub trait CrossChainResolver: Send + Sync {
    async fn resolve(&self, did: &str) -> Result<DidResolutionResult, DidError>;
    async fn supports(&self, method: &str) -> bool;
}

// Gestionnaire d'identité décentralisée
pub struct DidManager {
    registry: DashMap<DecentralizedId, DidDocument>,
    revoked_dids: DashMap<DecentralizedId, DateTime<Utc>>,
    resolution_cache: Cache<String, DidResolutionResult>,
    cross_chain_resolvers: HashMap<String, Arc<dyn CrossChainResolver>>,
}

impl DidManager {
    pub fn new() -> Self {
        Self {
            registry: DashMap::new(),
            revoked_dids: DashMap::new(),
            resolution_cache: Cache::new(1000, StdDuration::from_secs(3600)),
            cross_chain_resolvers: HashMap::new(),
        }
    }
    
    // Création d'un nouveau DID
    pub fn create_did(&self, key_pair: &Ed25519KeyPair) -> Result<DecentralizedId, DidError> {
        let public_key = key_pair.public_key().as_ref().to_vec();
        let hash = Sha256::digest(&public_key);
        let encoded = bs58::encode(hash).into_string();
        let did_string = format!("did:neural:{}", encoded);
        
        let did = DecentralizedId {
            did_string,
            public_key,
            version: 1,
        };
        
        // Créer un document DID minimal
        let document = DidDocument {
            id: did.clone(),
            controller: None,
            verification_methods: vec![
                VerificationMethod {
                    id: format!("{}#keys-1", did.did_string),
                    type_: "Ed25519VerificationKey2018".to_string(),
                    controller: did.did_string.clone(),
                    public_key_multibase: format!("z{}", bs58::encode(&public_key).into_string()),
                }
            ],
            authentication: vec![
                Authentication::Referenced(format!("{}#keys-1", did.did_string))
            ],
            assertion_method: vec![format!("{}#keys-1", did.did_string)],
            service_endpoints: Vec::new(),
            created: Utc::now(),
            updated: None,
        };
        
        // Enregistrer le document
        self.registry.insert(did.clone(), document);
        
        Ok(did)
    }
    
    // Résolution d'un DID
    pub async fn resolve_did(&self, did: &str) -> Result<DidResolutionResult, DidError> {
        // Vérifier dans le cache d'abord
        if let Some(cached) = self.resolution_cache.get(&did.to_string()) {
            return Ok(cached);
        }
        
        // Vérifier le format
        if !did.starts_with("did:") {
            return Err(DidError::InvalidDidFormat);
        }
        
        let parts: Vec<&str> = did.split(':').collect();
        if parts.len() < 3 {
            return Err(DidError::InvalidDidFormat);
        }
        
        let method = parts[1];
        
        let result = match method {
            "neural" => {
                // C'est notre méthode native
                self.resolve_neural_did(did).await?
            },
            _ => {
                // Méthode externe, utiliser un résolveur cross-chain
                if let Some(resolver) = self.cross_chain_resolvers.get(method) {
                    resolver.resolve(did).await?
                } else {
                    return Err(DidError::UnsupportedDidMethod);
                }
            }
        };
        
        // Mettre en cache le résultat
        self.resolution_cache.insert(did.to_string(), result.clone());
        
        Ok(result)
    }
    
    // Résolution d'un DID de la méthode neural
    async fn resolve_neural_did(&self, did: &str) -> Result<DidResolutionResult, DidError> {
        let parts: Vec<&str> = did.split(':').collect();
        let id_value = parts[2];
        
        if id_value.len() < 16 {
            return Err(DidError::InvalidDidValue);
        }
        
        // Chercher dans notre registre
        for item in self.registry.iter() {
            if item.key().did_string == did {
                let is_revoked = self.revoked_dids.contains_key(item.key());
                
                return Ok(DidResolutionResult {
                    did_document: Some(item.value().clone()),
                    metadata: DidMetadata {
                        created: item.value().created,
                        updated: item.value().updated,
                        deactivated: is_revoked,
                    },
                    status: if is_revoked {
                        DidResolutionStatus::Deactivated
                    } else {
                        DidResolutionStatus::Success
                    },
                });
            }
        }
        
        Ok(DidResolutionResult {
            did_document: None,
            metadata: DidMetadata {
                created: Utc::now(),
                updated: None,
                deactivated: false,
            },
            status: DidResolutionStatus::NotFound,
        })
    }
    
    // Ajouter un résolveur cross-chain
    pub fn add_cross_chain_resolver(&mut self, method: &str, resolver: Arc<dyn CrossChainResolver>) {
        self.cross_chain_resolvers.insert(method.to_string(), resolver);
    }
    
    // Mise à jour d'un document DID
    pub fn update_did_document(&self, did: &DecentralizedId, new_document: DidDocument, 
                              signature: &[u8]) -> Result<(), DidError> {
        // Vérifier que le DID existe
        if !self.registry.contains_key(did) {
            return Err(DidError::DidNotFound);
        }
        
        // Vérifier que le DID n'a pas été révoqué
        if self.revoked_dids.contains_key(did) {
            return Err(DidError::DidRevoked);
        }
        
        // Vérifier que l'identifiant dans le document correspond
        if new_document.id != *did {
            return Err(DidError::DidMismatch);
        }
        
        // Vérifier la signature
        let document_bytes = serde_json::to_vec(&new_document)
            .map_err(|_| DidError::SerializationError)?;
        
        let signature_verified = self.verify_signature(did, &document_bytes, signature)?;
        
        if !signature_verified {
            return Err(DidError::InvalidSignature);
        }
        
        // Mettre à jour le document avec le timestamp actuel
        let mut updated_document = new_document;
        updated_document.updated = Some(Utc::now());
        
        // Remplacer le document existant
        self.registry.insert(did.clone(), updated_document);
        
        Ok(())
    }
    
    // Révoquer un DID
    pub fn revoke_did(&self, did: &DecentralizedId, signature: &[u8]) -> Result<(), DidError> {
        // Vérifier que le DID existe
        if !self.registry.contains_key(did) {
            return Err(DidError::DidNotFound);
        }
        
        // Vérifier que le DID n'est pas déjà révoqué
        if self.revoked_dids.contains_key(did) {
            return Err(DidError::DidAlreadyRevoked);
        }
        
        // Vérifier la signature
        let revocation_message = format!("revoke:{}", did.did_string);
        let signature_verified = self.verify_signature(
            did, 
            revocation_message.as_bytes(), 
            signature
        )?;
        
        if !signature_verified {
            return Err(DidError::InvalidSignature);
        }
        
        // Révoquer le DID
        self.revoked_dids.insert(did.clone(), Utc::now());
        
        Ok(())
    }
    
    // Vérifier une signature
    fn verify_signature(&self, did: &DecentralizedId, message: &[u8], signature: &[u8]) 
                       -> Result<bool, DidError> {
        let document = self.registry.get(did)
            .ok_or(DidError::DidNotFound)?;
            
        // Récupérer la méthode de vérification appropriée
        let verification_method = document.verification_methods.iter()
            .find(|vm| vm.controller == did.did_string)
            .ok_or(DidError::NoVerificationMethod)?;
        
        // Décoder la clé publique
        let key_str = &verification_method.public_key_multibase;
        if !key_str.starts_with('z') {
            return Err(DidError::UnsupportedKeyFormat);
        }
        
        let key_bytes = bs58::decode(&key_str[1..])
            .into_vec()
            .map_err(|_| DidError::InvalidPublicKey)?;
        
        // Vérifier la signature avec la clé publique
        let signature_algorithm = &signature::ED25519;
        let public_key = signature::UnparsedPublicKey::new(signature_algorithm, &key_bytes);
        
        match public_key.verify(message, signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    // Enregistrer un service pour un DID
    pub fn add_service(&self, did: &DecentralizedId, service: ServiceEndpoint, 
                      signature: &[u8]) -> Result<(), DidError> {
        // Récupérer le document existant
        let mut document = self.registry.get(did)
            .ok_or(DidError::DidNotFound)?
            .clone();
        
        // Vérifier que le service n'existe pas déjà
        if document.service_endpoints.iter().any(|s| s.id == service.id) {
            return Err(DidError::SerializationError); // Utiliser une erreur plus spécifique
        }
        
        // Créer le message à vérifier
        let message = format!("add_service:{}:{}", did.did_string, service.id);
        
        // Vérifier la signature
        let signature_verified = self.verify_signature(did, message.as_bytes(), signature)?;
        
        if !signature_verified {
            return Err(DidError::InvalidSignature);
        }
        
        // Ajouter le service
        document.service_endpoints.push(service);
        document.updated = Some(Utc::now());
        
        // Mettre à jour le document
        self.registry.insert(did.clone(), document);
        
        Ok(())
    }
    
    // Vérifier si un DID est valide et actif
    pub fn is_did_valid(&self, did: &str) -> bool {
        // Format valide
        if !did.starts_with("did:") {
            return false;
        }
        
        let parts: Vec<&str> = did.split(':').collect();
        if parts.len() < 3 {
            return false;
        }
        
        // Pour les DIDs de notre méthode
        if parts[1] == "neural" {
            let id_value = parts[2];
            if id_value.len() < 16 {
                return false;
            }
            
            // Chercher dans le registre
            for item in self.registry.iter() {
                if item.key().did_string == did {
                    // Vérifier que le DID n'est pas révoqué
                    return !self.revoked_dids.contains_key(item.key());
                }
            }
        }
        
        // Par défaut, on ne peut pas confirmer la validité
        false
    }
}

// Implémentation d'un résolveur cross-chain pour Ethereum
pub struct EthereumDIDResolver {
    rpc_url: String,
    registry_address: String,
}

#[async_trait]
impl CrossChainResolver for EthereumDIDResolver {
    async fn resolve(&self, did: &str) -> Result<DidResolutionResult, DidError> {
        // Extraire l'adresse Ethereum du DID
        let parts: Vec<&str> = did.split(':').collect();
        if parts.len() < 3 || parts[1] != "ethr" {
            return Err(DidError::UnsupportedDidMethod);
        }
        
        let eth_address = parts[2];
        
        // Vérifier le format de l'adresse Ethereum
        if !eth_address.starts_with("0x") || eth_address.len() != 42 {
            return Err(DidError::InvalidDidValue);
        }
        
        // Ici, on ferait un appel RPC à Ethereum pour récupérer les données du registre
        // Pour l'instant, on renvoie un document factice
        
        let now = Utc::now();
        
        Ok(DidResolutionResult {
            did_document: Some(DidDocument {
                id: DecentralizedId {
                    did_string: did.to_string(),
                    public_key: hex::decode(&eth_address[2..]).unwrap_or_default(),
                    version: 1,
                },
                controller: None,
                verification_methods: vec![
                    VerificationMethod {
                        id: format!("{}#keys-1", did),
                        type_: "EcdsaSecp256k1RecoveryMethod2020".to_string(),
                        controller: did.to_string(),
                        public_key_multibase: format!("z{}", eth_address),
                    }
                ],
                authentication: vec![
                    Authentication::Referenced(format!("{}#keys-1", did))
                ],
                assertion_method: vec![format!("{}#keys-1", did)],
                service_endpoints: Vec::new(),
                created: now,
                updated: None,
            }),
            metadata: DidMetadata {
                created: now,
                updated: None,
                deactivated: false,
            },
            status: DidResolutionStatus::Success,
        })
    }
    
    async fn supports(&self, method: &str) -> bool {
        method == "ethr"
    }
}
