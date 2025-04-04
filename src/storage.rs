use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use serde::{Serialize, Deserialize};
use rocksdb::{DB, Options, WriteBatch, ColumnFamilyDescriptor, SliceTransform};
use thiserror::Error;

use crate::block::{Block, BlockId};
use crate::blockchain::Blockchain;
use crate::transaction::Transaction;

/// Erreurs de stockage
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Erreur de base de données: {0}")]
    DBError(#[from] rocksdb::Error),
    
    #[error("Erreur de sérialisation: {0}")]
    SerializationError(#[from] bincode::Error),
    
    #[error("Bloc non trouvé: {0:?}")]
    BlockNotFound(BlockId),
    
    #[error("Transaction non trouvée: {0:?}")]
    TransactionNotFound([u8; 32]),
    
    #[error("Path invalide: {0}")]
    InvalidPath(String),
    
    #[error("Erreur d'IO: {0}")]
    IOError(#[from] std::io::Error),
    
    #[error("Fichier corrompu: {0}")]
    CorruptedFile(String),
    
    #[error("État incohérent: {0}")]
    InconsistentState(String),
}

/// Nom des familles de colonnes
const CF_BLOCKS: &str = "blocks";
const CF_BLOCK_HEIGHT_INDEX: &str = "block_height_index";
const CF_TRANSACTIONS: &str = "transactions";
const CF_TX_BLOCK_INDEX: &str = "tx_block_index";
const CF_BLOCKCHAIN_STATE: &str = "blockchain_state";
const CF_UTXOS: &str = "utxos";
const CF_ACCOUNT_BALANCES: &str = "account_balances";
const CF_ACCOUNT_NONCES: &str = "account_nonces";
const CF_METADATA: &str = "metadata";

/// Méta-données du stockage
#[derive(Serialize, Deserialize)]
struct StorageMetadata {
    version: u32,
    blocks_count: u64,
    transactions_count: u64,
    last_block_hash: [u8; 32],
    last_block_height: u64,
    genesis_hash: [u8; 32],
    creation_date: i64,
    last_updated: i64,
}

/// Système de stockage optimisé
pub struct OptimizedStorage {
    db: DB,
    flush_pending: AtomicBool,
    path: String,
    // Cache en mémoire pour les opérations fréquentes
    metadata_cache: parking_lot::RwLock<Option<StorageMetadata>>,
}

impl OptimizedStorage {
    /// Crée ou ouvre une nouvelle instance de stockage
    pub fn new(path: &str) -> Result<Self, StorageError> {
        let path = Path::new(path);
        
        // Créer le répertoire s'il n'existe pas
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        
        // Configurer les options RocksDB
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_max_open_files(256);
        opts.set_keep_log_file_num(10);
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
        opts.set_max_write_buffer_number(4);
        opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB
        opts.set_level_zero_file_num_compaction_trigger(4);
        opts.increase_parallelism(num_cpus::get() as i32);
        
        // Définir les préfixes pour une recherche efficace
        let mut block_height_opts = Options::default();
        let prefix_extractor = SliceTransform::create_fixed_prefix(8); // 8 octets pour u64
        block_height_opts.set_prefix_extractor(prefix_extractor);
        
        // Définir les familles de colonnes
        let cfs = vec![
            ColumnFamilyDescriptor::new(CF_BLOCKS, Options::default()),
            ColumnFamilyDescriptor::new(CF_BLOCK_HEIGHT_INDEX, block_height_opts),
            ColumnFamilyDescriptor::new(CF_TRANSACTIONS, Options::default()),
            ColumnFamilyDescriptor::new(CF_TX_BLOCK_INDEX, Options::default()),
            ColumnFamilyDescriptor::new(CF_BLOCKCHAIN_STATE, Options::default()),
            ColumnFamilyDescriptor::new(CF_UTXOS, Options::default()),
            ColumnFamilyDescriptor::new(CF_ACCOUNT_BALANCES, Options::default()),
            ColumnFamilyDescriptor::new(CF_ACCOUNT_NONCES, Options::default()),
            ColumnFamilyDescriptor::new(CF_METADATA, Options::default()),
        ];
        
        // Ouvrir la base de données
        let db = DB::open_cf_descriptors(&opts, path, cfs)?;
        
        // Initialiser les métadonnées si elles n'existent pas
        let metadata_cache = parking_lot::RwLock::new(None);
        
        let storage = Self {
            db,
            flush_pending: AtomicBool::new(false),
            path: path.to_string_lossy().to_string(),
            metadata_cache,
        };
        
        // Charger les métadonnées
        storage.load_metadata()?;
        
        Ok(storage)
    }
    
    /// Charge les métadonnées du stockage
    fn load_metadata(&self) -> Result<(), StorageError> {
        let cf_metadata = self.db.cf_handle(CF_METADATA).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes METADATA non trouvée".to_string())
        })?;
        
        match self.db.get_cf(&cf_metadata, b"metadata")? {
            Some(data) => {
                let metadata: StorageMetadata = bincode::deserialize(&data)?;
                let mut cache = self.metadata_cache.write();
                *cache = Some(metadata);
                Ok(())
            },
            None => {
                // Initialiser des métadonnées par défaut
                let now = chrono::Utc::now().timestamp();
                let metadata = StorageMetadata {
                    version: 1,
                    blocks_count: 0,
                    transactions_count: 0,
                    last_block_hash: [0; 32],
                    last_block_height: 0,
                    genesis_hash: [0; 32],
                    creation_date: now,
                    last_updated: now,
                };
                
                // Sauvegarder les métadonnées
                let serialized = bincode::serialize(&metadata)?;
                self.db.put_cf(&cf_metadata, b"metadata", serialized)?;
                
                let mut cache = self.metadata_cache.write();
                *cache = Some(metadata);
                
                Ok(())
            }
        }
    }
    
    /// Sauvegarde les métadonnées du stockage
    fn save_metadata(&self, metadata: &StorageMetadata) -> Result<(), StorageError> {
        let cf_metadata = self.db.cf_handle(CF_METADATA).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes METADATA non trouvée".to_string())
        })?;
        
        let serialized = bincode::serialize(metadata)?;
        self.db.put_cf(&cf_metadata, b"metadata", serialized)?;
        
        // Mettre à jour le cache
        let mut cache = self.metadata_cache.write();
        *cache = Some(metadata.clone());
        
        Ok(())
    }
    
    /// Sauvegarde un bloc dans le stockage
    pub fn save_block(&self, block: &Block) -> Result<(), StorageError> {
        let cf_blocks = self.db.cf_handle(CF_BLOCKS).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes BLOCKS non trouvée".to_string())
        })?;
        
        let cf_height = self.db.cf_handle(CF_BLOCK_HEIGHT_INDEX).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes BLOCK_HEIGHT_INDEX non trouvée".to_string())
        })?;
        
        let cf_tx = self.db.cf_handle(CF_TRANSACTIONS).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes TRANSACTIONS non trouvée".to_string())
        })?;
        
        let cf_tx_block = self.db.cf_handle(CF_TX_BLOCK_INDEX).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes TX_BLOCK_INDEX non trouvée".to_string())
        })?;
        
        // Préparer un lot d'écritures
        let mut batch = WriteBatch::default();
        
        // Sérialiser et stocker le bloc
        let block_data = bincode::serialize(block)?;
        batch.put_cf(&cf_blocks, &block.hash, &block_data);
        
        // Créer un index par hauteur
        batch.put_cf(&cf_height, &block.height.to_be_bytes(), &block.hash);
        
        // Stocker les transactions et indexer
        for tx in &block.transactions {
            let tx_data = bincode::serialize(tx)?;
            batch.put_cf(&cf_tx, &tx.id, &tx_data);
            
            // Index des transactions vers le bloc
            batch.put_cf(&cf_tx_block, &tx.id, &block.hash);
        }
        
        // Écrire le lot
        self.db.write(batch)?;
        
        // Mettre à jour les métadonnées si nécessaire
        let mut update_metadata = false;
        
        {
            let mut cache = self.metadata_cache.write();
            if let Some(ref mut metadata) = *cache {
                metadata.blocks_count += 1;
                metadata.transactions_count += block.transactions.len() as u64;
                
                if block.height > metadata.last_block_height {
                    metadata.last_block_height = block.height;
                    metadata.last_block_hash = block.hash;
                    metadata.last_updated = chrono::Utc::now().timestamp();
                }
                
                if block.height == 0 {
                    metadata.genesis_hash = block.hash;
                }
                
                update_metadata = true;
            }
        }
        
        // Sauvegarder les métadonnées mises à jour
        if update_metadata {
            let metadata = self.metadata_cache.read().clone().unwrap();
            self.save_metadata(&metadata)?;
        }
        
        // Marquer une vidange en attente
        self.flush_pending.store(true, Ordering::SeqCst);
        
        Ok(())
    }
    
    /// Récupère un bloc par son hash
    pub fn get_block(&self, hash: &[u8; 32]) -> Result<Block, StorageError> {
        let cf_blocks = self.db.cf_handle(CF_BLOCKS).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes BLOCKS non trouvée".to_string())
        })?;
        
        match self.db.get_cf(&cf_blocks, hash)? {
            Some(data) => {
                let block: Block = bincode::deserialize(&data)?;
                Ok(block)
            },
            None => Err(StorageError::BlockNotFound(*hash)),
        }
    }
    
    /// Récupère un bloc par sa hauteur
    pub fn get_block_by_height(&self, height: u64) -> Result<Block, StorageError> {
        let cf_height = self.db.cf_handle(CF_BLOCK_HEIGHT_INDEX).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes BLOCK_HEIGHT_INDEX non trouvée".to_string())
        })?;
        
        // Récupérer le hash du bloc à cette hauteur
        match self.db.get_cf(&cf_height, &height.to_be_bytes())? {
            Some(hash) => {
                if hash.len() == 32 {
                    let mut block_hash = [0u8; 32];
                    block_hash.copy_from_slice(&hash);
                    self.get_block(&block_hash)
                } else {
                    Err(StorageError::CorruptedFile(format!(
                        "Index de bloc corrompu pour la hauteur {}", height
                    )))
                }
            },
            None => Err(StorageError::BlockNotFound([0; 32])),
        }
    }
    
    /// Récupère une transaction par son ID
    pub fn get_transaction(&self, tx_id: &[u8; 32]) -> Result<Transaction, StorageError> {
        let cf_tx = self.db.cf_handle(CF_TRANSACTIONS).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes TRANSACTIONS non trouvée".to_string())
        })?;
        
        match self.db.get_cf(&cf_tx, tx_id)? {
            Some(data) => {
                let transaction: Transaction = bincode::deserialize(&data)?;
                Ok(transaction)
            },
            None => Err(StorageError::TransactionNotFound(*tx_id)),
        }
    }
    
    /// Trouve le bloc contenant une transaction
    pub fn find_block_for_transaction(&self, tx_id: &[u8; 32]) -> Result<BlockId, StorageError> {
        let cf_tx_block = self.db.cf_handle(CF_TX_BLOCK_INDEX).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes TX_BLOCK_INDEX non trouvée".to_string())
        })?;
        
        match self.db.get_cf(&cf_tx_block, tx_id)? {
            Some(hash) => {
                if hash.len() == 32 {
                    let mut block_hash = [0u8; 32];
                    block_hash.copy_from_slice(&hash);
                    Ok(block_hash)
                } else {
                    Err(StorageError::CorruptedFile(format!(
                        "Index de transaction corrompu pour l'ID {}", hex::encode(tx_id)
                    )))
                }
            },
            None => Err(StorageError::TransactionNotFound(*tx_id)),
        }
    }
    
    /// Récupère la plage des N derniers blocs
    pub fn get_last_n_blocks(&self, n: u64) -> Result<Vec<Block>, StorageError> {
        let mut blocks = Vec::new();
        
        // Récupérer la hauteur du dernier bloc
        let last_height = {
            let metadata = self.metadata_cache.read();
            metadata.as_ref().map_or(0, |m| m.last_block_height)
        };
        
        // Aucun bloc si la hauteur est 0
        if last_height == 0 {
            return Ok(blocks);
        }
        
        // Calculer la plage de hauteurs
        let start_height = if n >= last_height { 0 } else { last_height - n };
        
        // Récupérer les blocs dans la plage
        for height in start_height..=last_height {
            match self.get_block_by_height(height) {
                Ok(block) => blocks.push(block),
                Err(e) => {
                    if let StorageError::BlockNotFound(_) = e {
                        // Continuer si le bloc n'est pas trouvé
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        
        Ok(blocks)
    }
    
    /// Sauvegarde l'état complet de la blockchain
    pub fn save_blockchain(&self, blockchain: &Blockchain) -> Result<(), StorageError> {
        let cf_state = self.db.cf_handle(CF_BLOCKCHAIN_STATE).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes BLOCKCHAIN_STATE non trouvée".to_string())
        })?;
        
        // Sérialiser l'état
        let state_data = bincode::serialize(blockchain)?;
        
        // Écrire l'état
        self.db.put_cf(&cf_state, b"current_state", state_data)?;
        
        // Mettre à jour les métadonnées
        let mut cache = self.metadata_cache.write();
        if let Some(ref mut metadata) = *cache {
            metadata.last_block_height = blockchain.height();
            metadata.last_block_hash = blockchain.last_block_hash();
            metadata.blocks_count = blockchain.height() + 1; // +1 pour le bloc de genèse
            metadata.last_updated = chrono::Utc::now().timestamp();
        }
        
        // Forcer une vidange sur disque
        if self.flush_pending.load(Ordering::SeqCst) {
            self.db.flush()?;
            self.flush_pending.store(false, Ordering::SeqCst);
        }
        
        Ok(())
    }
    
    /// Charge l'état complet de la blockchain
    pub fn load_blockchain(&self) -> Result<Blockchain, StorageError> {
        let cf_state = self.db.cf_handle(CF_BLOCKCHAIN_STATE).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes BLOCKCHAIN_STATE non trouvée".to_string())
        })?;
        
        match self.db.get_cf(&cf_state, b"current_state")? {
            Some(data) => {
                let blockchain: Blockchain = bincode::deserialize(&data)?;
                Ok(blockchain)
            },
            None => {
                // Si aucun état n'est trouvé, retourner une erreur
                Err(StorageError::InconsistentState(
                    "Aucun état de blockchain trouvé".to_string()
                ))
            }
        }
    }
    
    /// Met à jour le solde d'un compte
    pub fn update_account_balance(&self, account: &[u8; 32], balance: u64) -> Result<(), StorageError> {
        let cf_balances = self.db.cf_handle(CF_ACCOUNT_BALANCES).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes ACCOUNT_BALANCES non trouvée".to_string())
        })?;
        
        // Écrire le nouveau solde
        self.db.put_cf(&cf_balances, account, &balance.to_be_bytes())?;
        
        Ok(())
    }
    
    /// Récupère le solde d'un compte
    pub fn get_account_balance(&self, account: &[u8; 32]) -> Result<u64, StorageError> {
        let cf_balances = self.db.cf_handle(CF_ACCOUNT_BALANCES).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes ACCOUNT_BALANCES non trouvée".to_string())
        })?;
        
        match self.db.get_cf(&cf_balances, account)? {
            Some(data) => {
                if data.len() == 8 {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&data);
                    Ok(u64::from_be_bytes(bytes))
                } else {
                    Err(StorageError::CorruptedFile(
                        "Données de solde corrompues".to_string()
                    ))
                }
            },
            None => Ok(0), // Compte inexistant = solde zéro
        }
    }
    
    /// Met à jour le nonce d'un compte
    pub fn update_account_nonce(&self, account: &[u8; 32], nonce: u64) -> Result<(), StorageError> {
        let cf_nonces = self.db.cf_handle(CF_ACCOUNT_NONCES).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes ACCOUNT_NONCES non trouvée".to_string())
        })?;
        
        // Écrire le nouveau nonce
        self.db.put_cf(&cf_nonces, account, &nonce.to_be_bytes())?;
        
        Ok(())
    }
    
    /// Récupère le nonce d'un compte
    pub fn get_account_nonce(&self, account: &[u8; 32]) -> Result<u64, StorageError> {
        let cf_nonces = self.db.cf_handle(CF_ACCOUNT_NONCES).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes ACCOUNT_NONCES non trouvée".to_string())
        })?;
        
        match self.db.get_cf(&cf_nonces, account)? {
            Some(data) => {
                if data.len() == 8 {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&data);
                    Ok(u64::from_be_bytes(bytes))
                } else {
                    Err(StorageError::CorruptedFile(
                        "Données de nonce corrompues".to_string()
                    ))
                }
            },
            None => Ok(0), // Compte inexistant = nonce zéro
        }
    }
    
    /// Ajoute ou met à jour un UTXO
    pub fn update_utxo(&self, utxo_id: &[u8; 32], amount: u64, spent: bool) -> Result<(), StorageError> {
        let cf_utxos = self.db.cf_handle(CF_UTXOS).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes UTXOS non trouvée".to_string())
        })?;
        
        // Préparer les données (montant + état dépensé)
        let mut data = Vec::with_capacity(9);
        data.extend_from_slice(&amount.to_be_bytes());
        data.push(if spent { 1 } else { 0 });
        
        // Écrire l'UTXO
        self.db.put_cf(&cf_utxos, utxo_id, data)?;
        
        Ok(())
    }
    
    /// Récupère un UTXO
    pub fn get_utxo(&self, utxo_id: &[u8; 32]) -> Result<Option<(u64, bool)>, StorageError> {
        let cf_utxos = self.db.cf_handle(CF_UTXOS).ok_or_else(|| {
            StorageError::InconsistentState("Famille de colonnes UTXOS non trouvée".to_string())
        })?;
        
        match self.db.get_cf(&cf_utxos, utxo_id)? {
            Some(data) => {
                if data.len() == 9 {
                    let mut amount_bytes = [0u8; 8];
                    amount_bytes.copy_from_slice(&data[0..8]);
                    let amount = u64::from_be_bytes(amount_bytes);
                    let spent = data[8] == 1;
                    
                    Ok(Some((amount, spent)))
                } else {
                    Err(StorageError::CorruptedFile(
                        "Données UTXO corrompues".to_string()
                    ))
                }
            },
            None => Ok(None),
        }
    }
    
    /// Force une vidange du stockage sur disque
    pub fn flush(&self) -> Result<(), StorageError> {
        self.db.flush()?;
        self.flush_pending.store(false, Ordering::SeqCst);
        Ok(())
    }
    
    /// Récupère les métadonnées du stockage
    pub fn get_metadata(&self) -> Result<StorageMetadata, StorageError> {
        let cache = self.metadata_cache.read();
        match &*cache {
            Some(metadata) => Ok(metadata.clone()),
            None => Err(StorageError::InconsistentState(
                "Métadonnées non chargées".to_string()
            )),
        }
    }
    
    /// Compacte la base de données pour optimiser l'espace
    pub fn compact(&self) -> Result<(), StorageError> {
        // Compacter les familles de colonnes principales
        for cf_name in &[CF_BLOCKS, CF_TRANSACTIONS, CF_UTXOS, CF_ACCOUNT_BALANCES] {
            let cf = self.db.cf_handle(cf_name).ok_or_else(|| {
                StorageError::InconsistentState(format!(
                    "Famille de colonnes {} non trouvée", cf_name
                ))
            })?;
            
            self.db.compact_range_cf(&cf, None, None);
        }
        
        Ok(())
    }
    
    /// Vérifie l'intégrité de la base de données
    pub fn check_integrity(&self) -> Result<bool, StorageError> {
        let last_height = {
            let metadata = self.metadata_cache.read();
            metadata.as_ref().map_or(0, |m| m.last_block_height)
        };
        
        // Si aucun bloc n'existe, considérer que l'intégrité est ok
        if last_height == 0 {
            return Ok(true);
        }
        
        // Vérifier le chaînage des blocs
        let mut cur_height = last_height;
        let mut cur_hash = match self.get_block_by_height(cur_height) {
            Ok(block) => block.header.parent_hash,
            Err(_) => return Ok(false),
        };
        
        while cur_height > 0 {
            cur_height -= 1;
            
            match self.get_block_by_height(cur_height) {
                Ok(block) => {
                    if block.hash != cur_hash {
                        return Ok(false);
                    }
                    
                    cur_hash = block.header.parent_hash;
                },
                Err(_) => return Ok(false),
            }
        }
        
        Ok(true)
    }
}
