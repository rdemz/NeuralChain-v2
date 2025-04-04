mod adaptive_reward;
mod block;
mod blockchain;
mod config;
mod consensus;
mod continuous_mining;
mod defi_layers;
mod decentralized_identity;
mod governance;
mod mempool;
mod monitoring;
mod neural_pow;
mod network_protocol;
mod oracle_system;
mod p2p_network;
mod peer_manager;
mod reputation_system;
mod storage;
mod transaction;
mod transaction_validation;
mod utils;
mod wallet;
mod zk_rollups;

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::Result;
use mimalloc::MiMalloc;
use tokio::sync::Mutex;
use tracing::{info, warn, error};

use crate::blockchain::Blockchain;
use crate::config::Config;
use crate::consensus::ConsensusEngine;
use crate::continuous_mining::{BlockQueue, MiningConfig, launch_continuous_mining};
use crate::mempool::OptimizedMempool;
use crate::monitoring::init_monitoring;
use crate::neural_pow::NeuralPoW;
use crate::p2p_network::start_p2p_server;
use crate::reputation_system::ReputationSystem;
use crate::storage::OptimizedStorage;
use crate::transaction_validation::UTXOCache;

// Utiliser mimalloc comme allocateur global
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> Result<()> {
    // Charger la configuration
    let config = Config::load_or_default()?;
    
    // Initialiser le monitoring
    let _prometheus_handle = init_monitoring(config.prometheus_port);
    info!("NeuralChain démarrage, version {}", env!("CARGO_PKG_VERSION"));
    
    // Initialiser le stockage optimisé
    let storage = match OptimizedStorage::new(&config.data_dir) {
        Ok(storage) => {
            info!("Stockage initialisé avec succès dans {}", config.data_dir);
            Arc::new(storage)
        },
        Err(e) => {
            error!("Erreur d'initialisation du stockage: {}", e);
            return Err(anyhow::anyhow!("Échec d'initialisation du stockage"));
        }
    };
    
    // Créer ou charger la blockchain
    let blockchain = match Blockchain::load_from_storage(Arc::clone(&storage)) {
        Ok(chain) => {
            info!("Blockchain chargée avec {} blocs", chain.height());
            Arc::new(Mutex::new(chain))
        },
        Err(e) => {
            warn!("Impossible de charger la blockchain: {}. Création d'une nouvelle chaîne", e);
            let new_chain = Blockchain::new(config.network_id);
            Arc::new(Mutex::new(new_chain))
        }
    };
    
    // Initialiser le système de réputation
    let reputation_system = Arc::new(ReputationSystem::new());
    
    // Initialiser le mempool optimisé
    let mempool = Arc::new(OptimizedMempool::new(
        config.mempool_max_size, 
        config.mempool_max_age_secs
    ));
    
    // Initialiser le cache UTXO
    let utxo_cache = Arc::new(UTXOCache::new());
    
    // Initialiser le système PoW neuronal
    let neural_pow = Arc::new(Mutex::new(NeuralPoW::new(config.consensus_difficulty)));
    
    // Créer le moteur de consensus
    let consensus_engine = Arc::new(ConsensusEngine::new(
        Arc::clone(&blockchain),
        Arc::clone(&neural_pow),
        Arc::clone(&reputation_system),
        config.clone(),
    ));
    
    // Initialiser le nettoyage périodique du mempool
    {
        let mempool_cleaner = Arc::clone(&mempool);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                let removed = mempool_cleaner.clean_expired();
                if removed > 0 {
                    info!("{} transactions expirées supprimées du mempool", removed);
                }
            }
        });
    }
    
    // Initialiser la file de blocs pour le mining
    let block_queue = Arc::new(BlockQueue::new(config.mining_queue_capacity));
    
    // Lancer le serveur P2P
    {
        let p2p_config = config.p2p_config.clone();
        let blockchain_for_p2p = Arc::clone(&blockchain);
        let mempool_for_p2p = Arc::clone(&mempool);
        let reputation_for_p2p = Arc::clone(&reputation_system);
        let consensus_for_p2p = Arc::clone(&consensus_engine);
        
        tokio::spawn(async move {
            if let Err(e) = start_p2p_server(
                p2p_config,
                blockchain_for_p2p,
                mempool_for_p2p,
                reputation_for_p2p,
                consensus_for_p2p
            ).await {
                error!("Erreur fatale du serveur P2P: {}", e);
            }
        });
    }
    
    // Lancer l'API REST
    {
        let blockchain_for_api = Arc::clone(&blockchain);
        let mempool_for_api = Arc::clone(&mempool);
        let reputation_for_api = Arc::clone(&reputation_system);
        let consensus_for_api = Arc::clone(&consensus_engine);
        let storage_for_api = Arc::clone(&storage);
        
        tokio::spawn(async move {
            if let Err(e) = crate::p2p_network::start_api_server(
                config.api_port,
                blockchain_for_api,
                mempool_for_api,
                reputation_for_api,
                consensus_for_api,
                storage_for_api
            ).await {
                error!("Erreur fatale du serveur API: {}", e);
            }
        });
    }
    
    // Configurer et lancer le mining continu
    let mining_config = MiningConfig {
        interval: Duration::from_nanos(config.mining_interval_nanos),
        batch_size: config.mining_batch_size,
        difficulty_adjustment: config.mining_difficulty_adjustment,
        thread_priority: config.mining_thread_priority,
    };
    
    launch_continuous_mining(
        Arc::clone(&block_queue),
        mining_config,
        Arc::clone(&neural_pow),
        Arc::clone(&blockchain),
        Arc::clone(&mempool),
        Arc::clone(&consensus_engine),
    );
    
    // Tâche de sauvegarde périodique
    {
        let blockchain_for_save = Arc::clone(&blockchain);
        let storage_for_save = Arc::clone(&storage);
        let save_interval = Duration::from_secs(config.blockchain_save_interval);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(save_interval);
            loop {
                interval.tick().await;
                info!("Démarrage de la sauvegarde périodique de la blockchain");
                
                let blockchain_guard = blockchain_for_save.lock().await;
                match storage_for_save.save_blockchain(&blockchain_guard) {
                    Ok(_) => info!("Blockchain sauvegardée avec succès"),
                    Err(e) => error!("Erreur lors de la sauvegarde de la blockchain: {}", e),
                }
                
                drop(blockchain_guard); // Libérer explicitement le verrou
            }
        });
    }
    
    info!("NeuralChain est en cours d'exécution. Utilisez Ctrl+C pour arrêter");
    
    // Attendre le signal d'arrêt
    tokio::signal::ctrl_c().await?;
    info!("Signal d'arrêt reçu, arrêt en cours...");
    
    // Sauvegarder la blockchain avant de quitter
    let blockchain_guard = blockchain.lock().await;
    if let Err(e) = storage.save_blockchain(&blockchain_guard) {
        error!("Erreur lors de la sauvegarde finale de la blockchain: {}", e);
    } else {
        info!("Blockchain sauvegardée avec succès");
    }
    
    info!("Arrêt propre terminé");
    Ok(())
}
