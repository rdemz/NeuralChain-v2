use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use crossbeam::channel::{self, Receiver, Sender};
use parking_lot::Mutex;
use rayon::prelude::*;
use spin_sleep::sleep;

use crate::block::{Block, BlockHeader};
use crate::blockchain::{Blockchain, BlockchainState};
use crate::consensus::ConsensusEngine;
use crate::mempool::OptimizedMempool;
use crate::monitoring::BLOCK_MINING_TIME;
use crate::neural_pow::{NeuralPoW, PoWResult};

// Configuration du mining
pub struct MiningConfig {
    pub interval: Duration,        // Intervalle entre tentatives de mining
    pub batch_size: usize,         // Taille du lot de nonces à essayer
    pub difficulty_adjustment: f64, // Facteur d'ajustement dynamique de difficulté
    pub thread_priority: i32,      // Priorité du thread de mining
}

// Structure pour gérer une file de blocs minés
pub struct BlockQueue {
    queue: Mutex<Vec<Block>>,
    capacity: usize,
}

impl BlockQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Mutex::new(Vec::with_capacity(capacity)),
            capacity,
        }
    }
    
    // Ajouter un bloc à la file
    pub fn push(&self, block: Block) -> bool {
        let mut queue = self.queue.lock();
        
        if queue.len() < self.capacity {
            queue.push(block);
            true
        } else {
            false
        }
    }
    
    // Récupérer tous les blocs de la file
    pub fn drain(&self) -> Vec<Block> {
        let mut queue = self.queue.lock();
        std::mem::take(&mut *queue)
    }
    
    // Récupérer le nombre de blocs dans la file
    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }
}

// Lancement du mining continu dans un thread dédié
pub fn launch_continuous_mining(
    block_queue: Arc<BlockQueue>,
    config: MiningConfig,
    pow_system: Arc<Mutex<NeuralPoW>>,
    blockchain: Arc<tokio::sync::Mutex<Blockchain>>,
    mempool: Arc<OptimizedMempool>,
    consensus_engine: Arc<ConsensusEngine>,
) {
    // Créer des canaux pour communiquer entre les threads
    let (tx, rx) = channel::bounded(1);
    
    // Lancer le thread de mining
    thread::Builder::new()
        .name("continuous-mining".to_string())
        .spawn(move || {
            // Configurer la priorité du thread
            #[cfg(unix)]
            {
                use std::os::unix::thread::JoinHandleExt;
                let native_id = unsafe { libc::pthread_self() };
                let mut sched_param: libc::sched_param = unsafe { std::mem::zeroed() };
                sched_param.sched_priority = config.thread_priority;
                unsafe {
                    libc::pthread_setschedparam(
                        native_id,
                        libc::SCHED_RR,
                        &sched_param,
                    );
                }
            }
            
            continuous_mining_loop(
                block_queue,
                config,
                pow_system,
                blockchain,
                mempool,
                consensus_engine,
                rx,
            );
        })
        .expect("Échec de la création du thread de mining");
        
    // Le canal d'arrêt peut être utilisé plus tard pour arrêter proprement le mining
    let _ = tx;
}

fn continuous_mining_loop(
    block_queue: Arc<BlockQueue>,
    config: MiningConfig,
    pow_system: Arc<Mutex<NeuralPoW>>,
    blockchain: Arc<tokio::sync::Mutex<Blockchain>>,
    mempool: Arc<OptimizedMempool>,
    consensus_engine: Arc<ConsensusEngine>,
    rx: Receiver<()>,
) {
    // Compteur de blocs minés
    let mut blocks_mined = 0;
    
    loop {
        // Vérifier si on a reçu un signal d'arrêt
        if rx.try_recv().is_ok() {
            tracing::info!("Arrêt du thread de mining");
            break;
        }
        
        // Limiter la taille de la file d'attente des blocs
        if block_queue.len() >= block_queue.capacity / 2 {
            tracing::debug!("La file de blocs est à moitié pleine, pause du mining");
            sleep(Duration::from_millis(1000));
            continue;
        }
        
        // Mining en continu
        let mining_result = mine_next_block(
            &blockchain,
            &mempool,
            &pow_system,
            &consensus_engine,
            config.batch_size,
        );
        
        match mining_result {
            Ok(Some(block)) => {
                tracing::info!(
                    "Bloc miné avec succès: hauteur={}, hash={}, txs={}, difficulté={}",
                    block.height,
                    hex::encode(&block.hash),
                    block.transactions.len(),
                    block.difficulty,
                );
                
                // Ajouter le bloc à la file
                if !block_queue.push(block) {
                    tracing::warn!("File de blocs pleine, bloc perdu");
                }
                
                blocks_mined += 1;
                
                // Ajuster dynamiquement le délai entre tentatives
                let delay = if blocks_mined % 10 == 0 {
                    // Tous les 10 blocs, prendre une pause plus longue
                    config.interval * 5
                } else {
                    // Pause normale
                    config.interval
                };
                
                sleep(delay);
            }
            Ok(None) => {
                // Pas de bloc miné cette fois-ci
                tracing::trace!("Tentative de mining infructueuse");
                sleep(config.interval);
            }
            Err(e) => {
                tracing::error!("Erreur lors du mining: {}", e);
                sleep(config.interval * 2); // Pause plus longue en cas d'erreur
            }
        }
    }
}

// Fonction pour miner le prochain bloc
fn mine_next_block(
    blockchain: &Arc<tokio::sync::Mutex<Blockchain>>,
    mempool: &Arc<OptimizedMempool>,
    pow_system: &Arc<Mutex<NeuralPoW>>,
    consensus_engine: &Arc<ConsensusEngine>,
    batch_size: usize,
) -> Result<Option<Block>, anyhow::Error> {
    let start_time = Instant::now();
    
    // Acquérir l'état actuel de la blockchain
    let blockchain_guard = match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle.block_on(async { blockchain.lock().await }),
        Err(_) => {
            // Créer un nouveau runtime si nécessaire
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async { blockchain.lock().await })
        }
    };
    
    // Récupérer des infos de base
    let height = blockchain_guard.height() + 1;
    let parent_hash = blockchain_guard.last_block_hash();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    
    // Récupérer le mineur (notre adresse)
    let miner = consensus_engine.get_miner_address();
    
    // Récupérer les transactions du mempool
    let transactions = mempool.get_transactions_for_block(1000, 1_000_000);
    
    // Créer l'en-tête du bloc à miner
    let header = BlockHeader {
        version: 1,
        parent_hash,
        timestamp,
        merkle_root: [0; 32], // Sera calculé plus tard
        height,
        nonce: 0,
        miner: miner.clone(),
        difficulty: 0, // Sera défini plus tard
    };
    
    // Construire l'état de la blockchain pour le PoW
    let state = BlockchainState {
        mempool_size: mempool.size(),
        average_fee_last_100_blocks: blockchain_guard.average_fee_last_n_blocks(100),
        max_fee_ever: blockchain_guard.max_fee_ever(),
        hashrate_estimate: blockchain_guard.estimate_hashrate(),
        network_activity_score: 0.7, // Exemple
        node_distribution_entropy: 0.8, // Exemple
        avg_block_time_last_100: blockchain_guard.average_block_time(100),
        avg_transactions_per_block_last_100: blockchain_guard.average_transactions_per_block(100),
        difficulty_adjustment_factor: 1.0,
        average_transaction_value: blockchain_guard.average_transaction_value(100),
        max_transaction_value_ever: blockchain_guard.max_transaction_value_ever(),
        fee_to_reward_ratio: blockchain_guard.fee_to_reward_ratio(100),
        utxo_set_size: blockchain_guard.utxo_set_size(),
    };
    
    // Libérer la blockchain pour ne pas bloquer d'autres processus pendant le mining
    drop(blockchain_guard);
    
    // Calculer la target pour le mining
    let header_bytes = bincode::serialize(&header)?;
    let target = pow_system.lock().calculate_target(&header, &state);
    
    // Miner le bloc avec la parallélisation
    let pow_result = pow_system.lock().mine_block(
        &header_bytes, 
        &target, 
        batch_size as u64
    );
    
    match pow_result {
        Some(pow) => {
            // Bloc miné avec succès!
            let elapsed = start_time.elapsed();
            tracing::debug!(
                "Bloc miné en {}.{:03}s, nonce={}", 
                elapsed.as_secs(), 
                elapsed.subsec_millis(),
                pow.nonce
            );
            
            BLOCK_MINING_TIME.record(elapsed.as_millis() as f64);
            
            // Récupérer à nouveau la blockchain pour créer le bloc final
            let blockchain_guard = match tokio::runtime::Handle::try_current() {
                Ok(handle) => handle.block_on(async { blockchain.lock().await }),
                Err(_) => {
                    let rt = tokio::runtime::Runtime::new()?;
                    rt.block_on(async { blockchain.lock().await })
                }
            };
            
            // Construire le bloc final
            let block = blockchain_guard.create_block(
                height,
                parent_hash,
                timestamp,
                pow.nonce,
                miner,
                transactions,
                pow.target.clone(),
            )?;
            
            Ok(Some(block))
        }
        None => {
            // Pas de solution trouvée dans ce lot de nonces
            tracing::trace!("Pas de solution trouvée avec batch_size={}", batch_size);
            Ok(None)
        }
    }
}
