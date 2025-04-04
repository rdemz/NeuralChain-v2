use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc, broadcast};
use tokio::task;
use uuid::Uuid;
use libp2p::{
    core::transport::upgrade,
    dns::DnsConfig,
    identity,
    mplex,
    noise,
    swarm::{SwarmBuilder, SwarmEvent},
    tcp::TokioTcpConfig,
    Swarm, PeerId, Transport,
    gossipsub::{
        self, Gossipsub, GossipsubEvent, GossipsubMessage, 
        MessageId, MessageAuthenticity, ValidationMode,
    },
    identify::{Identify, IdentifyConfig, IdentifyEvent},
    kad::{self, Kademlia, KademliaEvent, KademliaConfig, store::MemoryStore},
    ping::{Ping, PingConfig, PingEvent},
    NetworkBehaviour, core::muxing::StreamMuxerBox, floodsub::Topic,
};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use warp::{Filter, http::Response as HttpResponse};
use tracing::{info, warn, error, debug, trace};

use crate::block::Block;
use crate::blockchain::Blockchain;
use crate::config::P2PConfig;
use crate::consensus::ConsensusEngine;
use crate::mempool::OptimizedMempool;
use crate::reputation_system::ReputationSystem;
use crate::storage::OptimizedStorage;
use crate::transaction::Transaction;
use crate::monitoring::ACTIVE_CONNECTIONS;

/// Type de message P2P
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum P2PMessage {
    // Messages liés aux blocs
    NewBlock(Block),
    RequestBlock { hash: [u8; 32] },
    RequestBlockByHeight { height: u64 },
    BlockResponse { block: Option<Block> },
    
    // Messages liés aux transactions
    NewTransaction(Transaction),
    RequestTransaction { id: [u8; 32] },
    TransactionResponse { transaction: Option<Transaction> },
    
    // Messages liés au réseau
    Ping(u64),
    Pong(u64),
    Handshake { version: u32, genesis_hash: [u8; 32] },
    Disconnect { reason: String },
    
    // Messages liés à la synchronisation
    RequestBlockRange { start: u64, end: u64 },
    BlockRangeResponse { blocks: Vec<Block> },
    RequestMempool,
    MempoolResponse { transaction_ids: Vec<[u8; 32]> },
    
    // Messages liés au consensus
    ProposeBlock(Block),
    VoteBlock { hash: [u8; 32], approve: bool },
    ConsensusState { round: u64, leader: [u8; 32] },
}

/// Comportement réseau combiné
#[derive(NetworkBehaviour)]
#[behaviour(out_event = "P2PEvent")]
struct NeuralChainBehaviour {
    gossipsub: Gossipsub,
    kademlia: Kademlia<MemoryStore>,
    ping: Ping,
    identify: Identify,
}

/// Événements réseau
#[derive(Debug)]
enum P2PEvent {
    Gossipsub(GossipsubEvent),
    Kademlia(KademliaEvent),
    Ping(PingEvent),
    Identify(IdentifyEvent),
}

impl From<GossipsubEvent> for P2PEvent {
    fn from(event: GossipsubEvent) -> Self {
        P2PEvent::Gossipsub(event)
    }
}

impl From<KademliaEvent> for P2PEvent {
    fn from(event: KademliaEvent) -> Self {
        P2PEvent::Kademlia(event)
    }
}

impl From<PingEvent> for P2PEvent {
    fn from(event: PingEvent) -> Self {
        P2PEvent::Ping(event)
    }
}

impl From<IdentifyEvent> for P2PEvent {
    fn from(event: IdentifyEvent) -> Self {
        P2PEvent::Identify(event)
    }
}

/// Gestionnaire de réseau P2P
pub struct P2PNetwork {
    local_peer_id: PeerId,
    swarm: Arc<Mutex<Swarm<NeuralChainBehaviour>>>,
    message_sender: mpsc::Sender<(P2PMessage, Option<PeerId>)>,
    message_receiver: Arc<Mutex<mpsc::Receiver<(P2PMessage, Option<PeerId>)>>>,
    shutdown_sender: broadcast::Sender<()>,
    known_peers: Arc<Mutex<HashMap<PeerId, PeerInfo>>>,
    blockchain: Arc<Mutex<Blockchain>>,
    mempool: Arc<OptimizedMempool>,
    reputation_system: Arc<ReputationSystem>,
    consensus_engine: Arc<ConsensusEngine>,
}

/// Informations sur un pair
#[derive(Clone, Debug)]
struct PeerInfo {
    peer_id: PeerId,
    addresses: Vec<String>,
    version: Option<u32>,
    genesis_hash: Option<[u8; 32]>,
    last_seen: chrono::DateTime<chrono::Utc>,
    reputation: f64,
    connection_count: u32,
    connection_failures: u32,
    is_bootstrap_node: bool,
}

// Mutex partagée pour les collections de pairs
struct PeerCollections {
    connected_peers: HashSet<PeerId>,
    banned_peers: HashSet<PeerId>,
    peer_topics: HashMap<PeerId, Vec<String>>,
}

// Topics pour Gossipsub
const BLOCK_TOPIC: &str = "blocks";
const TX_TOPIC: &str = "transactions";
const CONSENSUS_TOPIC: &str = "consensus";
const DISCOVERY_TOPIC: &str = "discovery";

impl P2PNetwork {
    /// Crée un nouveau réseau P2P
    pub async fn new(
        config: P2PConfig,
        blockchain: Arc<Mutex<Blockchain>>,
        mempool: Arc<OptimizedMempool>,
        reputation_system: Arc<ReputationSystem>,
        consensus_engine: Arc<ConsensusEngine>,
    ) -> Result<Self> {
        // Créer une paire de clés pour l'identification locale
        let local_key = identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());
        info!("Peer ID local: {}", local_peer_id);
        
        // Créer le transport sécurisé
        let noise_keys = noise::Keypair::<noise::X25519Spec>::new()
            .into_authentic(&local_key)
            .expect("Échec de la création des clés Noise");
            
        let transport = TokioTcpConfig::new()
            .upgrade(upgrade::Version::V1)
            .authenticate(noise::NoiseConfig::xx(noise_keys).into_authenticated())
            .multiplex(mplex::MplexConfig::new())
            .boxed();
        
        // Configurer Gossipsub
        let gossipsub_config = gossipsub::GossipsubConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(10))
            .validation_mode(ValidationMode::Strict)
            .build()
            .map_err(|e| anyhow::anyhow!("Échec de la configuration de Gossipsub: {}", e))?;
            
        let gossipsub = Gossipsub::new(
            MessageAuthenticity::Signed(local_key.clone()),
            gossipsub_config
        )
        .map_err(|e| anyhow::anyhow!("Échec de la création de Gossipsub: {}", e))?;
        
        // Configurer Kademlia
        let mut kademlia_config = KademliaConfig::default();
        kademlia_config.set_query_timeout(Duration::from_secs(60));
        kademlia_config.set_publication_interval(None); // Désactiver la publication périodique
        
        let store = MemoryStore::new(local_peer_id);
        let mut kademlia = Kademlia::with_config(local_peer_id, store, kademlia_config);
        
        // Configurer Ping
        let ping = Ping::new(PingConfig::new()
            .with_interval(Duration::from_secs(config.ping_interval_secs))
            .with_timeout(Duration::from_secs(10))
        );
        
        // Configurer Identify
        let identify = Identify::new(IdentifyConfig::new(
            "/neuralchain/1.0.0".into(),
            local_key.public(),
        ));
        
        // Créer le comportement combiné
        let behaviour = NeuralChainBehaviour {
            gossipsub,
            kademlia,
            ping,
            identify,
        };
        
        // Créer le Swarm
        let mut swarm = SwarmBuilder::new(transport, behaviour, local_peer_id)
            .executor(Box::new(|fut| {
                tokio::spawn(fut);
            }))
            .build();
        
        // Créer les canaux de communication
        let (message_sender, message_receiver) = mpsc::channel(100);
        let (shutdown_sender, _) = broadcast::channel(1);
        
        // Configurer l'adresse d'écoute
        let listen_addr = format!("/ip4/{}/tcp/{}", config.listen_addr, config.port)
            .parse()
            .map_err(|e| anyhow::anyhow!("Adresse d'écoute invalide: {}", e))?;
            
        swarm.listen_on(listen_addr)?;
        
        // Abonnement aux topics
        let mut swarm_lock = swarm.lock().await;
        
        let block_topic = gossipsub::IdentTopic::new(BLOCK_TOPIC);
        let tx_topic = gossipsub::IdentTopic::new(TX_TOPIC);
        let consensus_topic = gossipsub::IdentTopic::new(CONSENSUS_TOPIC);
        let discovery_topic = gossipsub::IdentTopic::new(DISCOVERY_TOPIC);
        
        swarm_lock.behaviour_mut().gossipsub.subscribe(&block_topic)?;
        swarm_lock.behaviour_mut().gossipsub.subscribe(&tx_topic)?;
        swarm_lock.behaviour_mut().gossipsub.subscribe(&consensus_topic)?;
        swarm_lock.behaviour_mut().gossipsub.subscribe(&discovery_topic)?;
        
        // Ajouter les nœuds de bootstrap
        for peer in &config.bootstrap_peers {
            match peer.parse() {
                Ok(addr) => {
                    swarm_lock.behaviour_mut().kademlia.add_address(&addr, addr);
                    info!("Ajout du nœud de bootstrap: {}", addr);
                },
                Err(e) => {
                    warn!("Adresse de bootstrap invalide {}: {}", peer, e);
                }
            }
        }
        
        // Si Kademlia est activé, lancer une requête bootstrap
        if config.enable_kad {
            match swarm_lock.behaviour_mut().kademlia.bootstrap() {
                Ok(_) => info!("Bootstrap Kademlia lancé"),
                Err(e) => warn!("Échec du bootstrap Kademlia: {}", e),
            }
        }
        
        drop(swarm_lock);
        
        let p2p_network = Self {
            local_peer_id,
            swarm: Arc::new(Mutex::new(swarm)),
            message_sender,
            message_receiver: Arc::new(Mutex::new(message_receiver)),
            shutdown_sender,
            known_peers: Arc::new(Mutex::new(HashMap::new())),
            blockchain,
            mempool,
            reputation_system,
            consensus_engine,
        };
        
        Ok(p2p_network)
    }
    
    /// Démarre le réseau P2P
    pub async fn start(&self) -> Result<()> {
        let swarm = self.swarm.clone();
        let message_receiver = self.message_receiver.clone();
        let known_peers = self.known_peers.clone();
        let blockchain = self.blockchain.clone();
        let mempool = self.mempool.clone();
        let reputation_system = self.reputation_system.clone();
        let consensus_engine = self.consensus_engine.clone();
        
        // Canal pour les messages à envoyer
        let (message_sender, mut message_receiver) = mpsc::channel(100);
        
        // Spawner une tâche pour traiter les événements du swarm
        task::spawn(async move {
            let mut swarm_lock = swarm.lock().await;
            
            loop {
                tokio::select! {
                    // Événements du Swarm
                    event = swarm_lock.select_next_some() => {
                        match event {
                            SwarmEvent::NewListenAddr { address, .. } => {
                                info!("À l'écoute sur {}", address);
                            }
                            SwarmEvent::Behaviour(P2PEvent::Gossipsub(GossipsubEvent::Message {
                                propagation_source,
                                message_id,
                                message,
                            })) => {
                                process_gossipsub_message(
                                    &message,
                                    propagation_source,
                                    &blockchain,
                                    &mempool,
                                    &reputation_system,
                                    &message_sender,
                                    &known_peers,
                                ).await;
                            }
                            SwarmEvent::Behaviour(P2PEvent::Ping(PingEvent { peer, result, .. })) => {
                                match result {
                                    Ok(duration) => {
                                        trace!("Ping vers {} réussi en {:?}", peer, duration);
                                    }
                                    Err(err) => {
                                        debug!("Ping vers {} échoué: {}", peer, err);
                                    }
                                }
                            }
                            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                                info!("Connexion établie avec {}", peer_id);
                                
                                // Mettre à jour les métriques
                                ACTIVE_CONNECTIONS.increment(1);
                                
                                // Mettre à jour les pairs connus
                                update_peer_connection(&known_peers, peer_id, true).await;
                                
                                // Envoyer un handshake
                                let blockchain_guard = blockchain.lock().await;
                                let handshake = P2PMessage::Handshake {
                                    version: 1,  // Version du protocole
                                    genesis_hash: blockchain_guard.genesis_hash(),
                                };
                                drop(blockchain_guard);
                                
                                // Envoyer le message
                                let _ = message_sender.send((handshake, Some(peer_id))).await;
                            }
                            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                                info!("Connexion fermée avec {}", peer_id);
                                
                                // Mettre à jour les métriques
                                ACTIVE_CONNECTIONS.decrement(1);
                                
                                // Mettre à jour les pairs connus
                                update_peer_connection(&known_peers, peer_id, false).await;
                            }
                            SwarmEvent::Behaviour(P2PEvent::Identify(event)) => {
                                handle_identify_event(&known_peers, event).await;
                            }
                            SwarmEvent::Behaviour(P2PEvent::Kademlia(event)) => {
                                handle_kademlia_event(&known_peers, &swarm_lock, event).await;
                            }
                            _ => {} // Ignorer les autres événements
                        }
                    }
                    
                    // Messages de sortie à envoyer
                    Some((message, maybe_peer)) = message_receiver.recv() => {
                        if let Some(peer) = maybe_peer {
                            // Message à un pair spécifique
                            // Ceci nécessiterait une implémentation de message direct
                            // qui n'est pas supportée nativement par Gossipsub
                            warn!("Message direct non implémenté");
                        } else {
                            // Diffuser le message aux topics appropriés
                            let topic = match &message {
                                P2PMessage::NewBlock(_) | P2PMessage::RequestBlock { .. } |
                                P2PMessage::RequestBlockByHeight { .. } | P2PMessage::BlockResponse { .. } |
                                P2PMessage::RequestBlockRange { .. } | P2PMessage::BlockRangeResponse { .. } => {
                                    BLOCK_TOPIC
                                }
                                P2PMessage::NewTransaction(_) | P2PMessage::RequestTransaction { .. } |
                                P2PMessage::TransactionResponse { .. } | P2PMessage::RequestMempool |
                                P2PMessage::MempoolResponse { .. } => {
                                    TX_TOPIC
                                }
                                P2PMessage::ProposeBlock(_) | P2PMessage::VoteBlock { .. } |
                                P2PMessage::ConsensusState { .. } => {
                                    CONSENSUS_TOPIC
                                }
                                P2PMessage::Ping(_) | P2PMessage::Pong(_) |
                                P2PMessage::Handshake { .. } | P2PMessage::Disconnect { .. } => {
                                    DISCOVERY_TOPIC
                                }
                            };
                            
                            // Sérialiser et publier le message
                            match bincode::serialize(&message) {
                                Ok(bytes) => {
                                    let topic = gossipsub::IdentTopic::new(topic);
                                    match swarm_lock.behaviour_mut().gossipsub.publish(topic, bytes) {
                                        Ok(_) => trace!("Message publié"),
                                        Err(e) => warn!("Échec de la publication: {}", e),
                                    }
                                }
                                Err(e) => {
                                    error!("Échec de la sérialisation du message: {}", e);
                                }
                            }
                        }
                    }
                }
            }
        });
        
        // Spawner une tâche pour traiter les messages entrants
        let sender = self.message_sender.clone();
        let mut receiver = self.message_receiver.lock().await;
        
        task::spawn(async move {
            while let Some((message, peer)) = receiver.recv().await {
                // Transférer le message au handler de gossipsub
                let _ = message_sender.send((message, peer)).await;
            }
        });
        
        Ok(())
    }
    
    /// Diffuser un nouveau bloc au réseau
    pub async fn broadcast_block(&self, block: Block) -> Result<()> {
        let message = P2PMessage::NewBlock(block);
        self.message_sender.send((message, None)).await
            .map_err(|e| anyhow::anyhow!("Échec de l'envoi du message: {}", e))?;
            
        Ok(())
    }
    
    /// Diffuser une nouvelle transaction au réseau
    pub async fn broadcast_transaction(&self, transaction: Transaction) -> Result<()> {
        let message = P2PMessage::NewTransaction(transaction);
        self.message_sender.send((message, None)).await
            .map_err(|e| anyhow::anyhow!("Échec de l'envoi du message: {}", e))?;
            
        Ok(())
    }
    
    /// Demander un bloc spécifique à un pair
    pub async fn request_block(&self, hash: [u8; 32], peer: Option<PeerId>) -> Result<()> {
        let message = P2PMessage::RequestBlock { hash };
        self.message_sender.send((message, peer)).await
            .map_err(|e| anyhow::anyhow!("Échec de l'envoi du message: {}", e))?;
            
        Ok(())
    }
    
    /// Demander un bloc par sa hauteur
    pub async fn request_block_by_height(&self, height: u64, peer: Option<PeerId>) -> Result<()> {
        let message = P2PMessage::RequestBlockByHeight { height };
        self.message_sender.send((message, peer)).await
            .map_err(|e| anyhow::anyhow!("Échec de l'envoi du message: {}", e))?;
            
        Ok(())
    }
    
    /// Obtenir la liste des pairs connus
    pub async fn get_known_peers(&self) -> HashMap<PeerId, PeerInfo> {
        self.known_peers.lock().await.clone()
    }
    
    /// Arrêter proprement le réseau P2P
    pub fn shutdown(&self) -> Result<()> {
        let _ = self.shutdown_sender.send(());
        Ok(())
    }
}

/// Traiter un message Gossipsub
async fn process_gossipsub_message(
    message: &GossipsubMessage,
    source: PeerId,
    blockchain: &Arc<Mutex<Blockchain>>,
    mempool: &Arc<OptimizedMempool>,
    reputation_system: &Arc<ReputationSystem>,
    message_sender: &mpsc::Sender<(P2PMessage, Option<PeerId>)>,
    known_peers: &Arc<Mutex<HashMap<PeerId, PeerInfo>>>,
) {
    // Désérialiser le message
    match bincode::deserialize::<P2PMessage>(&message.data) {
        Ok(p2p_message) => {
            match p2p_message {
                P2PMessage::NewBlock(block) => {
                    // Valider et ajouter le bloc
                    let mut blockchain_guard = blockchain.lock().await;
                    
                    match blockchain_guard.add_block(block.clone()) {
                        Ok(_) => {
                            info!("Nouveau bloc reçu et ajouté: hauteur={}, hash={}", 
                                 block.height, hex::encode(&block.hash[0..8]));
                                 
                            // Retirer les transactions du mempool
                            let tx_ids: Vec<_> = block.transactions.iter().map(|tx| tx.id).collect();
                            mempool.mark_transactions_included(&tx_ids);
                            
                            // Mettre à jour la réputation du pair qui a partagé le bloc
                            let source_bytes = source.to_bytes();
                            let mut source_id = [0u8; 32];
                            source_id[..source_bytes.len().min(32)].copy_from_slice(&source_bytes[..source_bytes.len().min(32)]);
                            
                            reputation_system.record_successful_mining(source_id, block.hash, block.difficulty);
                        }
                        Err(e) => {
                            warn!("Bloc rejeté: {}", e);
                        }
                    }
                },
                P2PMessage::NewTransaction(transaction) => {
                    // Ajouter la transaction au mempool
                    if mempool.add_transaction(transaction.clone()) {
                        debug!("Nouvelle transaction ajoutée au mempool: {}", 
                               hex::encode(&transaction.id[0..8]));
                               
                        // Améliorer la réputation du pair pour avoir partagé une transaction valide
                        let source_bytes = source.to_bytes();
                        let mut source_id = [0u8; 32];
                        source_id[..source_bytes.len().min(32)].copy_from_slice(&source_bytes[..source_bytes.len().min(32)]);
                        
                        reputation_system.record_validation_contribution(source_id, 1, 1.0);
                    } else {
                        trace!("Transaction rejetée ou déjà connue");
                    }
                },
                P2PMessage::RequestBlock { hash } => {
                    // Répondre avec le bloc demandé
                    let blockchain_guard = blockchain.lock().await;
                    let block = blockchain_guard.get_block_by_hash(&hash);
                    
                    let response = P2PMessage::BlockResponse { block };
                    let _ = message_sender.send((response, Some(source))).await;
                },
                P2PMessage::RequestBlockByHeight { height } => {
                    // Répondre avec le bloc demandé
                    let blockchain_guard = blockchain.lock().await;
                    let block = blockchain_guard.get_block_by_height(height);
                    
                    let response = P2PMessage::BlockResponse { block };
                    let _ = message_sender.send((response, Some(source))).await;
                },
                P2PMessage::Handshake { version, genesis_hash } => {
                    // Traiter le handshake
                    info!("Handshake reçu de {}: version={}", source, version);
                    
                    // Vérifier la compatibilité
                    let blockchain_guard = blockchain.lock().await;
                    let our_genesis = blockchain_guard.genesis_hash();
                    
                    if genesis_hash != our_genesis {
                        warn!("Pair {} avec un genesis incompatible", source);
                        
                        // Déconnecter le pair
                        let disconnect = P2PMessage::Disconnect { 
                            reason: "Genesis incompatible".to_string() 
                        };
                        let _ = message_sender.send((disconnect, Some(source))).await;
                    } else {
                        // Mettre à jour les informations du pair
                        let mut peers = known_peers.lock().await;
                        if let Some(peer_info) = peers.get_mut(&source) {
                            peer_info.version = Some(version);
                            peer_info.genesis_hash = Some(genesis_hash);
                        }
                    }
                },
                P2PMessage::RequestMempool => {
                    // Répondre avec les IDs des transactions du mempool
                    let transactions = mempool.get_transactions_for_block(1000, 1_000_000);
                    let transaction_ids: Vec<_> = transactions.iter().map(|tx| tx.id).collect();
                    
                    let response = P2PMessage::MempoolResponse { transaction_ids };
                    let _ = message_sender.send((response, Some(source))).await;
                },
                P2PMessage::RequestBlockRange { start, end } => {
                    // Répondre avec une plage de blocs
                    let mut blocks = Vec::new();
                    let blockchain_guard = blockchain.lock().await;
                    
                    for height in start..=end.min(start + 100) { // Limiter à 100 blocs
                        if let Some(block) = blockchain_guard.get_block_by_height(height) {
                            blocks.push(block);
                        } else {
                            break;
                        }
                    }
                    
                    let response = P2PMessage::BlockRangeResponse { blocks };
                    let _ = message_sender.send((response, Some(source))).await;
                },
                _ => {
                    // Autres types de messages traités dans leurs gestionnaires respectifs
                                        trace!("Message de type {:?} reçu de {}", 
                          std::mem::discriminant(&p2p_message),
                          source);
                }
            }
        },
        Err(e) => {
            warn!("Échec de désérialisation du message Gossipsub: {}", e);
        }
    }
}

/// Gérer un événement Identify
async fn handle_identify_event(
    known_peers: &Arc<Mutex<HashMap<PeerId, PeerInfo>>>,
    event: IdentifyEvent,
) {
    match event {
        IdentifyEvent::Received { peer_id, info } => {
            debug!("Identify reçu de {}: {}", peer_id, info.agent_version);
            
            // Mettre à jour les informations du pair
            let mut peers = known_peers.lock().await;
            let peer_info = peers.entry(peer_id).or_insert_with(|| PeerInfo {
                peer_id,
                addresses: Vec::new(),
                version: None,
                genesis_hash: None,
                last_seen: chrono::Utc::now(),
                reputation: 0.5,
                connection_count: 0,
                connection_failures: 0,
                is_bootstrap_node: false,
            });
            
            // Ajouter les adresses
            for addr in info.listen_addrs {
                let addr_str = addr.to_string();
                if !peer_info.addresses.contains(&addr_str) {
                    peer_info.addresses.push(addr_str);
                }
            }
            
            peer_info.last_seen = chrono::Utc::now();
        }
        _ => {}
    }
}

/// Gérer un événement Kademlia
async fn handle_kademlia_event(
    known_peers: &Arc<Mutex<HashMap<PeerId, PeerInfo>>>,
    swarm: &Swarm<NeuralChainBehaviour>,
    event: KademliaEvent,
) {
    match event {
        KademliaEvent::RoutingUpdated { peer, .. } => {
            debug!("Routage Kademlia mis à jour pour {}", peer);
            
            // Mettre à jour les informations du pair
            let mut peers = known_peers.lock().await;
            if !peers.contains_key(&peer) {
                peers.insert(peer, PeerInfo {
                    peer_id: peer,
                    addresses: Vec::new(),
                    version: None,
                    genesis_hash: None,
                    last_seen: chrono::Utc::now(),
                    reputation: 0.5,
                    connection_count: 0,
                    connection_failures: 0,
                    is_bootstrap_node: false,
                });
            }
        }
        KademliaEvent::OutboundQueryCompleted { result, .. } => {
            match result {
                kad::QueryResult::Bootstrap(Ok(_)) => {
                    debug!("Bootstrap Kademlia terminé avec succès");
                    
                    // Lancer une recherche DHT pour notre propre pair ID
                    // pour optimiser notre position dans la DHT
                    if let Err(e) = swarm.behaviour_mut().kademlia.get_closest_peers(swarm.local_peer_id().clone()) {
                        warn!("Échec de la recherche de pairs proches: {}", e);
                    }
                }
                kad::QueryResult::Bootstrap(Err(e)) => {
                    warn!("Bootstrap Kademlia échoué: {}", e);
                }
                kad::QueryResult::GetClosestPeers(Ok(result)) => {
                    debug!("Trouvé {} pairs proches", result.peers.len());
                    
                    // Mettre à jour les informations des pairs
                    let mut peers = known_peers.lock().await;
                    for peer in result.peers {
                        if !peers.contains_key(&peer) {
                            peers.insert(peer, PeerInfo {
                                peer_id: peer,
                                addresses: Vec::new(),
                                version: None,
                                genesis_hash: None,
                                last_seen: chrono::Utc::now(),
                                reputation: 0.5,
                                connection_count: 0,
                                connection_failures: 0,
                                is_bootstrap_node: false,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
}

/// Mettre à jour les informations de connexion d'un pair
async fn update_peer_connection(
    known_peers: &Arc<Mutex<HashMap<PeerId, PeerInfo>>>,
    peer_id: PeerId,
    connected: bool,
) {
    let mut peers = known_peers.lock().await;
    let peer_info = peers.entry(peer_id).or_insert_with(|| PeerInfo {
        peer_id,
        addresses: Vec::new(),
        version: None,
        genesis_hash: None,
        last_seen: chrono::Utc::now(),
        reputation: 0.5,
        connection_count: 0,
        connection_failures: 0,
        is_bootstrap_node: false,
    });
    
    if connected {
        peer_info.connection_count += 1;
        peer_info.last_seen = chrono::Utc::now();
    } else {
        peer_info.connection_failures += 1;
    }
}

/// Démarrer le serveur P2P
pub async fn start_p2p_server(
    config: P2PConfig,
    blockchain: Arc<Mutex<Blockchain>>,
    mempool: Arc<OptimizedMempool>,
    reputation_system: Arc<ReputationSystem>,
    consensus_engine: Arc<ConsensusEngine>,
) -> Result<()> {
    // Créer et démarrer le réseau P2P
    let p2p_network = P2PNetwork::new(
        config.clone(), 
        blockchain.clone(),
        mempool.clone(),
        reputation_system.clone(),
        consensus_engine.clone()
    ).await?;
    
    p2p_network.start().await?;
    
    // Attendre indéfiniment
    let forever = std::future::pending::<()>();
    forever.await;
    
    Ok(())
}

/// Démarrer le serveur API REST
pub async fn start_api_server(
    port: u16,
    blockchain: Arc<Mutex<Blockchain>>,
    mempool: Arc<OptimizedMempool>,
    reputation_system: Arc<ReputationSystem>,
    consensus_engine: Arc<ConsensusEngine>,
    storage: Arc<OptimizedStorage>,
) -> Result<()> {
    // Route pour la santé du service
    let health_route = warp::path("health")
        .map(|| "OK");
    
    // Route pour les informations de la blockchain
    let blockchain_clone = blockchain.clone();
    let blockchain_info_route = warp::path("blockchain")
        .and(warp::path("info"))
        .and_then(move || {
            let blockchain = blockchain_clone.clone();
            async move {
                let blockchain_guard = blockchain.lock().await;
                let info = json!({
                    "height": blockchain_guard.height(),
                    "difficulty": blockchain_guard.current_difficulty(),
                    "last_block_hash": hex::encode(blockchain_guard.last_block_hash()),
                    "last_block_time": blockchain_guard.last_block_time(),
                    "blockchain_size": blockchain_guard.size_bytes()
                });
                Ok::<_, warp::Rejection>(warp::reply::json(&info))
            }
        });
    
    // Route pour récupérer un bloc par sa hauteur
    let blockchain_clone = blockchain.clone();
    let block_by_height_route = warp::path!("blockchain" / "blocks" / u64)
        .and_then(move |height: u64| {
            let blockchain = blockchain_clone.clone();
            async move {
                let blockchain_guard = blockchain.lock().await;
                match blockchain_guard.get_block_by_height(height) {
                    Some(block) => Ok(warp::reply::json(&block)),
                    None => Ok(warp::reply::with_status(
                        warp::reply::json(&json!({"error": "Block not found"})),
                        warp::http::StatusCode::NOT_FOUND,
                    ))
                }
            }
        });
    
    // Route pour récupérer un bloc par son hash
    let blockchain_clone = blockchain.clone();
    let block_by_hash_route = warp::path!("blockchain" / "blocks" / "hash" / String)
        .and_then(move |hash_hex: String| {
            let blockchain = blockchain_clone.clone();
            async move {
                match hex::decode(&hash_hex) {
                    Ok(hash_bytes) => {
                        if hash_bytes.len() != 32 {
                            return Ok(warp::reply::with_status(
                                warp::reply::json(&json!({"error": "Invalid hash length"})),
                                warp::http::StatusCode::BAD_REQUEST,
                            ));
                        }
                        
                        let mut hash = [0u8; 32];
                        hash.copy_from_slice(&hash_bytes);
                        
                        let blockchain_guard = blockchain.lock().await;
                        match blockchain_guard.get_block_by_hash(&hash) {
                            Some(block) => Ok(warp::reply::json(&block)),
                            None => Ok(warp::reply::with_status(
                                warp::reply::json(&json!({"error": "Block not found"})),
                                warp::http::StatusCode::NOT_FOUND,
                            ))
                        }
                    }
                    Err(_) => {
                        Ok(warp::reply::with_status(
                            warp::reply::json(&json!({"error": "Invalid hash format"})),
                            warp::http::StatusCode::BAD_REQUEST,
                        ))
                    }
                }
            }
        });
    
    // Route pour récupérer une transaction par son ID
    let blockchain_clone = blockchain.clone();
    let mempool_clone = mempool.clone();
    let tx_by_id_route = warp::path!("blockchain" / "transactions" / String)
        .and_then(move |tx_id_hex: String| {
            let blockchain = blockchain_clone.clone();
            let mempool = mempool_clone.clone();
            async move {
                match hex::decode(&tx_id_hex) {
                    Ok(tx_bytes) => {
                        if tx_bytes.len() != 32 {
                            return Ok(warp::reply::with_status(
                                warp::reply::json(&json!({"error": "Invalid transaction ID length"})),
                                warp::http::StatusCode::BAD_REQUEST,
                            ));
                        }
                        
                        let mut tx_id = [0u8; 32];
                        tx_id.copy_from_slice(&tx_bytes);
                        
                        // Vérifier d'abord dans le mempool
                        if let Some(tx) = mempool.get_transaction(&tx_id) {
                            return Ok(warp::reply::json(&json!({
                                "transaction": tx,
                                "status": "pending"
                            })));
                        }
                        
                        // Sinon, chercher dans la blockchain
                        let blockchain_guard = blockchain.lock().await;
                        match blockchain_guard.get_transaction_by_id(&tx_id) {
                            Some((tx, block_hash, height)) => {
                                Ok(warp::reply::json(&json!({
                                    "transaction": tx,
                                    "status": "confirmed",
                                    "block_hash": hex::encode(block_hash),
                                    "height": height
                                })))
                            }
                            None => Ok(warp::reply::with_status(
                                warp::reply::json(&json!({"error": "Transaction not found"})),
                                warp::http::StatusCode::NOT_FOUND,
                            ))
                        }
                    }
                    Err(_) => {
                        Ok(warp::reply::with_status(
                            warp::reply::json(&json!({"error": "Invalid transaction ID format"})),
                            warp::http::StatusCode::BAD_REQUEST,
                        ))
                    }
                }
            }
        });
    
    // Route pour soumettre une nouvelle transaction
    let mempool_clone = mempool.clone();
    let submit_tx_route = warp::path!("blockchain" / "transactions")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(move |tx: Transaction| {
            let mempool = mempool_clone.clone();
            async move {
                if mempool.add_transaction(tx.clone()) {
                    Ok(warp::reply::with_status(
                        warp::reply::json(&json!({"status": "accepted", "transaction_id": hex::encode(tx.id)})),
                        warp::http::StatusCode::ACCEPTED,
                    ))
                } else {
                    Ok(warp::reply::with_status(
                        warp::reply::json(&json!({"status": "rejected", "reason": "Transaction already exists or invalid"})),
                        warp::http::StatusCode::BAD_REQUEST,
                    ))
                }
            }
        });
    
    // Route pour récupérer les informations du mempool
    let mempool_clone = mempool.clone();
    let mempool_info_route = warp::path!("blockchain" / "mempool")
        .and_then(move || {
            let mempool = mempool_clone.clone();
            async move {
                let (added, rejected, expired, included, replaced) = mempool.get_stats();
                let info = json!({
                    "size": mempool.size(),
                    "stats": {
                        "added": added,
                        "rejected": rejected,
                        "expired": expired,
                        "included": included,
                        "replaced": replaced
                    }
                });
                Ok::<_, warp::Rejection>(warp::reply::json(&info))
            }
        });
    
    // Combiner toutes les routes
    let routes = health_route
        .or(blockchain_info_route)
        .or(block_by_height_route)
        .or(block_by_hash_route)
        .or(tx_by_id_route)
        .or(submit_tx_route)
        .or(mempool_info_route)
        .with(warp::cors().allow_any_origin())
        .recover(handle_rejection);
    
    // Démarrer le serveur
    info!("API REST démarrée sur 0.0.0.0:{}", port);
    warp::serve(routes).run(([0, 0, 0, 0], port)).await;
    
    Ok(())
}

/// Gérer les rejets Warp
async fn handle_rejection(err: warp::Rejection) -> Result<impl warp::Reply, std::convert::Infallible> {
    let code;
    let message;

    if err.is_not_found() {
        code = warp::http::StatusCode::NOT_FOUND;
        message = "Resource not found".to_string();
    } else if let Some(e) = err.find::<warp::filters::body::BodyDeserializeError>() {
        code = warp::http::StatusCode::BAD_REQUEST;
        message = format!("Invalid request body: {}", e);
    } else {
        code = warp::http::StatusCode::INTERNAL_SERVER_ERROR;
        message = "Internal server error".to_string();
        error!("Unhandled rejection: {:?}", err);
    }

    let json = warp::reply::json(&json!({
        "error": message
    }));

    Ok(warp::reply::with_status(json, code))
}
