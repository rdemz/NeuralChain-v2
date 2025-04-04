use crate::network_protocol::{NetworkMessage, PeerInfo};
use anyhow::{Result, Context, bail};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use libp2p::{
    identity,
    PeerId,
    Transport,
    core::transport::upgrade,
    noise,
    yamux,
};
use std::net::{SocketAddr, IpAddr};
use tracing::{info, warn, error, debug};

/// Configuration du gestionnaire de pairs
pub struct PeerManagerConfig {
    pub max_peers: usize,
    pub min_peers: usize,
    pub peer_timeout_secs: u64,
    pub reconnect_interval_secs: u64,
    pub discovery_interval_secs: u64,
    pub bootstrap_peers: Vec<String>,
}

impl Default for PeerManagerConfig {
    fn default() -> Self {
        Self {
            max_peers: 25,
            min_peers: 8,
            peer_timeout_secs: 300, // 5 minutes
            reconnect_interval_secs: 60, // 1 minute
            discovery_interval_secs: 300, // 5 minutes
            bootstrap_peers: vec![],
        }
    }
}

/// État d'une connexion à un pair
#[derive(Debug, Clone, PartialEq)]
pub enum PeerConnectionState {
    Connected,
    Disconnected,
    Connecting,
    Failed,
    Banned,
}

/// Informations détaillées sur un pair connecté
#[derive(Debug, Clone)]
pub struct ConnectedPeer {
    pub info: PeerInfo,
    pub state: PeerConnectionState,
    pub last_seen: Instant,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub connection_attempts: u32,
    pub last_ping_latency_ms: Option<u64>,
}

/// Gestionnaire de pairs pour NeuralChain
pub struct PeerManager {
    config: PeerManagerConfig,
    peers: Arc<RwLock<HashMap<String, ConnectedPeer>>>,
    banned_peers: Arc<RwLock<HashSet<String>>>,
    message_sender: mpsc::Sender<(String, NetworkMessage)>,
    message_receiver: Arc<Mutex<mpsc::Receiver<(String, NetworkMessage)>>>,
    local_peer_id: String,
}

impl PeerManager {
    /// Crée un nouveau gestionnaire de pairs
    pub fn new(config: PeerManagerConfig) -> Self {
        let (tx, rx) = mpsc::channel(1000);
        
        // Générer un ID de pair local
        let local_keypair = identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_keypair.public()).to_string();
        
        Self {
            config,
            peers: Arc::new(RwLock::new(HashMap::new())),
            banned_peers: Arc::new(RwLock::new(HashSet::new())),
            message_sender: tx,
            message_receiver: Arc::new(Mutex::new(rx)),
            local_peer_id,
        }
    }
    
    /// Démarre le gestionnaire de pairs
    pub async fn start(&self) -> Result<()> {
        // Démarre les tâches de maintenance
        self.start_maintenance_tasks().await?;
        
        // Se connecte aux pairs de bootstrap
        for peer_addr in &self.config.bootstrap_peers {
            self.connect_to_peer(peer_addr).await?;
        }
        
        Ok(())
    }
    
    /// Démarrer les tâches de maintenance périodiques
    async fn start_maintenance_tasks(&self) -> Result<()> {
        let peers_clone = self.peers.clone();
        let banned_clone = self.banned_peers.clone();
        let config = self.config.clone();
        
        // Tâche de nettoyage des pairs inactifs
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                let now = Instant::now();
                let mut peers_write = peers_clone.write().await;
                let timeout = Duration::from_secs(config.peer_timeout_secs);
                
                peers_write.retain(|_, peer| {
                    now.duration_since(peer.last_seen) < timeout || 
                    peer.state == PeerConnectionState::Connecting
                });
            }
        });
        
        // Tâche de reconnexion aux pairs déconnectés
        let peers_for_reconnect = self.peers.clone();
        let config_for_reconnect = self.config.clone();
        let pm_clone = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(config_for_reconnect.reconnect_interval_secs));
            
            loop {
                interval.tick().await;
                
                let peers_read = peers_for_reconnect.read().await;
                let disconnected_peers: Vec<String> = peers_read
                    .iter()
                    .filter(|(_, peer)| {
                        peer.state == PeerConnectionState::Disconnected && 
                        peer.connection_attempts < 3
                    })
                    .map(|(addr, _)| addr.clone())
                    .collect();
                
                drop(peers_read);
                
                for peer_addr in disconnected_peers {
                    if let Err(e) = pm_clone.connect_to_peer(&peer_addr).await {
                        warn!("Échec de reconnexion au pair {}: {}", peer_addr, e);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Se connecte à un pair par son adresse
    pub async fn connect_to_peer(&self, peer_addr: &str) -> Result<()> {
        // Vérifier si le pair est banni
        {
            let banned = self.banned_peers.read().await;
            if banned.contains(peer_addr) {
                bail!("Le pair {} est banni", peer_addr);
            }
        }
        
        // Mise à jour de l'état du pair
        {
            let mut peers = self.peers.write().await;
            
            if let Some(mut peer) = peers.get_mut(peer_addr).cloned() {
                peer.state = PeerConnectionState::Connecting;
                peer.connection_attempts += 1;
                peers.insert(peer_addr.to_string(), peer);
            } else {
                // Nouveau pair
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                    
                let info = PeerInfo {
                    id: peer_addr.to_string(),
                    address: peer_addr.to_string(),
                    port: 30333, // Port par défaut
                    last_seen: timestamp,
                    reputation_score: 0.5,
                };
                
                let new_peer = ConnectedPeer {
                    info,
                    state: PeerConnectionState::Connecting,
                    last_seen: Instant::now(),
                    messages_sent: 0,
                    messages_received: 0,
                    connection_attempts: 1,
                    last_ping_latency_ms: None,
                };
                
                peers.insert(peer_addr.to_string(), new_peer);
            }
        }
        
        // La connexion effective serait implémentée ici
        // Pour l'exemple, on simule une connexion réussie
        {
            let mut peers = self.peers.write().await;
            if let Some(mut peer) = peers.get_mut(peer_addr).cloned() {
                peer.state = PeerConnectionState::Connected;
                peer.last_seen = Instant::now();
                peers.insert(peer_addr.to_string(), peer);
                info!("Connecté au pair {}", peer_addr);
            }
        }
        
        Ok(())
    }
    
    /// Déconnecte un pair
    pub async fn disconnect_peer(&self, peer_id: &str, reason: &str) -> Result<()> {
        let mut peers = self.peers.write().await;
        
        if let Some(mut peer) = peers.get_mut(peer_id).cloned() {
            peer.state = PeerConnectionState::Disconnected;
            peers.insert(peer_id.to_string(), peer);
            info!("Déconnecté du pair {}: {}", peer_id, reason);
        }
        
        Ok(())
    }
    
    /// Envoie un message à un pair spécifique
    pub async fn send_message(&self, peer_id: &str, message: NetworkMessage) -> Result<()> {
        // Vérifier si le pair est connecté
        let is_connected = {
            let peers = self.peers.read().await;
            if let Some(peer) = peers.get(peer_id) {
                peer.state == PeerConnectionState::Connected
            } else {
                false
            }
        };
        
        if !is_connected {
            bail!("Tentative d'envoi de message à un pair non connecté: {}", peer_id);
        }
        
        // Incrémenter le compteur de messages
        {
            let mut peers = self.peers.write().await;
            if let Some(mut peer) = peers.get_mut(peer_id).cloned() {
                peer.messages_sent += 1;
                peers.insert(peer_id.to_string(), peer);
            }
        }
        
        // Envoyer le message via le canal
        self.message_sender.send((peer_id.to_string(), message)).await
            .context("Échec de l'envoi du message dans le canal")?;
            
        Ok(())
    }
    
    /// Diffuse un message à tous les pairs connectés
    pub async fn broadcast_message(&self, message: NetworkMessage) -> Result<usize> {
        let peers = self.peers.read().await;
        let connected_peers: Vec<String> = peers
            .iter()
            .filter(|(_, peer)| peer.state == PeerConnectionState::Connected)
            .map(|(id, _)| id.clone())
            .collect();
            
        drop(peers);
        
        let mut successful_sends = 0;
        
        for peer_id in connected_peers {
            if self.send_message(&peer_id, message.clone()).await.is_ok() {
                successful_sends += 1;
            }
        }
        
        Ok(successful_sends)
    }
    
    /// Banne un pair
    pub async fn ban_peer(&self, peer_id: &str, reason: &str) -> Result<()> {
        {
            let mut banned = self.banned_peers.write().await;
            banned.insert(peer_id.to_string());
        }
        
        // Déconnecter le pair s'il est connecté
        self.disconnect_peer(peer_id, reason).await?;
        
        warn!("Pair {} banni: {}", peer_id, reason);
        
        Ok(())
    }
    
    /// Débanne un pair
    pub async fn unban_peer(&self, peer_id: &str) -> Result<()> {
        let mut banned = self.banned_peers.write().await;
        banned.remove(peer_id);
        
        info!("Pair {} débanni", peer_id);
        
        Ok(())
    }
    
    /// Retourne la liste des pairs connectés
    pub async fn get_connected_peers(&self) -> Vec<PeerInfo> {
        let peers = self.peers.read().await;
        
        peers.iter()
            .filter(|(_, peer)| peer.state == PeerConnectionState::Connected)
            .map(|(_, peer)| peer.info.clone())
            .collect()
    }
    
    /// Retourne l'ID du pair local
    pub fn get_local_peer_id(&self) -> &str {
        &self.local_peer_id
    }
    
    /// Clone du PeerManager, utile pour les fermetures
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            peers: self.peers.clone(),
            banned_peers: self.banned_peers.clone(),
            message_sender: self.message_sender.clone(),
            message_receiver: self.message_receiver.clone(),
            local_peer_id: self.local_peer_id.clone(),
        }
    }
}

impl Clone for PeerManagerConfig {
    fn clone(&self) -> Self {
        Self {
            max_peers: self.max_peers,
            min_peers: self.min_peers,
            peer_timeout_secs: self.peer_timeout_secs,
            reconnect_interval_secs: self.reconnect_interval_secs,
            discovery_interval_secs: self.discovery_interval_secs,
            bootstrap_peers: self.bootstrap_peers.clone(),
        }
    }
}
