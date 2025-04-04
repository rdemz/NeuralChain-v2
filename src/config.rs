use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    // Paramètres généraux
    pub network_id: u32,              // 1 = mainnet, 2 = testnet
    pub data_dir: String,             // Répertoire des données
    pub log_level: LogLevel,          // Niveau de journalisation
    
    // Paramètres réseau
    pub p2p_config: P2PConfig,
    pub api_port: u16,                // Port pour l'API REST
    pub prometheus_port: u16,         // Port pour métriques Prometheus
    
    // Paramètres consensus
    pub consensus_difficulty: u32,    // Difficulté de base pour PoW
    pub blockchain_save_interval: u64, // Intervalle de sauvegarde en secondes
    
    // Paramètres mempool
    pub mempool_max_size: usize,      // Capacité maximale du mempool
    pub mempool_max_age_secs: u64,    // Âge maximum d'une transaction en mempool
    
    // Paramètres mining
    pub mining_queue_capacity: usize, // Capacité de la file de blocs minés
    pub mining_interval_nanos: u64,   // Intervalle entre tentatives de mining en nanosecondes
    pub mining_batch_size: usize,     // Taille du lot de nonces à essayer
    pub mining_difficulty_adjustment: f64, // Facteur d'ajustement de difficulté
    pub mining_thread_priority: i32,  // Priorité des threads de mining
    
    // Paramètres DeFi
    pub defi_enabled: bool,           // Activation des fonctionnalités DeFi
    pub defi_liquidity_fee: f64,      // Frais pour les pools de liquidité (%)
    
    // Paramètres gouvernance
    pub governance_enabled: bool,     // Activation de la gouvernance DAO
    pub governance_voting_period_days: u32, // Durée de la période de vote en jours
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct P2PConfig {
    pub listen_addr: String,          // Adresse d'écoute pour P2P
    pub port: u16,                    // Port d'écoute P2P
    pub max_connections: usize,       // Nombre maximum de connexions
    pub bootstrap_peers: Vec<String>, // Pairs de bootstrap
    pub enable_mdns: bool,            // Découverte via mDNS
    pub enable_kad: bool,             // Utilisation de Kademlia
    pub connection_timeout_secs: u64, // Timeout de connexion en secondes
    pub ping_interval_secs: u64,      // Intervalle ping/pong en secondes
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Config {
    /// Charge la configuration depuis un fichier ou crée une configuration par défaut
    pub fn load_or_default() -> anyhow::Result<Self> {
        let config_path = Path::new("neuralchain.toml");
        
        if config_path.exists() {
            let config_str = fs::read_to_string(config_path)?;
            let config: Config = toml::from_str(&config_str)?;
            Ok(config)
        } else {
            let default_config = Config::default();
            let config_str = toml::to_string_pretty(&default_config)?;
            fs::write(config_path, config_str)?;
            Ok(default_config)
        }
    }
    
    /// Sauvegarde la configuration dans un fichier
    pub fn save(&self) -> anyhow::Result<()> {
        let config_path = Path::new("neuralchain.toml");
        let config_str = toml::to_string_pretty(self)?;
        fs::write(config_path, config_str)?;
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            network_id: 2, // testnet par défaut
            data_dir: "data".to_string(),
            log_level: LogLevel::Info,
            
            p2p_config: P2PConfig::default(),
            api_port: 8000,
            prometheus_port: 9000,
            
            consensus_difficulty: 4,
            blockchain_save_interval: 300, // 5 minutes
            
            mempool_max_size: 100000,
            mempool_max_age_secs: 3600, // 1 heure
            
            mining_queue_capacity: 1000,
            mining_interval_nanos: 100, // 100ns
            mining_batch_size: 10000,
            mining_difficulty_adjustment: 1.0,
            mining_thread_priority: 10,
            
            defi_enabled: true,
            defi_liquidity_fee: 0.3, // 0.3%
            
            governance_enabled: true,
            governance_voting_period_days: 7,
        }
    }
}

impl Default for P2PConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0".to_string(),
            port: 7000,
            max_connections: 100,
            bootstrap_peers: vec![
                "/dns4/bootstrap-1.neuralchain.network/tcp/7000/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt".to_string(),
                "/dns4/bootstrap-2.neuralchain.network/tcp/7000/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb".to_string(),
            ],
            enable_mdns: true,
            enable_kad: true,
            connection_timeout_secs: 10,
            ping_interval_secs: 60,
        }
    }
}
