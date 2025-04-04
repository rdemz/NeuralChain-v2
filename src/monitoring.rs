use std::net::SocketAddr;
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use once_cell::sync::Lazy;
use tracing::{info, Level};
use tracing_subscriber::{EnvFilter, filter::LevelFilter, prelude::*, Registry};

use crate::config::LogLevel;

// Métriques globales
pub static ACTIVE_CONNECTIONS: Lazy<metrics::Counter> = Lazy::new(|| {
    metrics::counter!("neuralchain_active_connections", "Nombre de connexions P2P actives")
});

pub static BLOCK_MINING_TIME: Lazy<metrics::Histogram> = Lazy::new(|| {
    metrics::histogram!("neuralchain_block_mining_time", "Temps de mining d'un bloc en ms")
});

pub static MEMPOOL_SIZE: Lazy<metrics::Gauge> = Lazy::new(|| {
    metrics::gauge!("neuralchain_mempool_size", "Nombre de transactions en attente dans le mempool")
});

pub static NETWORK_MESSAGES: Lazy<metrics::Counter> = Lazy::new(|| {
    metrics::counter!("neuralchain_network_messages", "Nombre de messages réseau")
});

pub static TRANSACTION_VALIDATION_TIME: Lazy<metrics::Histogram> = Lazy::new(|| {
    metrics::histogram!("neuralchain_transaction_validation_time", "Temps de validation d'une transaction en µs")
});

/// Initialiser l'exportateur Prometheus sur le port spécifié
pub fn init_prometheus(port: u16) -> PrometheusHandle {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    
    let builder = PrometheusBuilder::new()
        .with_http_listener(addr)
        .add_global_label("service", "neuralchain")
        .add_global_label("version", env!("CARGO_PKG_VERSION"))
        .idle_timeout(
            Matcher::Prefix("neuralchain_".to_string()),
            Some(std::time::Duration::from_secs(10))
        )
        .install_recorder()
        .expect("Failed to install Prometheus recorder");
    
    info!("Exportateur Prometheus lancé sur http://{}/metrics", addr);
    builder
}

/// Convertir LogLevel en Level de tracing
fn log_level_to_tracing_level(level: &LogLevel) -> Level {
    match level {
        LogLevel::Error => Level::ERROR,
        LogLevel::Warn => Level::WARN,
        LogLevel::Info => Level::INFO,
        LogLevel::Debug => Level::DEBUG,
        LogLevel::Trace => Level::TRACE,
    }
}

/// Initialiser la journalisation
pub fn init_tracing(log_level: &LogLevel) {
    let level = log_level_to_tracing_level(log_level);
    
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::from_level(level).into())
        .from_env_lossy();
    
    // Configuration du formatage
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_timer(tracing_subscriber::fmt::time::time());
    
    // Assembler les couches
    let subscriber = Registry::default()
        .with(env_filter)
        .with(fmt_layer);
    
    // Définir comme subscriber global
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global tracing subscriber");
    
    info!("Journalisation initialisée au niveau {}", level);
}

/// Initialiser le monitoring complet (tracing + Prometheus)
pub fn init_monitoring(prometheus_port: u16) -> PrometheusHandle {
    // Initialiser la journalisation avec le niveau par défaut
    init_tracing(&crate::config::LogLevel::Info);
    
    // Initialiser Prometheus
    let handle = init_prometheus(prometheus_port);
    
    info!("Système de monitoring initialisé");
    handle
}
