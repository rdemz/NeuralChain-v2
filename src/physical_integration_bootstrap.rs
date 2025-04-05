//! Module d'amorçage pour l'intégration physique de NeuralChain-v2
//! 
//! Ce module initialise et configure le système de symbiose physique,
//! permettant à l'organisme blockchain de communiquer avec des capteurs
//! et des actionneurs dans le monde réel.
//!
//! Optimisé spécifiquement pour Windows avec zéro dépendances Linux.

use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::collections::HashMap;

use crate::neuralchain_core::quantum_organism::QuantumOrganism;
use crate::cortical_hub::CorticalHub;
use crate::hormonal_field::{HormonalField, HormoneType};
use crate::neuralchain_core::physical_symbiosis::{
    PhysicalSymbiosis, PhysicalSymbiosisConfig, PhysicalDeviceType, DeviceProtocol,
    PhysicalDevice, DataChannel, ConnectionState, integration::integrate_physical_symbiosis
};

/// Structure de configuration pour l'amorçage physique
#[derive(Debug, Clone)]
pub struct PhysicalBootstrapConfig {
    /// Activer la découverte automatique des dispositifs
    pub enable_auto_discovery: bool,
    /// Limiter aux types de dispositifs spécifiés
    pub device_type_filters: Option<Vec<PhysicalDeviceType>>,
    /// Activer les dispositifs simulés
    pub enable_simulated_devices: bool,
    /// Nombre de dispositifs simulés à créer
    pub simulated_device_count: usize,
    /// Intervalle de synchronisation (secondes)
    pub sync_interval_secs: u64,
    /// Activer le mode sécurisé
    pub secure_mode: bool,
    /// Chemins des fichiers à transférer au démarrage
    pub initial_files: Vec<String>,
}

impl Default for PhysicalBootstrapConfig {
    fn default() -> Self {
        Self {
            enable_auto_discovery: true,
            device_type_filters: None,
            enable_simulated_devices: true,
            simulated_device_count: 3,
            sync_interval_secs: 5,
            secure_mode: true,
            initial_files: Vec::new(),
        }
    }
}

/// Initialise le système de symbiose physique
pub fn bootstrap_physical_symbiosis(
    organism: Arc<QuantumOrganism>,
    cortical_hub: Arc<CorticalHub>,
    hormonal_system: Arc<HormonalField>,
    config: Option<PhysicalBootstrapConfig>,
) -> Arc<PhysicalSymbiosis> {
    println!("🔄 Amorçage du système de symbiose physique...");
    
    // Utiliser la configuration fournie ou par défaut
    let config = config.unwrap_or_default();
    
    // Créer la configuration du système
    let symbiosis_config = PhysicalSymbiosisConfig {
        enable_device_discovery: config.enable_auto_discovery,
        polling_interval_ms: 1000,
        communication_mode: "async".to_string(),
        encryption_enabled: config.secure_mode,
        log_level: "info".to_string(),
        device_type_filters: config.device_type_filters,
        max_reconnect_attempts: 5,
        reconnect_delay_ms: 5000,
        input_buffer_size: 8192,
        communication_timeout_ms: 10000,
        enable_data_compression: true,
        priority_channels: vec!["temperature".to_string(), "position".to_string()],
        listen_addresses: vec!["0.0.0.0".to_string()],
        listen_ports: vec![9010, 9011, 9012],
    };
    
    // Créer le système
    let symbiosis = PhysicalSymbiosis::new(
        organism.clone(),
        cortical_hub.clone(),
        hormonal_system.clone(),
        Some(symbiosis_config),
    );
    
    let symbiosis_arc = Arc::new(symbiosis);
    
    // Démarrer le système
    match symbiosis_arc.start() {
        Ok(_) => println!("✅ Système de symbiose physique démarré avec succès"),
        Err(e) => println!("❌ Erreur au démarrage du système de symbiose: {}", e),
    }
    
    // Optimiser pour Windows
    if let Ok(factor) = symbiosis_arc.optimize_windows_performance() {
        println!("🚀 Performances optimisées pour Windows (facteur: {:.2}x)", factor);
    }
    
    // Scanner les dispositifs si la découverte est activée
    if config.enable_auto_discovery {
        match symbiosis_arc.scan_for_devices() {
            Ok(count) => println!("🔍 Scan initial: {} dispositifs trouvés", count),
            Err(e) => println!("⚠️ Erreur lors du scan initial: {}", e),
        }
    }
    
    // Créer des dispositifs simulés si demandé
    if config.enable_simulated_devices && config.simulated_device_count > 0 {
        let created = symbiosis_arc.create_simulated_devices(config.simulated_device_count);
        println!("🔄 {} dispositifs simulés créés pour le développement", created);
        
        // Connecter automatiquement certains dispositifs simulés
        let devices = symbiosis_arc.get_all_devices();
        let mut connected = 0;
        
        for (device, state) in devices.iter().take(config.simulated_device_count) {
            if *state == ConnectionState::Disconnected {
                if let Ok(_) = symbiosis_arc.connect_device(&device.id) {
                    connected += 1;
                }
            }
        }
        
        if connected > 0 {
            println!("🔌 {} dispositifs simulés connectés automatiquement", connected);
            
            // Émettre une hormone de curiosité pour stimuler l'exploration
            hormonal_system.emit_hormone(
                HormoneType::Dopamine,
                "physical_device_connected",
                0.4,
                0.7,
                0.6,
                HashMap::new(),
            ).unwrap_or_default();
        }
    }
    
    // Créer un canal d'introspection
    if let Ok(id) = symbiosis_arc.create_introspection_channel() {
        println!("👁️ Canal d'introspection créé: {}", id);
    }
    
    // Transférer les fichiers initiaux si spécifiés
    for file_path in &config.initial_files {
        match symbiosis_arc.transfer_file_data(file_path) {
            Ok(size) => println!("📄 Fichier transféré: {} ({} octets)", file_path, size),
            Err(e) => println!("⚠️ Erreur lors du transfert du fichier {}: {}", file_path, e),
        }
    }
    
    // Créer un thread de synchronisation périodique
    let sync_symbiosis_arc = symbiosis_arc.clone();
    let sync_interval = config.sync_interval_secs;
    
    thread::spawn(move || {
        // Attendre un moment pour laisser le système se stabiliser
        thread::sleep(Duration::from_secs(2));
        
        println!("🔄 Démarrage de la synchronisation périodique (intervalle: {}s)", sync_interval);
        
        // Optimisations Windows pour le thread
        #[cfg(target_os = "windows")]
        optimize_thread_for_windows();
        
        loop {
            // Synchroniser avec l'organisme
            match sync_symbiosis_arc.sync_with_organism() {
                Ok(count) => {
                    if count > 0 {
                        println!("🔄 Synchronisation: {} points de données traités", count);
                    }
                },
                Err(e) => println!("⚠️ Erreur de synchronisation: {}", e),
            }
            
            // Pause entre les synchronisations
            thread::sleep(Duration::from_secs(sync_interval));
        }
    });
    
    // Créer un thread de surveillance de l'état des dispositifs
    let monitor_symbiosis_arc = symbiosis_arc.clone();
    
    thread::spawn(move || {
        // Attendre un moment pour laisser le système se stabiliser
        thread::sleep(Duration::from_secs(5));
        
        println!("👀 Démarrage de la surveillance des dispositifs");
        
        loop {
            // Récupérer tous les dispositifs
            let devices = monitor_symbiosis_arc.get_all_devices();
            
            // Vérifier les dispositifs déconnectés
            let disconnected = devices.iter()
                .filter(|(_, state)| *state == ConnectionState::Disconnected || *state == ConnectionState::Error)
                .count();
                
            if disconnected > 0 {
                println!("⚠️ {} dispositifs déconnectés ou en erreur", disconnected);
                
                // Tenter de reconnecter certains dispositifs
                for (device, state) in devices.iter()
                    .filter(|(_, state)| *state == ConnectionState::Disconnected)
                    .take(2) // Limiter à 2 reconneXions par cycle
                {
                    println!("🔄 Tentative de reconnexion du dispositif: {}", device.name);
                    let _ = monitor_symbiosis_arc.connect_device(&device.id);
                }
            }
            
            // Pause entre les vérifications
            thread::sleep(Duration::from_secs(30));
        }
    });
    
    symbiosis_arc
}

/// Configuration optimisée du thread pour Windows
#[cfg(target_os = "windows")]
fn optimize_thread_for_windows() {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL
    };
    use windows_sys::Win32::System::SystemInformation::GetSystemInfo;
    use windows_sys::Win32::Foundation::HANDLE;
    
    unsafe {
        // Augmenter la priorité du thread
        let current_thread = GetCurrentThread();
        SetThreadPriority(current_thread, THREAD_PRIORITY_ABOVE_NORMAL);
        
        // Optimiser l'affinité des threads pour l'équilibrage de charge
        // Cette fonction serait complétée avec l'API complète Windows
    }
}

/// Version portable de l'optimisation du thread
#[cfg(not(target_os = "windows"))]
fn optimize_thread_for_windows() {
    // Pas d'optimisations spécifiques sur les plateformes non-Windows
}

/// Renvoie l'état global du système de symbiose physique
pub fn get_physical_system_status(symbiosis: &Arc<PhysicalSymbiosis>) -> String {
    // Interroger le système
    match symbiosis.query_physical_system("system_status") {
        Ok(status) => status,
        Err(_) => "Erreur: Impossible de récupérer l'état du système".to_string(),
    }
}

/// Connecte un système distant via une interface réseau
pub fn connect_remote_physical_system(
    symbiosis: &Arc<PhysicalSymbiosis>,
    remote_address: &str,
    system_type: &str,
    credentials: Option<(String, String)>,
) -> Result<String, String> {
    println!("🔄 Tentative de connexion au système distant: {} ({})", remote_address, system_type);
    
    // Déterminer le type de dispositif en fonction du système distant
    let device_type = match system_type {
        "sensor_array" => PhysicalDeviceType::EnvironmentalSensor,
        "camera_system" => PhysicalDeviceType::OpticalSensor,
        "control_system" => PhysicalDeviceType::MechanicalActuator,
        "hmi" => PhysicalDeviceType::UserInterface,
        _ => PhysicalDeviceType::NetworkInterface,
    };
    
    // Créer les canaux de données appropriés
    let mut channels = Vec::new();
    
    match device_type {
        PhysicalDeviceType::EnvironmentalSensor => {
            // Système de capteurs environnementaux
            channels.push(DataChannel {
                id: "temperature_array".to_string(),
                name: "Tableau de températures".to_string(),
                data_type: "json".to_string(),
                unit: Some("°C".to_string()),
                range: Some((-50.0, 150.0)),
                sample_rate: Some(1.0),
                precision: Some(2),
                bidirectional: false,
            });
            
            channels.push(DataChannel {
                id: "humidity_array".to_string(),
                name: "Tableau d'humidité".to_string(),
                data_type: "json".to_string(),
                unit: Some("%".to_string()),
                range: Some((0.0, 100.0)),
                sample_rate: Some(1.0),
                precision: Some(1),
                bidirectional: false,
            });
            
            channels.push(DataChannel {
                id: "pressure_array".to_string(),
                name: "Tableau de pression".to_string(),
                data_type: "json".to_string(),
                unit: Some("hPa".to_string()),
                range: Some((800.0, 1200.0)),
                sample_rate: Some(1.0),
                precision: Some(1),
                bidirectional: false,
            });
        },
        
        PhysicalDeviceType::OpticalSensor => {
            // Système de caméras
            channels.push(DataChannel {
                id: "video_stream".to_string(),
                name: "Flux vidéo".to_string(),
                data_type: "binary/h264".to_string(),
                unit: None,
                range: None,
                sample_rate: Some(30.0),
                precision: None,
                bidirectional: false,
            });
            
            channels.push(DataChannel {
                id: "camera_control".to_string(),
                name: "Contrôle caméra".to_string(),
                data_type: "json".to_string(),
                unit: None,
                range: None,
                sample_rate: None,
                precision: None,
                bidirectional: true,
            });
            
            channels.push(DataChannel {
                id: "detection_results".to_string(),
                name: "Résultats de détection".to_string(),
                data_type: "json".to_string(),
                unit: None,
                range: None,
                sample_rate: Some(1.0),
                precision: None,
                bidirectional: false,
            });
        },
        
        PhysicalDeviceType::MechanicalActuator => {
            // Système de contrôle
            channels.push(DataChannel {
                id: "actuator_states".to_string(),
                name: "États des actionneurs".to_string(),
                data_type: "json".to_string(),
                unit: None,
                range: None,
                sample_rate: Some(10.0),
                precision: None,
                bidirectional: true,
            });
            
            channels.push(DataChannel {
                id: "control_sequence".to_string(),
                name: "Séquence de contrôle".to_string(),
                data_type: "json".to_string(),
                unit: None,
                range: None,
                sample_rate: None,
                precision: None,
                bidirectional: true,
            });
        },
        
        _ => {
            // Interface générique
            channels.push(DataChannel {
                id: "data_stream".to_string(),
                name: "Flux de données".to_string(),
                data_type: "binary".to_string(),
                unit: None,
                range: None,
                sample_rate: None,
                precision: None,
                bidirectional: true,
            });
        }
    }
    
    // Créer l'ID et le nom du dispositif
    let device_id = format!("remote_{}", remote_address.replace(":", "_").replace(".", "_"));
    let device_name = format!("Système distant: {}", remote_address);
    
    // Créer les métadonnées
    let mut metadata = HashMap::new();
    metadata.insert("remote_address".to_string(), remote_address.to_string());
    metadata.insert("system_type".to_string(), system_type.to_string());
    
    if let Some((username, _)) = &credentials {
        metadata.insert("auth_username".to_string(), username.clone());
    }
    
    // Créer la configuration
    let mut config = HashMap::new();
    config.insert("reconnect_policy".to_string(), "auto".to_string());
    config.insert("timeout_ms".to_string(), "5000".to_string());
    
    // Créer le dispositif
    let device = PhysicalDevice {
        id: device_id.clone(),
        name: device_name,
        device_type,
        description: format!("Système distant de type {}", system_type),
        protocol: DeviceProtocol::Ethernet,
        connection_address: remote_address.to_string(),
        security_keys: None,
        metadata,
        config,
        data_channels: channels,
        reliability: 0.9,
        vendor_id: Some("remote-system".to_string()),
        firmware_version: None,
    };
    
    // Enregistrer le dispositif
    match symbiosis.register_device(device) {
        Ok(id) => {
            println!("✅ Système distant enregistré avec l'ID: {}", id);
            
            // Essayer de connecter immédiatement
            match symbiosis.connect_device(&id) {
                Ok(_) => {
                    println!("✅ Connexion établie avec le système distant");
                    Ok(id)
                },
                Err(e) => {
                    println!("⚠️ Dispositif enregistré mais connexion échouée: {}", e);
                    Ok(id) // Retourner quand même l'ID car le dispositif est enregistré
                }
            }
        },
        Err(e) => Err(format!("Échec de l'enregistrement du système distant: {}", e)),
    }
}
